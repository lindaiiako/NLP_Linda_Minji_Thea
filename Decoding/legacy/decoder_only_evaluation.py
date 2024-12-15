import torch
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)

from common.mwoz_data import CustomMwozDataset
from synctod.text_process import preprocess_text
from synctod.metrics import compute_metrics
from common import constants
from tqdm import tqdm

@dataclass
class ModelConfig:
    model_name: str
    model_path: str
    token_response_marker: str = "<sys>"  # Marker to split response
    load_in_4bit: bool = True
    use_flash_attention: bool = False
    torch_dtype: torch.dtype = torch.float16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class LLMEvaluator:
    """Generic evaluator for language models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer based on configuration"""
        logging.info(f"Loading model: {self.config.model_name} from {self.config.model_path}")
        
        # Configure quantization if needed
        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # Model specific kwargs
        model_kwargs = {
            "quantization_config": quantization_config,
            "device_map": "auto",
            "torch_dtype": self.config.torch_dtype,
        }
        
        # Add flash attention for supported models
        if self.config.use_flash_attention:
            if "llama" in self.config.model_name.lower():
                model_kwargs["attn_implementation"] = "flash_attention_2"
            elif "mistral" in self.config.model_name.lower():
                model_kwargs["use_flash_attention_2"] = True
                
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            **model_kwargs
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        
        # Handle padding token based on model type
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        
        return model, tokenizer
    
    def decode_response(self, output_ids: torch.Tensor) -> str:
        """Decode model output and extract response"""
        decoded = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        if self.config.token_response_marker in decoded:
            response = decoded.split(self.config.token_response_marker)[1].strip()
        else:
            response = decoded.strip()
        return preprocess_text(response)
    
    def beam_search_generation(self, 
                             input_ids: torch.Tensor,
                             attention_mask: torch.Tensor,
                             num_beams: int = 5,
                             num_return_sequences: int = 1,
                             **kwargs) -> List[str]:
        """Generate responses using beam search"""
        generation_config = {
            "max_new_tokens": 384,
            "num_beams": num_beams,
            "num_return_sequences": num_return_sequences,
            "no_repeat_ngram_size": 3,
            "length_penalty": 0.6,
            "temperature": 1.0,
            **kwargs  # Allow overriding defaults
        }
        
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        
        # Handle multiple sequences
        if num_return_sequences > 1:
            outputs = outputs.reshape(input_ids.shape[0], num_return_sequences, -1)
            responses = []
            for batch_outputs in outputs:
                batch_responses = []
                for sequence in batch_outputs:
                    generated_ids = sequence[input_ids.shape[1]:]
                    response = self.decode_response(generated_ids)
                    batch_responses.append(response)
                responses.append(batch_responses)
            return responses
        else:
            generated_ids = outputs[:, input_ids.shape[1]:]
            responses = [self.decode_response(ids) for ids in generated_ids]
            return responses

    def self_consistency_generation(self, 
                                  input_ids: torch.Tensor,
                                  attention_mask: torch.Tensor,
                                  num_samples: int = 5,
                                  temperature: float = 0.7,
                                  **kwargs) -> List[str]:
        """Generate responses using self-consistency sampling"""
        generation_config = {
            "max_new_tokens": 384,
            "do_sample": True,
            "num_return_sequences": num_samples,
            "temperature": temperature,
            "no_repeat_ngram_size": 3,
            "top_p": 0.9,
            **kwargs  # Allow overriding defaults
        }
        
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        
        outputs = outputs.reshape(input_ids.shape[0], num_samples, -1)
        final_responses = []
        
        for batch_outputs in outputs:
            candidates = []
            for sequence in batch_outputs:
                generated_ids = sequence[input_ids.shape[1]:]
                response = self.decode_response(generated_ids)
                candidates.append(response)
            
            # Select final response using voting
            final_response = self._vote_for_response(candidates)
            final_responses.append(final_response)
            
        return final_responses
    
    def _vote_for_response(self, candidates: List[str], 
                          similarity_threshold: float = 0.8) -> str:
        """
        Select the best response from candidates using voting.
        Can be extended with more sophisticated selection methods.
        """
        from collections import Counter
        return Counter(candidates).most_common(1)[0][0]

    def evaluate(self, 
                test_data: str,
                decoding_strategy: str = "beam_search",
                batch_size: int = 1,
                save_results: bool = True,
                **decoding_params) -> Dict:
        """
        Evaluate the model using specified decoding strategy
        """
        logging.info(f"Starting evaluation with {decoding_strategy}")
        
        # Load test dataset
        test_dataset = CustomMwozDataset(
            self.tokenizer, 
            data_filename=test_data,
            model_type=self.config.model_name.lower(),  # Adapt based on model
            mode='infer',
            response_pred=True
        ).data
        
        dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        all_responses = []
        all_ground_truths = []
        
        for batch in tqdm(dataloader, desc=f"Evaluating with {decoding_strategy}"):
            formatted_prompts = batch['text']
            inputs = self.tokenizer(
                formatted_prompts,
                padding="longest",
                return_tensors="pt"
            ).to(self.config.device)
            
            if decoding_strategy == "beam_search":
                responses = self.beam_search_generation(
                    inputs['input_ids'],
                    inputs['attention_mask'],
                    **decoding_params
                )
            elif decoding_strategy == "self_consistency":
                responses = self.self_consistency_generation(
                    inputs['input_ids'],
                    inputs['attention_mask'],
                    **decoding_params
                )
            else:
                raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")
            
            all_responses.extend(responses)
            all_ground_truths.extend(batch['output_seq'])
        
        # Compute metrics
        results = compute_metrics(
            all_responses,
            all_ground_truths,
            'MultiWOZ',
            f'{constants.DATA_DIR}entities.json'
        )
        
        if save_results:
            # Save detailed results
            output = []
            for idx, resp in enumerate(all_responses):
                output.append({
                    'uuid': test_dataset[idx]['uuid'],
                    'turn_id': test_dataset[idx]['turn_id'],
                    'ground_resp': test_dataset[idx]['output_seq'],
                    'prediction': resp
                })
                
            save_path = f"{constants.TEST_RESULT_FILE[self.config.model_name]}.{decoding_strategy}"
            with open(save_path, 'w') as f:
                json.dump(output, f, indent=2)
            
        return results

def main():
    # Example configurations for different models
    model_configs = {
        "llama": ModelConfig(
            model_name="llama",
            model_path=constants.MERGED_MODEL['llama-response-pred'],
            use_flash_attention=True
        ),
        "mistral": ModelConfig(
            model_name="mistral",
            model_path=constants.MERGED_MODEL['mistral-response-pred'],
            use_flash_attention=True
        ),
        "gemma": ModelConfig(
            model_name="gemma",
            model_path=constants.MERGED_MODEL['gemma-response-pred'],
            use_flash_attention=False  # Gemma doesn't support flash attention
        )
    }
    
    # Evaluation parameters
    eval_params = {
        "beam_search": {
            "num_beams": 5,
            "num_return_sequences": 1,
            "length_penalty": 0.6
        },
        "self_consistency": {
            "num_samples": 5,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    
    # Run evaluation for each model
    results = {}
    for model_name, config in model_configs.items():
        logging.info(f"\nEvaluating {model_name}")
        evaluator = LLMEvaluator(config)
        
        # Evaluate with beam search
        results[f"{model_name}_beam"] = evaluator.evaluate(
            f'{constants.DATA_DIR}test.json',
            decoding_strategy="beam_search",
            **eval_params["beam_search"]
        )
        
        # Evaluate with self-consistency
        results[f"{model_name}_sc"] = evaluator.evaluate(
            f'{constants.DATA_DIR}test.json',
            decoding_strategy="self_consistency",
            **eval_params["self_consistency"]
        )
    
    # Print final results
    print("\nFinal Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()