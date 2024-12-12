import torch
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
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
class T5Config:
    model_name: str
    model_path: str
    token_response_marker: str = "<sys>"
    load_in_8bit: bool = False  # T5 typically uses 8-bit quantization
    torch_dtype: torch.dtype = torch.float16
    max_source_length: int = 512
    max_target_length: int = 384
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class T5Evaluator:
    """Evaluator for T5-based models"""
    
    def __init__(self, config: T5Config):
        self.config = config
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load T5 model and tokenizer"""
        logging.info(f"Loading T5 model: {self.config.model_name} from {self.config.model_path}")
        
        # Configure quantization if needed
        quantization_config = None
        if self.config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )

        # Load model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=self.config.torch_dtype
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        
        return model, tokenizer
    
    def preprocess_batch(self, batch: Dict) -> Dict:
        """Preprocess batch for T5 model"""
        source_texts = batch['text']
        
        # Tokenize inputs
        inputs = self.tokenizer(
            source_texts,
            max_length=self.config.max_source_length,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        ).to(self.config.device)
        
        return inputs
    
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
        """Generate responses using beam search for T5"""
        generation_config = {
            "max_length": self.config.max_target_length,
            "num_beams": num_beams,
            "num_return_sequences": num_return_sequences,
            "no_repeat_ngram_size": 3,
            "length_penalty": 0.6,
            "early_stopping": True,  # T5 specific
            **kwargs
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
                    response = self.decode_response(sequence)
                    batch_responses.append(response)
                responses.append(batch_responses)
            return responses
        else:
            responses = [self.decode_response(ids) for ids in outputs]
            return responses

    def self_consistency_generation(self, 
                                  input_ids: torch.Tensor,
                                  attention_mask: torch.Tensor,
                                  num_samples: int = 5,
                                  temperature: float = 0.7,
                                  **kwargs) -> List[str]:
        """Generate responses using self-consistency sampling for T5"""
        generation_config = {
            "max_length": self.config.max_target_length,
            "do_sample": True,
            "num_return_sequences": num_samples,
            "temperature": temperature,
            "no_repeat_ngram_size": 3,
            "top_p": 0.9,
            "early_stopping": True,
            **kwargs
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
                response = self.decode_response(sequence)
                candidates.append(response)
            
            final_response = self._vote_for_response(candidates)
            final_responses.append(final_response)
            
        return final_responses
    
    def _vote_for_response(self, candidates: List[str]) -> str:
        """Select the best response from candidates"""
        from collections import Counter
        return Counter(candidates).most_common(1)[0][0]
    
    def evaluate(self, 
                test_data: str,
                decoding_strategy: str = "beam_search",
                batch_size: int = 1,
                save_results: bool = True,
                **decoding_params) -> Dict:
        """Evaluate T5 model using specified decoding strategy"""
        logging.info(f"Starting T5 evaluation with {decoding_strategy}")
        
        # Load test dataset
        test_dataset = CustomMwozDataset(
            self.tokenizer, 
            data_filename=test_data,
            model_type='t5',  # Specify T5 type
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
            # Preprocess batch
            inputs = self.preprocess_batch(batch)
            
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
            output = []
            for idx, resp in enumerate(all_responses):
                output.append({
                    'uuid': test_dataset[idx]['uuid'],
                    'turn_id': test_dataset[idx]['turn_id'],
                    'ground_resp': test_dataset[idx]['output_seq'],
                    'prediction': resp,
                })
                
            save_path = f"{constants.TEST_RESULT_FILE[self.config.model_name]}.{decoding_strategy}"
            with open(save_path, 'w') as f:
                json.dump(output, f, indent=2)
            
        return results

def main():
    # Example T5 configuration
    t5_config = T5Config(
        model_name="t5-response-pred",
        model_path=constants.MERGED_MODEL['t5-response-pred'],
        max_source_length=512,
        max_target_length=384
    )
    
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
    
    # Initialize evaluator
    evaluator = T5Evaluator(t5_config)
    
    # Run evaluations
    results = {}
    
    # Beam search evaluation
    results["beam_search"] = evaluator.evaluate(
        f'{constants.DATA_DIR}test.json',
        decoding_strategy="beam_search",
        **eval_params["beam_search"]
    )
    
    # Self-consistency evaluation
    results["self_consistency"] = evaluator.evaluate(
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