
import torch
import numpy as np
import pandas as pd
import json
from transformers import set_seed
from common.mwoz_data import CustomMwozDataset
from common import constants
from datasets import Dataset
from torch.utils.data import DataLoader
from common.utils import compute_prediction_scores
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig


# Set for reproducibility
np.random.seed(constants.SEED)
torch.manual_seed(constants.SEED)
set_seed(constants.SEED)


class LlamaTrainer():
    # Loads Llama tokenizer and model
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(constants.LLAMA_MODEL_ID)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Double quantization config
        # https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
            )
        self.model = AutoModelForCausalLM.from_pretrained(constants.LLAMA_MODEL_ID, 
                                                          quantization_config=bnb_config)

        # No caching in training because every batch is processed independently
        self.model.config.use_cache=False
        
        # No tensor paralellism (single GPU setup)
        self.model.config.pretraining_tp = 1
        
        '''
        #https://huggingface.co/docs/transformers/en/model_doc/llama3
        # Add new pad token
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("<pad>")
        # Adjust embedding layer to accommodate the new padding token
        self.model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=8)
        # Set pad token in the model
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        '''

    # Evaluates entity type prediction using F1 scores for the test dataset
    def evaluate(self, test_set, save_results, result_path, metric_key_prefix):
        # Set to eval mode
        self.model.eval()

        # Iterate in order
        dataloader = DataLoader(test_set,
                                batch_size=16,
                                sampler=torch.utils.data.SequentialSampler(test_set),
        )

        responses = []
        probs = []

        for batch in dataloader:
            formatted_inputs = batch['text']
            llm_input = self.tokenizer(formatted_inputs, padding=True, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                llm_output = self.model.generate(**llm_input, 
                                            max_new_tokens=constants.MAX_NEW_TOKENS,
                                            do_sample=False,
                                            num_beams=1,
                                            return_dict_in_generate=True,
                                            output_scores=True,
                                            )
            
            raw_output = self.tokenizer.batch_decode(llm_output.sequences.to('cpu'), skip_special_tokens=True)
            resp = [item.split("<|im_start|>assistant\n")[1] for item in raw_output]
            responses.extend(resp)
            transition_scores = self.model.compute_transition_scores(llm_output.sequences, llm_output.scores, normalize_logits=True)
            probs.extend(np.exp(transition_scores.to('cpu').numpy()))

        # Compute scores
        prec, rec, f1, accuracy = compute_prediction_scores(responses, test_set)
        
        metrics = dict()
        metrics[f"{metric_key_prefix}_f1"] = f1
        metrics[f"{metric_key_prefix}_prec"] = prec
        metrics[f"{metric_key_prefix}_rec"] = rec
        metrics[f"{metric_key_prefix}_acc"] = accuracy

        print(f'Evaluation results: {json.dumps(metrics, indent=2)}')

        if save_results:
            output = []
            for idx, resp in enumerate(responses):
                output.append(dict())
                output[-1]['uuid'] = test_set[idx]['uuid']
                output[-1]['turn_id'] = test_set[idx]['turn_id']
                tetypes = resp.split('|')
                tetypes = [x.strip() for x in tetypes if '[no entity]' != x]
                output[-1]['prediction'] = tetypes
                # Get average score as proxy for confidence score
                output[-1]['confidence'] = str(np.mean(probs[idx]))     

            with open(result_path, 'w') as f:
                json.dump(output, f, indent=2)

        return metrics
    

    # Main training procedure
    def train(self):
        train_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}train.json', model_type='llama', mode='train').data
        validation_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}valid.json', model_type='llama', mode='eval').data

        #QLoRA config
        peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

        training_args =  SFTConfig(
            output_dir=constants.LLAMA_FT_MODEL_NAME,
            overwrite_output_dir=True,
            gradient_accumulation_steps=4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            group_by_length=True,
            optim="paged_adamw_32bit",
            learning_rate=2e-4,
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            num_train_epochs=8,
            warmup_ratio=0.03,
            fp16=False,
            bf16 = False,
            seed=constants.SEED,
            report_to='tensorboard',
            run_name='llama_exp',
            eval_strategy='steps',
            save_strategy='steps', 
            log_level='warning',
            logging_steps=10,
            save_steps=200,
            eval_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True, 
            max_seq_length=5000,
            packing=False
        )

        # DataCollatorForLanguageModeling (default) trains on both instructions and completions
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            peft_config=peft_config,
            args=training_args,
            train_dataset=Dataset.from_list(train_set), 
            eval_dataset=Dataset.from_list(validation_set),
            dataset_text_field="text",
        )

        trainer.train()

        # Get test performance
        test_set = Dataset.from_list(CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}test.json', model_type='llama', mode='infer').data)
        test_results = self.evaluate(test_set, save_results=True, result_path=constants.LLAMA_TEST_RESULT_FILE, metric_key_prefix='test')
        print(test_results)