import numpy as np
import json
import torch
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from common import constants
from common.mwoz_data import CustomMwozDataset
from common.utils import compute_prediction_scores
from peft import LoraConfig
from trl import SFTTrainer, setup_chat_format, SFTConfig, DataCollatorForCompletionOnlyLM


# Set for reproducibility
np.random.seed(constants.SEED)
torch.manual_seed(constants.SEED)
set_seed(constants.SEED)


class LlamaTrainer():
    # Loads Llama tokenizer and model
    def __init__(self):
        # Double quantization config
        # https://www.llama.com/docs/how-to-guides/fine-tuning/#qlora-fine-tuning
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
            )
                
        model = AutoModelForCausalLM.from_pretrained(constants.LLAMA_IT_MODEL_ID, 
                                                          quantization_config=bnb_config,
                                                          low_cpu_mem_usage=True,)
        tokenizer = AutoTokenizer.from_pretrained(constants.LLAMA_IT_MODEL_ID)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id
        
        # No caching in training because every batch is processed independently
        model.config.use_cache=False
        
        # No tensor paralellism (single GPU setup)
        model.config.pretraining_tp = 1

        # Setup model and tokenizer for chat-style format
        self.model, self.tokenizer = setup_chat_format(model, tokenizer)


    # Evaluates entity type prediction using F1 scores for the test dataset
    def evaluate(self, test_set, save_results, result_path, metric_key_prefix):
        # Set to eval mode
        self.model.eval()

        responses = []
        output = []

        for sample in test_set:
            formatted_prompt = sample['text']
            print("IN")
            print(formatted_prompt)

            tokenized_inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
            input_ids = tokenized_inputs["input_ids"]
            with torch.no_grad():
                outputs = self.model.generate(input_ids=input_ids,
                                            attention_mask=tokenized_inputs["attention_mask"], 
                                            max_new_tokens=constants.MAX_NEW_TOKENS,
                                            do_sample=False,
                                            num_beams=1,
                                            )
            
            # Get only the generated text (skip the input prompt)
            llm_response = outputs[0][input_ids.shape[-1]:]
            decoded_response = self.tokenizer.decode(llm_response, skip_special_tokens=True)
            print("OUT")
            print(decoded_response)
            responses.append(decoded_response)

            output.append(dict())
            output[-1]['uuid'] = sample['uuid']
            output[-1]['turn_id'] = sample['turn_id']
            tetypes = decoded_response.split('|')
            tetypes = [x.strip() for x in tetypes if '[no entity]' != x]
            output[-1]['prediction'] = tetypes
            
        # Compute scores
        prec, rec, f1, accuracy = compute_prediction_scores(responses, test_set)
        
        metrics = dict()
        metrics[f"{metric_key_prefix}_f1"] = f1
        metrics[f"{metric_key_prefix}_prec"] = prec
        metrics[f"{metric_key_prefix}_rec"] = rec
        metrics[f"{metric_key_prefix}_acc"] = accuracy

        print(f'Evaluation results: {json.dumps(metrics, indent=2)}')

        if save_results:
            with open(result_path, 'w') as f:
                json.dump(output, f, indent=2)

        return metrics


    # Main training procedure
    def train(self):
        train_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}train.json', model_type='llama_it', mode='train').data
        validation_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}valid.json', model_type='llama_it', mode='eval').data

        #QLoRA config
        peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

        training_args =  SFTConfig(
            output_dir=constants.LLAMA_FT_MODEL_NAME,
            overwrite_output_dir=True,
            gradient_accumulation_steps=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            group_by_length=True,
            optim="paged_adamw_32bit",
            learning_rate=2e-4,
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            num_train_epochs=1,
            #max_steps = 10,
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
            packing=False,
            dataset_text_field="text",
        )

        # Compute loss only on the completion made by the assistant
        #response_template = "<|im_start|>assistant"
        #collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=self.tokenizer, mlm=False)

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            peft_config=peft_config,
            args=training_args,
            train_dataset=Dataset.from_list(train_set), 
            eval_dataset=Dataset.from_list(validation_set),
        )

        trainer.train()

        # Get test performance
        test_set = Dataset.from_list(CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}test.json', model_type='llama_it', mode='infer').data)
        test_results = self.evaluate(test_set, save_results=True, result_path=constants.LLAMA_TEST_RESULT_FILE, metric_key_prefix='test')
        print(test_results)