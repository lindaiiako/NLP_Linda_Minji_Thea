import os
import torch
import numpy as np
import pandas as pd
import json
from transformers import set_seed, GenerationConfig
from common.mwoz_data import CustomMwozDataset
from common import constants
from datasets import Dataset
from common.utils import format_train_for_llama, format_test_for_llama
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from common.utils import compute_prediction_scores

#from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


# Set for reproducibility
np.random.seed(constants.SEED)
torch.manual_seed(constants.SEED)
set_seed(constants.SEED)

# Push to GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"    # Temporary setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LlamaTrainer():
    # Loads Llama tokenizer and model
    def __init__(self):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(constants.LLAMA_MODEL_ID, model_max_length=None)
        #self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype="float16", 
            bnb_4bit_use_double_quant=True
            )        
        self.model = AutoModelForCausalLM.from_pretrained(constants.LLAMA_MODEL_ID, 
                                                          quantization_config=bnb_config)

        self.model.config.use_cache=False

        #self.model.config.pad_token_id = self.tokenizer.pad_token_id
        #self.model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=8)

        response_template = " ### Response:"
        self.collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)

    
    def evaluate(self, test_set, save_results, result_path, metric_key_prefix):
        max_new_tokens = 128
        self.model.eval()
        
        # Apply prompt
        dataloader = DataLoader(test_set,
                                batch_size=8,
                                #collate_fn=self.collator,
                                sampler=torch.utils.data.SequentialSampler(test_set),
        )

        responses = []
        for batch in dataloader:
            formatted_inputs = [format_test_for_llama(v) for k, v in batch.items()]
            inputs = self.tokenizer(formatted_inputs, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, 
                                            max_new_tokens=max_new_tokens,
                                            do_sample=False,
                                            num_beams=1
                                            )
            
            responses.extend(self.tokenizer.batch_decode(outputs, skip_special_tokens=True))

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
                output[-1]['uuid'] = test_set.raw_dataset[idx]['uuid']
                output[-1]['turn_id'] = test_set.raw_dataset[idx]['turn_id']
                tetypes = resp.split('|')
                tetypes = [x.strip() for x in tetypes if '[no entity]' != x]
                output[-1]['prediction'] = tetypes

            with open(result_path, 'w') as f:
                json.dump(output, f, indent=2)

        return metrics
    

    # Main training procedure
    def train(self):
        train_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}train.json', model_type='llama', mode='train').data
        validation_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}valid.json', model_type='llama', mode='eval').data

        peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

        # Init training args
        training_args = TrainingArguments(
            output_dir=constants.LLAMA_FT_MODEL_NAME,
            gradient_accumulation_steps=4,
            per_device_train_batch_size=3,
            per_device_eval_batch_size=4,
            optim="paged_adamw_32bit",
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            logging_steps=10,
            num_train_epochs=3,
            max_steps=10, #250,
            fp16=True,
            push_to_hub=False,
            seed=constants.SEED,
            report_to='tensorboard',
            run_name='llama_exp',
            load_best_model_at_end=True, 
            eval_strategy='steps',
            save_strategy='steps', 
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            peft_config=peft_config,
            args=training_args,
            train_dataset=Dataset.from_list(train_set), 
            eval_dataset=Dataset.from_list(validation_set),
            max_seq_length=5000,
            formatting_func=format_train_for_llama,
            data_collator=self.collator,
            packing=False,
        )

        trainer.train()

        # Get test performance
        test_set = Dataset.from_list(CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}test.json', model_type='llama', mode='infer').data)
        test_results = self.evaluate(test_set, save_results=True, result_path=constants.LLAMA_TEST_RESULT_FILE, metric_key_prefix='test')
        print(test_results)