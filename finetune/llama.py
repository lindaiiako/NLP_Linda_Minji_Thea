import os
import torch
import numpy as np
import pandas as pd
from transformers import set_seed, GenerationConfig
from common.mwoz_data import CustomMwozDataset
from common import constants
from datasets import Dataset

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
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        #self.tokenizer.pad_token = self.tokenizer.eos_token
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype="float16", 
            bnb_4bit_use_double_quant=True
            )        
        self.model = AutoModelForCausalLM.from_pretrained(constants.LLAMA_MODEL_ID, 
                                                          quantization_config=bnb_config)

        self.model.config.use_cache=False

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=8)


    def formatted_prompt(self, question):
        return f"### Question: {question}\n ### Answer: "
    
    def generate_response(self, user_input):
        prompt = self.formatted_prompt(user_input)
        inputs = self.tokenizer([prompt], return_tensors="pt")
        generation_config = GenerationConfig(penalty_alpha=0.6,do_sample = True,
            top_k=5,temperature=0.5,repetition_penalty=1.2,
            max_new_tokens=60,pad_token_id=self.tokenizer.eos_token_id
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = self.model.generate(**inputs, generation_config=generation_config)
        theresponse = (self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))


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
            save_strategy="epoch",
            logging_steps=10,
            num_train_epochs=3,
            max_steps=250,
            fp16=True,
            push_to_hub=False,
            seed=constants.SEED,
            report_to='tensorboard',
            run_name='llama_exp',
        )

        def formatting_prompts_func(example):
            output_texts = []
            for i in range(len(example['instruction'])):
                text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
                output_texts.append(text)
            return output_texts

        response_template = " ### Answer:"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            peft_config=peft_config,
            args=training_args,
            train_dataset=Dataset.from_list(train_set), 
            eval_dataset=Dataset.from_list(validation_set),
            max_seq_length=5000,
            formatting_func=formatting_prompts_func,
            data_collator=collator,
            packing=False,
        )

        trainer.train()

        # Get test performance
        test_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}test.json', model_type='t5', mode='eval')
        #test_results = trainer.evaluate(test_set, save_results=True, result_path=constants.LLAMA_TEST_RESULT_FILE)
        #print(test_results)