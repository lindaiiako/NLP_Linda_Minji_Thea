import numpy as np
import json
import torch
from torch.utils.data import DataLoader
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from common import constants
from common.mwoz_data import CustomMwozDataset
from common.utils import compute_prediction_scores
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, PeftModel

# Set for reproducibility
np.random.seed(constants.SEED)
torch.manual_seed(constants.SEED)
set_seed(constants.SEED)


class MySFTTrainer(SFTTrainer):
    
    # Decodes output with the tokenizer 
    def decode_responses(self, outputs, to_print=False):
        # Placeholder for decoded outputs
        preds = []

        responses = self.tokenizer.batch_decode(outputs.to('cpu'), skip_special_tokens=True)

        # Parse output
        for decoded_response in responses:
            et_preds = ""
            if "ASSISTANT:" in decoded_response:
                et_preds = decoded_response.split("ASSISTANT:")[1].strip()
            else:
                et_preds = decoded_response

            # Handle proper formats
            if 'no entity' in et_preds:
                et_preds = '[no entity]'
            else:
                et_preds = et_preds.replace('[', '').replace(']', '').strip()

            preds.append(et_preds)

            if to_print:
                print("RAW RESP")
                print(decoded_response)
                print("CLEAN OUT")
                print(et_preds)

        return preds


    # Obtains predictions with greedy decoding
    def get_preds(self, dataset, metric_key_prefix):
        model = self._wrap_model(self.model, training=False)
        # Temporarily set to eval mode
        model.eval()

        if metric_key_prefix == 'test':
            # For evaluation on test dataset, load raw dataset
            dataloader = DataLoader(dataset,
                                    batch_size=self.args.eval_batch_size,
                                    shuffle=False,
            )
        else:
            # For evaluation on eval dataset
            dataloader = DataLoader(dataset,
                                    batch_size=self.args.eval_batch_size,
                                    collate_fn=self.data_collator,
                                    sampler=torch.utils.data.SequentialSampler(dataset),
            )

        # Placeholder for outputs
        responses = []
        ground_truth_responses = []

        to_print = False

        for inputs in dataloader:
            if metric_key_prefix == 'test':
                formatted_prompts = inputs['text']
                tokenized_inputs = self.tokenizer(formatted_prompts, padding="longest", return_tensors="pt").to(self.model.device)
                # No need to recover inputs and outputs
                ground_truth_responses = dataset
                print("TEST IN")
                print(formatted_prompts)
                print("CORRECT OUT")
                print(inputs['output_seq'])
                to_print=True
            else:
                # In eval mode input is already tokenized
                # Recover input and output component
                input_components = []
                input_ids = inputs['input_ids']
                untokenized_inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
                for untokenized_text in untokenized_inputs:
                    marker = "<start_of_turn>model"
                    split_text = untokenized_text.split(marker)
                    input_component = split_text[0] + marker
                    input_components.append(input_component)
                    output_component = split_text[1].replace("ASSISTANT:", '').replace('<end_of_turn>','').replace('<eos>', '').strip()
                    ground_truth_responses.append({'output_seq': output_component})
               
                tokenized_inputs = self.tokenizer(input_components, add_special_tokens=False, padding="longest", return_tensors="pt").to(self.model.device)
            input_ids = tokenized_inputs["input_ids"]
             
            # Run prediction       
            with torch.no_grad():
                outputs = model.generate(**tokenized_inputs,
                                         max_new_tokens=constants.MAX_NEW_TOKENS,
                                         do_sample=False,
                                         num_beams=1,
                                         )

            # Get only the generated text (skip the input prompt)
            generated_ids_batch = outputs[:, input_ids.shape[1]:]
            responses.extend(self.decode_responses(generated_ids_batch, to_print))

        # Back to train mode
        model.train()

        return responses, ground_truth_responses


    # Evaluates entity type prediction using F1 scores (used in validation and test set)
    def evaluate(self, 
                 eval_dataset=None, 
                 ignore_keys=None,
                 metric_key_prefix="eval", 
                 save_results=False, 
                 result_path=None
                 ):

        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        # Get predictions
        responses, ground_truth_responses = self.get_preds(eval_dataset, metric_key_prefix)

        # Compute scores
        prec, rec, f1, accuracy = compute_prediction_scores(responses, ground_truth_responses)
        
        metrics = dict()
        metrics[f"{metric_key_prefix}_f1"] = f1
        metrics[f"{metric_key_prefix}_prec"] = prec
        metrics[f"{metric_key_prefix}_rec"] = rec
        metrics[f"{metric_key_prefix}_acc"] = accuracy

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        print(f'Evaluation results: {json.dumps(metrics, indent=2)}')

        if metric_key_prefix == 'test' and save_results:
            output = []
            for idx, resp in enumerate(responses):
                output.append(dict())
                output[-1]['uuid'] = eval_dataset[idx]['uuid']
                output[-1]['turn_id'] = eval_dataset[idx]['turn_id']
                tetypes = resp.split('|')
                tetypes = [x.strip() for x in tetypes if '[no entity]' != x]
                output[-1]['prediction'] = tetypes
                
            with open(result_path, 'w') as f:
                json.dump(output, f, indent=2)

        return metrics


class GemmaTrainer():
    # Loads tokenizer and model
    def __init__(self, model_type):
        self.model_type = model_type
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = constants.GEMMA_MODEL_ID[self.model_type]
        self.model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                          quantization_config=bnb_config,
                                                          low_cpu_mem_usage=True,
                                                          attn_implementation="eager",
                                                          )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        print(f"Loaded {model_id}")


    # Main training procedure
    def train(self):
        train_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}train.json', model_type='gemma', mode='train').data.shuffle(seed=constants.SEED)
        validation_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}valid.json', model_type='gemma', mode='eval').data.shuffle(seed=constants.SEED)

        #QLoRA config (with 50% scaling factor)
        #peft_config = LoraConfig(r=64, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        # Try double scaling factor
        peft_config = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

        # Wrap the base model with peft_config
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        training_args = SFTConfig(
            gradient_accumulation_steps=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=6,
            tf32=False,
            fp16=False,
            warmup_ratio=0.03,
            learning_rate=2e-4,
            optim="paged_adamw_8bit",
            dataset_text_field="text",
            max_seq_length=5000,
            output_dir=constants.GEMMA_TRAIN_OUTPUT_DIR[self.model_type],
            overwrite_output_dir=True,
            log_level='warning',
            logging_steps=50,
            save_strategy='steps', 
            save_steps=1000,
            seed=constants.SEED,
            eval_strategy='steps',
            eval_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True, 
            metric_for_best_model='f1',
            greater_is_better=True,         
            report_to='tensorboard',
            run_name='gemma_exp',   
        )

        # Setup training on completion only
        response_template = "<start_of_turn>model"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
        lm_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        trainer = MySFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            peft_config=peft_config,
            args=training_args,
            train_dataset=train_set, 
            eval_dataset=validation_set,
            data_collator=collator,
            #data_collator=lm_collator
        )
        
        
        # Start training
        trainer.train()
        
        # Save the fine-tuned model
        trainer.model.save_pretrained(constants.GEMMA_FT_MODEL[self.model_type])

        # Merge the model with LoRA weights
        base_model = AutoModelForCausalLM.from_pretrained(
            constants.GEMMA_MODEL_ID[self.model_type],
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
        )
        merged_model= PeftModel.from_pretrained(base_model, constants.GEMMA_FT_MODEL[self.model_type])
        merged_model= merged_model.merge_and_unload()

        # Save the merged model for later use
        merged_model.save_pretrained(constants.GEMMA_MERGED_MODEL[self.model_type], safe_serialization=True)
        self.tokenizer.save_pretrained(constants.GEMMA_MERGED_MODEL[self.model_type])
        
        # Get test performance
        test_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}test.json', model_type='gemma', mode='infer').data
        trainer.evaluate(test_set, save_results=True, result_path=constants.GEMMA_TEST_RESULT_FILE[self.model_type], metric_key_prefix='test')