import numpy as np
import json
import torch
from torch.utils.data import DataLoader
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from common import constants
from common.mwoz_data import CustomMwozDataset
from common.utils import compute_prediction_scores
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, PeftModel
from synctod.text_process import preprocess_text
from synctod.metrics import compute_metrics


# Set for reproducibility
np.random.seed(constants.SEED)
torch.manual_seed(constants.SEED)
set_seed(constants.SEED)


# SFTTrainer with custom eval metric
class MySFTTrainer(SFTTrainer):
    # Decodes output with the tokenizer 
    def decode_responses(self, outputs, to_print=False):
        # Placeholder for decoded outputs
        preds = []

        responses = self.tokenizer.batch_decode(outputs.to('cpu'), skip_special_tokens=True)

        # Parse output
        for decoded_response in responses:
            agent_resp = ""
            if "<sys>" in decoded_response:
                agent_resp = decoded_response.split("<sys>")[1].strip()
            else:
                agent_resp = decoded_response

            preds.append(preprocess_text(agent_resp))

            if to_print:
                print("RAW RESPONSE:")
                print(decoded_response)
                print("CLEANED RESPONSE:")
                print(agent_resp)

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
                ground_truth_responses.extend(inputs['output_seq'])
                
                print("TEST INPUT")
                print(formatted_prompts)
                print("CORRECT OUTPUT")
                print(inputs['output_seq'])

                # Print decoded outputs for debugging
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
                    output_component = split_text[1].split('<sys>')[1].replace('<end_of_turn>','').replace('<eos>', '').strip()
                    ground_truth_responses.append(output_component)
               
                tokenized_inputs = self.tokenizer(input_components, add_special_tokens=False, padding="longest", return_tensors="pt").to(self.model.device)
            input_ids = tokenized_inputs["input_ids"]
             
            # Run prediction       
            with torch.no_grad():
                outputs = model.generate(**tokenized_inputs,
                                         max_new_tokens=384,
                                         do_sample=False,
                                         num_beams=1,
                                         )

            # Get only the generated text (skip the input prompt)
            generated_ids_batch = outputs[:, input_ids.shape[1]:]
            responses.extend(self.decode_responses(generated_ids_batch, to_print))

        # Back to train mode
        model.train()

        return responses, ground_truth_responses


    # Evaluates response prediction with BLEU and Entity F1
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
        results = compute_metrics(responses, ground_truth_responses, 'MultiWOZ', f'{constants.DATA_DIR}entities.json')
        
        metrics = dict()
        metrics[f"{metric_key_prefix}_bleu"] = results["bleu"]
        metrics[f"{metric_key_prefix}_f1"] = results["entity_f1"]
        metrics[f"{metric_key_prefix}_precision"] = results["entity_precision"]
        metrics[f"{metric_key_prefix}_recall"] = results["entity_recall"]

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
                output[-1]['ground_resp'] = eval_dataset[idx]['output_seq']
                output[-1]['prediction'] = resp
                
            with open(result_path, 'w') as f:
                json.dump(output, f, indent=2)

        return metrics


class GemmaTrainer():
    # Loads tokenizer and model
    def __init__(self, model_type):
        self.model_type = model_type
        self.key_name = "gemma-response-pred"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model_id = constants.MODEL_ID[self.model_type]
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
        train_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}train.json', model_type='gemma', mode='train', response_pred=True).data.shuffle(seed=constants.SEED).take(100)
        # Full training data size: 8529
        print(f"Training data size: {len(train_set)}")
        print("SAMPLES")
        print(train_set[0])
        print(train_set[1])
        #print(train_set[55])

        validation_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}valid.json', model_type='gemma', mode='eval', response_pred=True).data.shuffle(seed=constants.SEED)

        # LoRA with double scaling factor
        peft_config = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

        # Wrap the base model with peft_config
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        training_args = SFTConfig(
            gradient_accumulation_steps=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=2,
            tf32=False,
            fp16=False,
            warmup_ratio=0.03,
            learning_rate=2e-4,
            optim="paged_adamw_8bit",
            dataset_text_field="text",
            max_seq_length=5000,
            output_dir=constants.TRAIN_OUTPUT_DIR[self.key_name],
            overwrite_output_dir=True,
            log_level='warning',
            logging_steps=50,
            save_strategy='steps', 
            save_steps=400,
            seed=constants.SEED,
            eval_strategy='steps',
            eval_steps=400,
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

        trainer = MySFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            peft_config=peft_config,
            args=training_args,
            train_dataset=train_set, 
            eval_dataset=validation_set,
            data_collator=collator,
        )
        
        # Start training
        trainer.train()
        
        # Save the fine-tuned model
        trainer.model.save_pretrained(constants.FT_MODEL[self.key_name])

        # Merge the model with LoRA weights
        base_model = AutoModelForCausalLM.from_pretrained(
            constants.MODEL_ID[self.model_type],
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
        )
        merged_model= PeftModel.from_pretrained(base_model, constants.FT_MODEL[self.key_name])
        merged_model= merged_model.merge_and_unload()

        # Save the merged model for later use
        merged_model.save_pretrained(constants.MERGED_MODEL[self.key_name], safe_serialization=True)
        self.tokenizer.save_pretrained(constants.MERGED_MODEL[self.key_name])
        
        
        # Get test performance
        test_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}test.json', model_type='gemma', mode='infer', response_pred=True).data
        trainer.evaluate(test_set, save_results=True, result_path=constants.TEST_RESULT_FILE[self.key_name], metric_key_prefix='test')