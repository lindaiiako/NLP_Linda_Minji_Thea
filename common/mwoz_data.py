import json
from torch.utils.data import Dataset
from datasets import Dataset as HF_Dataset
from common import utils
from common import prompts
from synctod.text_process import preprocess_text


class CustomMwozDataset(Dataset):
    def __init__(self, tokenizer, data_filename, model_type, mode, response_pred=False, is_self_consistency=False):       
        self.tokenizer = tokenizer
        self.mode = mode

        with open(data_filename, 'r') as f:
            self.raw_dataset = json.load(f)

        print(f"Processing: {data_filename} ...")
        if response_pred:
            if model_type == 't5':
                self.data = self.process_data_for_response_prediction_t5(self.raw_dataset)    
            else:
                self.data = HF_Dataset.from_list(self.process_data_for_response_prediction_llm(self.raw_dataset, model_type, 
                                                                                       with_kb=True, 
                                                                                       is_self_consistency=is_self_consistency))
        else:
            if model_type == 't5':
                self.data = self.process_data_for_t5(self.raw_dataset)    
            else:
                self.data = HF_Dataset.from_list(self.process_data_for_llm(self.raw_dataset, model_type, 
                                                                        is_self_consistency=is_self_consistency))
        print("Done.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


    def process_data_for_t5(self, raw_dataset):
        processed_dataset = []
        for row in raw_dataset:
            # Extract context component
            context = row['context_used']

            # Extract kb component
            kb = []
            if row['kb'] is not None:
                kb = row['kb']

            if len(kb) > 0:
                prompt = "Based on the [dialog] and [kb], generate entity types to be included in the response:"
                
                data_attributes = list(kb[0].keys())
                formatted_kb = utils.flatten_kb(kb, data_attributes)

                # Build input
                input = prompt + ' [dialog] ' + context + ' [kb] ' + formatted_kb
            else:
                prompt = "Based on the [dialog], generate entity types to be included in the response:"
                
                # Build input
                input = prompt + ' [dialog] ' + context

            # Build output
            et = row['hints']['entity_types']
            if len(et) > 0:
                output = ' | '.join(sorted(et)) 
            else:
                output = '[no entity]'

            # Build dataset data entry dict
            tokenized_input = self.tokenizer(input, return_tensors="np")
            data_sample = {
                'input_seq': input,
                'input_ids': tokenized_input.input_ids[0],
                'attention_mask': tokenized_input.attention_mask[0],
                'output_seq': output
            }

            # Include ground truth labels in train mode 
            if self.mode == 'train':
                data_sample['labels'] = self.tokenizer(output, return_tensors="np").input_ids[0]

            processed_dataset.append(data_sample)

        return processed_dataset
    
    def process_data_for_response_prediction_t5(self, raw_dataset):
        processed_dataset = []
        for row in raw_dataset:
            # Extract context component
            context = row['context_used']

            # Extract kb component
            kb = []
            if row['kb'] is not None:
                kb = row['kb']

            if len(kb) > 0:
                prompt = "Based on the [dialog] and [kb], generate the next <sys> response:"
                
                data_attributes = list(kb[0].keys())
                formatted_kb = utils.flatten_kb(kb, data_attributes)

                # Build input
                input = prompt + ' [dialog] ' + context + ' [kb] ' + formatted_kb
            else:
                prompt = "Based on the [dialog], generate the next <sys> response:"
                
                # Build input
                input = prompt + ' [dialog] ' + context

            # Build output
            cleaned_output = preprocess_text(row['output'])
            output = "<sys> " + cleaned_output
            
            # Build dataset data entry dict
            tokenized_input = self.tokenizer(input, return_tensors="np")
            data_sample = {
                'input_seq': input,
                'input_ids': tokenized_input.input_ids[0],
                'attention_mask': tokenized_input.attention_mask[0],
                'output_seq': output
            }

            # Include ground truth labels in train mode 
            if self.mode == 'train':
                data_sample['labels'] = self.tokenizer(output, return_tensors="np").input_ids[0]

            processed_dataset.append(data_sample)

        return processed_dataset
    
    
    def process_data_for_llm(self, raw_dataset, model_type, old_version=False, t5_style_prompt=True, with_kb=True, is_self_consistency=False):
        processed_dataset = []
        for row in raw_dataset:
            if t5_style_prompt:
                # Extract context component
                context = row['context_used']

                # Extract kb component
                kb = []
                if row['kb'] is not None:
                    kb = row['kb']

                if with_kb and len(kb) > 0:
                    prompt = "Based on the [dialog] and [kb], generate entity types to be included in the response:"
                    
                    data_attributes = list(kb[0].keys())
                    formatted_kb = utils.flatten_kb(kb, data_attributes)

                    # Build input
                    input = ' [dialog] ' + context + ' [kb] ' + formatted_kb
                else:
                    prompt = "Based on the [dialog], generate entity types to be included in the response:"
                    
                    # Build input
                    input = ' [dialog] ' + context

                # Build output
                et = row['hints']['entity_types']
                if len(et) > 0:
                    output = ' | '.join(sorted(et)) 
                else:
                    output = '[no entity]'

                et_list = output
            else:
                # Define sys prompt
                if old_version:
                    prompt = prompts.IT_SYS_PROMPT_OLD
                else:
                    prompt = prompts.IT_SYS_PROMPT
                
                if row['turn_id'] == 0:
                    # Reset
                    input = "\nDialog:\n"
                    
                human = "\nHUMAN: " + row['context'][-1]
                input += human

                et = row['hints']['entity_types']
                if len(et) > 0:
                    et_list = ' | '.join(sorted(list(set(et)))) 
                else:
                    et_list = '[no entity]'

                if old_version:
                    agent = "\nASSISTANT: " + row['output']
                    output = et_list
                else:
                    agent = "\nASSISTANT: " + et_list
                    output = agent
            
            # Format
            if model_type == 'gemma':
                formatted_seq = utils.format_for_gemma(prompt, input, output, self.mode, is_self_consistency)
            elif model_type == 'llama':
                formatted_seq = utils.format_for_llama(prompt, input, output, self.mode, is_self_consistency)
            elif model_type == 'mistral':
                formatted_seq = utils.format_for_mistral(prompt, input, output, self.mode, is_self_consistency)
            else:
                raise NotImplementedError
            data_sample = {'text': formatted_seq, 'output_seq': et_list, 'uuid': row['uuid'], 'turn_id': row['turn_id']}             
            processed_dataset.append(data_sample)

            if not t5_style_prompt:
                # Append to hist
                input += agent

        return processed_dataset


    def process_data_for_response_prediction_llm(self, raw_dataset, model_type, with_kb=True, is_self_consistency=False):
        processed_dataset = []
        for row in raw_dataset:
            # Extract context component
            context = row['context_used']

            # Extract kb component
            kb = []
            if row['kb'] is not None:
                kb = row['kb']

            if with_kb and len(kb) > 0:
                prompt = "Based on the [dialog] and [kb], generate the next <sys> response:"
                
                data_attributes = list(kb[0].keys())
                formatted_kb = utils.flatten_kb(kb, data_attributes)

                # Build input
                input = ' [dialog] ' + context + ' [kb] ' + formatted_kb
            else:
                prompt = "Based on the [dialog], generate the next <sys> response:"
                
                # Build input
                input = ' [dialog] ' + context

            cleaned_output = preprocess_text(row['output'])
            output = "<sys> " + cleaned_output

            # Format
            if model_type == 'gemma':
                formatted_seq = utils.format_for_gemma(prompt, input, output, self.mode, is_self_consistency)
            elif model_type == 'llama':
                formatted_seq = utils.format_llama_using_chat_template(prompt, input, output, self.mode, is_self_consistency)
            elif model_type == 'mistral':
                formatted_seq = utils.format_mistral_using_chat_template(prompt, input, output, self.mode, is_self_consistency)
            else:
                raise NotImplementedError
            data_sample = {'text': formatted_seq, 'output_seq': cleaned_output, 'uuid': row['uuid'], 'turn_id': row['turn_id']}             
            processed_dataset.append(data_sample)

        return processed_dataset