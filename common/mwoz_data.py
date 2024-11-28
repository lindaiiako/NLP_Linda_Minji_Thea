import json
from torch.utils.data import Dataset
from datasets import Dataset as HF_Dataset
from common import utils
from common import prompts


class CustomMwozDataset(Dataset):
    def __init__(self, tokenizer, data_filename, model_type, mode):       
        self.tokenizer = tokenizer
        self.mode = mode

        with open(data_filename, 'r') as f:
            self.raw_dataset = json.load(f)

        print(f"Processing: {data_filename} ...")
        if model_type == 't5':
            self.data = self.process_data_for_t5(self.raw_dataset)
        elif model_type == 'llama_it':
            self.data = HF_Dataset.from_list(self.process_data_for_llama_it(self.raw_dataset))
        elif model_type == 'gemma':
            self.data = HF_Dataset.from_list(self.process_data_for_gemma(self.raw_dataset))
        else:
            raise NotImplementedError
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
    

    def process_data_for_llama_it(self, raw_dataset):
        processed_dataset = []
        for row in raw_dataset:
            # Define sys prompt
            prompt = prompts.IT_SYS_PROMPT
            # Define input
            input = row['context_used']

            # Build output
            et = row['hints']['entity_types']
            if len(et) > 0:
                #output = ' | '.join(sorted(et))
                # Repetitions disturb training so this is changed!
                output = ' | '.join(list(set(et))) 
            else:
                output = '[no entity]'

            # Format for llama
            formatted_seq = utils.format_chat_template(prompt, input, output, self.mode, self.tokenizer)
            data_sample = {'text': formatted_seq, 'output_seq': output, 'uuid': row['uuid'], 'turn_id': row['turn_id'], 'prompt': prompt, 'input': input}             
            
            processed_dataset.append(data_sample)

        return processed_dataset
    
    def process_data_for_gemma(self, raw_dataset):
        processed_dataset = []
        for row in raw_dataset:
            # Define sys prompt
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

            agent = "\nASSISTANT: " + et_list
            
            # Format for gemma
            formatted_seq = utils.format_for_gemma(prompt, input, agent, self.mode)
            data_sample = {'text': formatted_seq, 'output_seq': et_list, 'uuid': row['uuid'], 'turn_id': row['turn_id']}             
            
            processed_dataset.append(data_sample)

            # Append to hist
            input += agent

        return processed_dataset