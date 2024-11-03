import json
from torch.utils.data import Dataset
from common import utils


class CustomMwozDataset(Dataset):
    def __init__(self, tokenizer, data_filename):       
        self.tokenizer = tokenizer

        with open(data_filename, 'r') as f:
            raw_dataset = json.load(f)

        self.data = self.process_data(raw_dataset)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]


    def process_data(self, raw_dataset):
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
                'output_seq': output,
                'labels': self.tokenizer(output, return_tensors="np").input_ids[0]
            }

            processed_dataset.append(data_sample)

        return processed_dataset