# import os
# import torch
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from common import utils
# from common.mwoz_data import CustomMwozDataset
# import torch.nn.functional as F
# import json
# from torch.utils.data import Dataset

# class CustomMwozDataset(Dataset):
#     def __init__(self, tokenizer, data_filename, mode):       
#         self.tokenizer = tokenizer
#         self.mode = mode

#         with open(data_filename, 'r') as f:
#             raw_dataset = json.load(f)

#         print(f"Processing: {data_filename} ...")
#         self.data = self.process_data(raw_dataset)
#         print("Done.")

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

#     def process_data(self, raw_dataset):
#         processed_dataset = []
        
#         # Define the allowed entity types
#         allowed_entity_types = [
#             "address", "area", "food", "name", "phone", "postcode", 
#             "pricerange", "type", "ref", "choice", "stars", "entrance_fee", "none"
#         ]

#         for row in raw_dataset:
#             # Extract context component
#             context = row['context_used']

#             # Extract kb component
#             kb = []
#             if row['kb'] is not None:
#                 kb = row['kb']

#             if False: #len(kb) > 0:
#                 prompt = "Based on the [dialog] and [kb], generate entity types to be included in the response:"
                
#                 data_attributes = list(kb[0].keys())
#                 formatted_kb = utils.flatten_kb(kb, data_attributes)

#                 # Build input
#                 input = prompt + ' [dialog] ' + context ######+ ' [kb] ' + formatted_kb #+ ". Based on given context the generated entity type can be empty. If any, the generated entity types as list is " #+ " The entity types as a list is : " #+ "\n\n\nThe generated entity types are " #+ "### Expected Output entity_types="
#             else:
#                 prompt = "You are given an incomplete dialog between a user and a task-oriented dialog system. As an expert, you should provide the most probable entity types required in the next response. The generated entity types need to be in the format of a list which each entity type is separated by | symbol. The possible values for entity types are address, area, food, name, phone, postcode, pricerange, type, ref, choice, stars, entrance_fee, or none if not matching previous one."#not applicable. "
#                 #"Based on the [dialog], generate entity types to be included in the response:"

#                 input = prompt + ' <dialog> ' + context #+ "<sys> " #+ ". Based on given context the generated entity type can be empty. If any, the generated entity types as list is " #+ " If possible, the generated entity types as list is " # The entity types as a list is : " #+ "\n\n\nThe generated entity types are " #+ "### Expected Output entity_types="

#             # Build output
#             et = row['hints']['entity_types']
#             if len(et) > 0:
#                 # Filter the entity types to include only the allowed ones
#                 filtered_et = [entity for entity in et if entity in allowed_entity_types]
#                 output = ' | '.join(sorted(filtered_et)) 
#             else:
#                 output = '[no entity]'

#             # Build dataset data entry dict
#             tokenized_input = self.tokenizer(input, return_tensors="np")
#             data_sample = {
#                 'input_seq': input,
#                 'input_ids': tokenized_input.input_ids[0],
#                 'attention_mask': tokenized_input.attention_mask[0],
#                 'output_seq': output
#             }

#             # Include ground truth labels in train mode 
#             if self.mode == 'train':
#                 data_sample['labels'] = self.tokenizer(output, return_tensors="np").input_ids[0]

#             processed_dataset.append(data_sample)

#         return processed_dataset

# import ast
# import re
# def extract_entities_from_generated_text(predicted_text):

#     # Find the starting point of "Entity types:"
#     # start_index = predicted_text.find("Entity types:")
#     start_index = predicted_text.find("The final answer is:")#"### Expected Output entity_types=")

#     if start_index != -1:
#         # Get the text after "Entity types:"
#         entity_text = predicted_text[start_index + len("The final answer is:"):].strip()
        
#         entity_text = entity_text[:entity_text.find(']')]
#         entity_text = entity_text.strip('```').strip().replace('\n', '').replace(' ', '')
#         entity_text = entity_text.replace(",", " | ")
        
#         return entity_text
#     else:
#         print("[no entity]")
#         return '[no entity]'



# # Computes average confidence score for text-based format outputs using logits scores
# # Note: assumes input is 1 sample (not batch)
# def decode_with_confidence_score(generation_outputs, tokenizer):
#     generated_sequence = generation_outputs.sequences[0]
    
#     # Get predicted text
#     decoded_seq = tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#     # predicted_text = decoded_seq.split("<pad>", 1)[-1].strip().split("</s>")[0].strip()
#     if "The most probable entity types required in the next response are:".lower() in decoded_seq.lower():
#         predicted_text = decoded_seq.split("The most probable entity types required in the next response are:", 1)[-1].strip().split("</s>")[0].strip()
        
#         # Find the starting index of the content inside \boxed{}
#         start_index = predicted_text.find("**") + len("**")
#         # Find the ending index of the content
#         end_index = predicted_text.find("**", start_index)

#         # Extract the content
#         predicted_text_to_return = predicted_text[start_index:end_index]
#         predicted_text_to_return = predicted_text_to_return.replace("|", " | ")
#         # print(result)
#     else:
#         predicted_text = decoded_seq.split("<pad>", 1)[-1].strip().split("</s>")[0].strip()
#         predicted_text_to_return = "[no entity]"
     
    
#     # Get logits for each token
#     logits = generation_outputs.scores                    
    
#     # Compute confidence scores for each token
#     probs = []
#     for idx in range(len(logits)):
#         token_logits = logits[idx]                        # logits of this token
#         token_id = generated_sequence[idx+1].item()       # corresponding token ID in T5

#         # TODO: add condition to check only for expected labels?
#         if token_id != tokenizer.pad_token_id and token_id != tokenizer.eos_token_id:
#             probabilities = F.softmax(token_logits, dim=-1) 
#             prob = probabilities[0, token_id]    
#             probs.append(prob)

#     # Calculate average confidence score for the entire sequence
#     if len(probs) > 0:
#         avg_confidence_score = sum(probs) / len(probs)
#     else:
#         avg_confidence_score = "unknown"                # maybe better than 0?

#     # Output results
#     print("\n################################################################################")
#     print(f"Model Prediction: {decoded_seq}")
#     # print("\n")
#     print(f"Clean text: {predicted_text_to_return}")
#     print(f"Model Prediction: {predicted_text} \nConfidence: {avg_confidence_score}")
#     print("\n################################################################################")
    
#     return predicted_text_to_return, avg_confidence_score

# def extract_entity_types_from_text(predicted_text, entity_types):
#     """
#     Function to extract entity types from the predicted text.
#     If no entity type is found, return '[no entity]'.
#     """
#     # Normalize the text for easier matching
#     predicted_text_lower = predicted_text.lower()

#     # Extract entity types mentioned in the text
#     extracted_entities = [entity for entity in entity_types if entity.lower() in predicted_text_lower]
    

#     # If no entities found, return '[no entity]'
#     if not extracted_entities:
#         return '[no entity]'

#     return " | ".join(extracted_entities)

# # Initialize tokenizer and model
# # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
# # tokenizer.pad_token = tokenizer.eos_token
# # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
# tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")

# # Move model to device (GPU if available)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Load validation set
# validation_set = CustomMwozDataset(tokenizer, data_filename='data/valid.json', mode="eval")
# eval_loader = DataLoader(validation_set, batch_size=1)

# # Set model to evaluation mode
# model.eval()

# all_preds = []
# all_labels = []

# # Iterate over validation set and generate predictions
# for batch in tqdm(eval_loader, desc="Evaluating"):
#     input_ids = batch['input_ids'].to(device)
#     attention_mask = batch['attention_mask'].to(device)
#     output_seq = batch['output_seq']  # Ground truth labels

#     print("len of input ids ", len(input_ids))
#     with torch.no_grad():
#         # Generate predictions
#         outputs = model.generate(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             max_new_tokens=1000, #len(input_ids),
#             pad_token_id=tokenizer.eos_token_id,
#             return_dict_in_generate=True,
#             output_scores=True
#         )

#         # Decode the generated sequence
#         predicted_text, confidence_score = decode_with_confidence_score(outputs, tokenizer)

#         cleaned_text = predicted_text #extract_entities_from_generated_text(predicted_text=predicted_text)

#         print("################################################################################")
#         # print(f"Original predicted text: {predicted_text}")
#         print("GT: ", output_seq)
#         print(f"Cleaned Predicted Text: {cleaned_text}")
#         print(f"Confidence Score: {confidence_score}")
#         print("################################################################################")

#     # Append predictions and labels
#     all_preds.append(cleaned_text)
#     all_labels.append(output_seq)

# # # Print first 5 decoded predictions
# # print("Decoded Predictions:")
# # for pred in all_preds[:5]:
# #     print(pred)

# # # Print first 5 ground truth labels
# # print("Ground Truth Labels:")
# # for label in all_labels[:5]:
# #     print(label)

# # Compute F1 score and accuracy
# precision, recall, f1, accuracy = utils.compute_prediction_scores(all_preds, validation_set, "|")

# # Print evaluation results
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1 Score: {f1:.4f}")
# print(f"Accuracy: {accuracy:.4f}")


import os
import json
from tqdm import tqdm
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import utils
from common.mwoz_data import CustomMwozDataset
import torch.nn.functional as F
import json
from torch.utils.data import Dataset


class CustomMwozDataset(Dataset):
    def __init__(self, tokenizer, data_filename, mode):       
        self.tokenizer = tokenizer
        self.mode = mode

        with open(data_filename, 'r') as f:
            raw_dataset = json.load(f)

        print(f"Processing: {data_filename} ...")
        self.data = self.process_data(raw_dataset)
        print("Done.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def process_data(self, raw_dataset):
        processed_dataset = []
        
        # Define the allowed entity types
        allowed_entity_types = [
            "address", "area", "food", "name", "phone", "postcode", 
            "pricerange", "type", "ref", "choice", "stars", "entrance_fee", "none"
        ]

        for row in raw_dataset:
            # Extract context component
            context = row['context_used']

            # Extract kb component
            kb = []
            if row['kb'] is not None:
                kb = row['kb']

            if False: #len(kb) > 0:
                prompt = "Based on the [dialog] and [kb], generate entity types to be included in the response:"
                
                data_attributes = list(kb[0].keys())
                formatted_kb = utils.flatten_kb(kb, data_attributes)

                # Build input
                input = prompt + ' [dialog] ' + context ######+ ' [kb] ' + formatted_kb #+ ". Based on given context the generated entity type can be empty. If any, the generated entity types as list is " #+ " The entity types as a list is : " #+ "\n\n\nThe generated entity types are " #+ "### Expected Output entity_types="
            else:
                prompt = "You are given an incomplete dialog between a user and a task-oriented dialog system. As an expert, you should provide the most probable entity types required in the next response. The predicting entity type are seperated by , symbol. The possible values for entity types are address, area, food, name, phone, postcode, pricerange, type, ref, choice, stars, entrance_fee, or none if not matching previous one."#not applicable. "
                #"Based on the [dialog], generate entity types to be included in the response:"

                input = prompt + ' <dialog> ' + context #+ "<sys> " #+ ". Based on given context the generated entity type can be empty. If any, the generated entity types as list is " #+ " If possible, the generated entity types as list is " # The entity types as a list is : " #+ "\n\n\nThe generated entity types are " #+ "### Expected Output entity_types="

            # Build output
            et = row['hints']['entity_types']
            if len(et) > 0:
                # Filter the entity types to include only the allowed ones
                filtered_et = [entity for entity in et if entity in allowed_entity_types]
                output = ' | '.join(sorted(filtered_et)) 
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

# Computes average confidence score for text-based format outputs using logits scores
# Note: assumes input is 1 sample (not batch)
def decode_with_confidence_score(generation_outputs, tokenizer):
    generated_sequence = generation_outputs.sequences[0]
    
    # Get predicted text
    decoded_seq = tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # predicted_text = decoded_seq.split("<pad>", 1)[-1].strip().split("</s>")[0].strip()
    if "entity types required in the next response are:".lower() in decoded_seq.lower():
        # predicted_text = decoded_seq.split("entity types required in the next response are:", 1)[-1].strip().split("</s>")[0].strip()
        predicted_text = decoded_seq

        # Find the starting index of the content inside \boxed{}
        start_index = predicted_text.find("**") + len("**")
        # Find the ending index of the content
        end_index = predicted_text.find("**", start_index)

        # Extract the content
        predicted_text_to_return = predicted_text[start_index:end_index]
        predicted_text_to_return = predicted_text_to_return.replace(", ", " | ")
        if "**" in predicted_text_to_return:
            predicted_text_to_return = predicted_text_to_return.replace("**", "")
        # print(result)
    else:
        predicted_text = decoded_seq.split("<pad>", 1)[-1].strip().split("</s>")[0].strip()
        predicted_text_to_return = "[no entity]"
     
    
    # Get logits for each token
    logits = generation_outputs.scores                    
    
    # Compute confidence scores for each token
    probs = []
    for idx in range(len(logits)):
        token_logits = logits[idx]                        # logits of this token
        token_id = generated_sequence[idx+1].item()       # corresponding token ID in T5

        # TODO: add condition to check only for expected labels?
        if token_id != tokenizer.pad_token_id and token_id != tokenizer.eos_token_id:
            probabilities = F.softmax(token_logits, dim=-1) 
            prob = probabilities[0, token_id]    
            probs.append(prob)

    # Calculate average confidence score for the entire sequence
    if len(probs) > 0:
        avg_confidence_score = sum(probs) / len(probs)
    else:
        avg_confidence_score = "unknown"                # maybe better than 0?

    # Output results
    print("\n################################################################################")
    print(f"Model Prediction: {decoded_seq}")
    # print("\n")
    print(f"Clean text: {predicted_text_to_return}")
    print(f"Model Prediction: {predicted_text} \nConfidence: {avg_confidence_score}")
    print("\n################################################################################")
    
    return predicted_text_to_return, avg_confidence_score

# Initialize paths for saving results
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")

# Move model to device (GPU if available)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load validation set
validation_set = CustomMwozDataset(tokenizer, data_filename='data/valid.json', mode="eval")
eval_loader = DataLoader(validation_set, batch_size=1)

# File paths
generated_results_file = os.path.join(results_dir, "generated_results.json")
evaluation_summary_file = os.path.join(results_dir, "evaluation_summary.json")

# Initialize containers for results
results = []


# # Set model to evaluation mode
model.eval()

all_preds = []
all_labels = []

# Iterate over validation set and generate predictions
for batch in tqdm(eval_loader, desc="Evaluating"):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    output_seq = batch['output_seq']  # Ground truth labels

    with torch.no_grad():
        # Generate predictions
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1000,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

        # Decode the generated sequence
        predicted_text, confidence_score = decode_with_confidence_score(outputs, tokenizer)

        cleaned_text = predicted_text

    # Save each result
    result_entry = {
        # "input_seq": batch['input_seq'][0],
        "output_seq": output_seq[0],
        # "generated_text": predicted_text,
        "cleaned_text": cleaned_text,
        # "confidence_score": confidence_score,
    }
    results.append(result_entry)

    # Write intermediate results to file
    with open(generated_results_file, "w") as f:
        json.dump(results, f, indent=4)

# Compute final scores
precision, recall, f1, accuracy = utils.compute_prediction_scores(all_preds, validation_set, "|")

# Save evaluation metrics
evaluation_summary = {
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "accuracy": accuracy
}

with open(evaluation_summary_file, "w") as f:
    json.dump(evaluation_summary, f, indent=4)

# Print evaluation results
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
