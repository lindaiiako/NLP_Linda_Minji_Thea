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
import re

def refine_context(context):
    context = context.replace("<sys>", "\n<sys>").replace("<user>", "\n<user>")
    return context

class CustomMwozDatasetICL(Dataset):
    def __init__(self, tokenizer, data_filename, mode, num_examples=3, is_icl=False):       
        self.tokenizer = tokenizer
        self.mode = mode
        self.num_examples = num_examples

        with open(data_filename, 'r') as f:
            raw_dataset = json.load(f)

        print(f"Processing: {data_filename} ...")
        self.raw_data = raw_dataset
        if is_icl:
            self.data = self.process_data_icl(raw_dataset)
        else: 
            self.data = self.process_data_zero(raw_dataset=raw_dataset)
        print("Done.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def process_data_zero(self, raw_dataset):
        processed_dataset = []
        
        allowed_entity_types = [
            "address", "area", "food", "name", "phone", "postcode", 
            "pricerange", "type", "ref", "choice", "stars", "entrance_fee", "none"
        ]

        for row in raw_dataset:
            context = row['context_used']

            prompt = "You are given a dialog between a user and a task-oriented dialog system. Your task is to predict the final the entity types required for the system's next response."
            prompt += "The possible values for entity types are address, area, food, name, phone, postcode, pricerange, type, ref, choice, stars, entrance_fee, or none if not applicable. Do not attempt to provide any entity type that does not match the entity types: address, area, food, name, phone, postcode, pricerange, type, ref, choice, stars, entrance_fee, or none."
            prompt = prompt + "\n\nFollow these rules:"
            prompt = prompt + "\n1. The possible values for entity types are address, area, food, name, phone, postcode, pricerange, type, ref, choice, stars, entrance_fee, or none if not applicable. Do not attempt to provide any entity type that does not match the entity types: address, area, food, name, phone, postcode, pricerange, type, ref, choice, stars, entrance_fee, or none."
            prompt = prompt + "\n1. If the conversation is over (e.g., user says 'that's all, thanks'), return [no entity]."
            
            prompt += "\n\nOutput Format: Put the predicted entity type in a list which is separated by | symbol. Give use only the final prediction. "
            
            input = prompt + '\n\nNow, given below dialog, your task is to predict the final the entity types that should be included in the next response: ' 
            input += refine_context(context) + "\n\n**Answers:**" #+ "**\nOutput format: please follow this format for the predicting entity types for next response.\n**Answers:**" #+ "\n</dialog>"
            
            # Build output
            et = row['hints']['entity_types']
            if len(et) > 0:
                # Filter the entity types to include only the allowed ones
                filtered_et = [entity for entity in et if entity in allowed_entity_types]
                output = '|'.join(sorted(filtered_et)) 
            else:
                output = '[no entity]' #"none"#'[no entity]'

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

    def process_data_icl(self, raw_dataset):
        processed_dataset = []

        allowed_entity_types = [
            "address", "area", "food", "name", "phone", "postcode", 
            "pricerange", "type", "ref", "choice", "stars", "entrance_fee", "none"
        ]

        for i, row in enumerate(raw_dataset):
            # Construct ICL examples from previous rows
            icl_examples = self.build_icl_examples(i, raw_dataset, allowed_entity_types)

            # Current dialog context
            context = row['context_used']

            icl_examples = icl_examples #refine_context(icl_examples)
            
            prompt = (
                "You are an expert assisting in task-oriented dialog. For each incomplete dialog, "
                "predict the most probable entity types required in the next response based on the dialog history. "
                "Only these entity types are valid: address, area, food, name, phone, postcode, pricerange, type, ref, choice, stars, entrance_fee."
                "\nDo not attempt to provide any entity type that does not match with these entity types, address, area, food, name, phone, postcode, pricerange, type, ref, choice, stars, entrance_fee."
                "\nProvide predicting entity type in the next response as a list seperated by | symbol."
                "\n\n**Important Rule:**" 
                "\n1. If the user has clearly indicated that the conversation is over (e.g., 'that's all, thanks', 'Thanks, bye', 'i'm all set. thanks again.', ' that is all. thanks for you help.'), you should not predict any entity type, even if the dialogue context mentions entities. Answer must be [no entity]\n"
                "\n2. Do not attempt to provide any entity type that does not match with these entity types, address, area, food, name, phone, postcode, pricerange, type, ref, choice, stars, entrance_fee."
                "\n3. Before making prediction, you need to read all the conversation between <sys> and <user>, some of the entity type you are providing might have answered already before user last message."

                "\nNow for the dialog below, provide the entity types that should be include in the next response:\n"
                f"\n<dialog> {context}\n"
                "\n\n**The final predicted entity types are: **"  
            )

            # Output
            et = row['hints']['entity_types']
            filtered_et = [entity for entity in et if entity in allowed_entity_types]
            output = '|'.join(sorted(filtered_et)) if filtered_et else '[no entity]'

            tokenized_input = self.tokenizer(prompt, return_tensors="np")
            data_sample = {
                'input_seq': prompt,
                'input_ids': tokenized_input.input_ids[0],
                'attention_mask': tokenized_input.attention_mask[0],
                'output_seq': output
            }

            if self.mode == 'train':
                data_sample['labels'] = self.tokenizer(output, return_tensors="np").input_ids[0]

            processed_dataset.append(data_sample)

        return processed_dataset

    def build_icl_examples(self, current_idx, dataset, allowed_entity_types):
        """Builds the ICL examples dynamically using prior samples."""
        examples = []
        for i in range(max(0, current_idx - self.num_examples), current_idx):
            row = dataset[i]
            context = row['context_used']
            et = row['hints']['entity_types']
            filtered_et = [entity for entity in et if entity in allowed_entity_types]
            output = '|'.join(sorted(filtered_et)) if filtered_et else '[no entity]'
            example = f"<dialog> {context} \nThe predicted entity types: {output}"
            examples.append(example)
        return "\n\n".join(examples)

def decode_with_confidence_score_zero(generation_outputs, tokenizer):
    generated_sequence = generation_outputs.sequences[0]
    
    # Get predicted text
    decoded_seq = tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    if "The final answer is".lower() in decoded_seq.lower():
        predicted_text = decoded_seq.split("The final answer is:", 1)[-1].strip().split("</s>")[0].strip()

        start_index = predicted_text.find("{") + len("{")
        end_index = predicted_text.find("}", start_index)

        predicted_text_to_return = predicted_text[start_index:end_index]
        predicted_text_to_return = predicted_text_to_return.replace(",", "|")
        predicted_text_to_return = predicted_text_to_return.replace("[", "")
        predicted_text_to_return = predicted_text_to_return.replace("]", "")
        
        if "reference number" in predicted_text_to_return:
            predicted_text_to_return = predicted_text_to_return.replace("reference number", "ref")
        predicted_text_to_return = predicted_text_to_return.replace(" ", "")
        
    else:
        if "The final predicted entity types are:".lower() in decoded_seq.lower():
            print("\n***\nhello from the final predicted entity type\n***\n")

            match = re.findall(r"The final predicted entity types are: *\[[^\]]*\]|\*\*The final predicted entity types are:\*\*[\s]*([^\n]*)", decoded_seq) #re.findall(r"The final predicted entity types are:\s*(?:\[)?([^\]\n]+)(?:\])?", decoded_seq) #re.findall(r"The final predicted entity types are:\s*\[([^\]]+)\]", decoded_seq)
            if match:
                entity_type = match[-1]
                print("***match#1\nmatch text ", entity_type, "\n***")
                predicted_text_to_return = entity_type #" | ".join(entity_type)
                predicted_text_to_return = predicted_text_to_return.replace(",", "|")
                predicted_text_to_return = predicted_text_to_return.replace("[", "")
                predicted_text_to_return = predicted_text_to_return.replace("]", "")
                
                predicted_text = decoded_seq.split("<pad>", 1)[-1].strip().split("</s>")[0].strip()
            else:
                print("else if ")
                predicted_text = decoded_seq.split("<pad>", 1)[-1].strip().split("</s>")[0].strip()
                predicted_text_to_return = "[no entity]"
                
            
            match = re.findall(r"\*\*The final predicted entity types are: \*\*\s*\[([^\]]+)\]", decoded_seq) #re.findall(r"The final predicted entity types are:\s*(?:\[)?([^\]\n]+)(?:\])?", decoded_seq) #re.findall(r"The final predicted entity types are:\s*\[([^\]]+)\]", decoded_seq)
            if match:
                print("match#2")
                entity_type = match[-1]
                print("***\nmatch text ", entity_type, "\n***")
                predicted_text_to_return = entity_type #" | ".join(entity_type)
                predicted_text_to_return = predicted_text_to_return.replace(",", "|")
                predicted_text_to_return = predicted_text_to_return.replace("[", "")
                predicted_text_to_return = predicted_text_to_return.replace("]", "")
                
                predicted_text = decoded_seq.split("<pad>", 1)[-1].strip().split("</s>")[0].strip()
            else:
                print("else if ")
                predicted_text = decoded_seq.split("<pad>", 1)[-1].strip().split("</s>")[0].strip()
                predicted_text_to_return = "[no entity]"  
        else:  
            predicted_text = decoded_seq.split("<pad>", 1)[-1].strip().split("</s>")[0].strip()
            predicted_text_to_return = "[no entity]" #"none"#"[no entity]"
    
    predicted_text_to_return = predicted_text_to_return.lower()
    predicted_text_to_return = predicted_text_to_return.replace("-", "")
    predicted_text_to_return = predicted_text_to_return.replace(":", "|")
    predicted_text_to_return = predicted_text_to_return.replace("**", "").replace("'", "")
    predicted_text_to_return = predicted_text_to_return.replace("list of entity types seperated by |", "[no entity]")
    
    predicted_text_to_return = predicted_text_to_return.split("\n", 1)[0]
    if predicted_text_to_return == "no entity" or predicted_text_to_return == "noentity" or predicted_text_to_return == "" or predicted_text_to_return == "[]" or predicted_text_to_return == "none" or predicted_text_to_return == "  " or predicted_text_to_return == " " or predicted_text_to_return == " none":
        predicted_text_to_return = "[no entity]"
    
    if "Thefinalpredictedentitytypesare:" in predicted_text_to_return:
        predicted_text_to_return = predicted_text_to_return.replace("Thefinalpredictedentitytypesare:", "")
    if "**Thefinalpredictedentitytypesare:**" in predicted_text_to_return:
        predicted_text_to_return = predicted_text_to_return.replace("**Thefinalpredictedentitytypesare:**", "")
        
    if "list the entity types" in predicted_text_to_return or "none" in predicted_text_to_return:
        print("Predicted text to return contains none")
        predicted_text_to_return = "[no entity]"
        
    # assume there is no entity 
    if "no entity" in predicted_text_to_return or "noentity" in predicted_text_to_return:
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
    print(f"Model Prediction:\n{decoded_seq}")
    print("\n################################################################################")
    
    return predicted_text_to_return, avg_confidence_score


def decode_with_confidence_score_icl(generation_outputs, tokenizer):
    generated_sequence = generation_outputs.sequences[0]

    decoded_seq = tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    if "The final answer is:".lower() in decoded_seq.lower():
        predicted_text = decoded_seq.split("The final answer is:", 1)[-1].strip().split("</s>")[0].strip()
        
        start_index = predicted_text.find("{") + len("{")
        end_index = predicted_text.find("}", start_index)

        predicted_text_to_return = predicted_text[start_index:end_index]
        predicted_text_to_return = predicted_text_to_return.replace(",", "|")
        predicted_text_to_return = predicted_text_to_return.replace("[", "")
        predicted_text_to_return = predicted_text_to_return.replace("]", "")
        
        if "reference number" in predicted_text_to_return:
            predicted_text_to_return = predicted_text_to_return.replace("reference number", "ref")
            
        predicted_text_to_return = predicted_text_to_return.replace(" ", "")
        
    else:
        
        if "The final predicted entity types are:".lower() in decoded_seq.lower():
            print("\n***\nhello from the final predicted entity type\n***\n")

            match = re.findall(r"The final predicted entity types are: *\[[^\]]*\]|\*\*The final predicted entity types are:\*\*[\s]*([^\n]*)", decoded_seq) #re.findall(r"The final predicted entity types are:\s*(?:\[)?([^\]\n]+)(?:\])?", decoded_seq) #re.findall(r"The final predicted entity types are:\s*\[([^\]]+)\]", decoded_seq)
            if match:
                entity_type = match[-1]
                print("***match#1\nmatch text ", entity_type, "\n***")
                predicted_text_to_return = entity_type #" | ".join(entity_type)
                predicted_text_to_return = predicted_text_to_return.replace(",", "|")
                predicted_text_to_return = predicted_text_to_return.replace("[", "")
                predicted_text_to_return = predicted_text_to_return.replace("]", "")
                
                predicted_text = decoded_seq.split("<pad>", 1)[-1].strip().split("</s>")[0].strip()
            else:
                predicted_text = decoded_seq.split("<pad>", 1)[-1].strip().split("</s>")[0].strip()
                predicted_text_to_return = "[no entity]"
                
            
            match = re.findall(r"\*\*The final predicted entity types are: \*\*\s*\[([^\]]+)\]", decoded_seq) #re.findall(r"The final predicted entity types are:\s*(?:\[)?([^\]\n]+)(?:\])?", decoded_seq) #re.findall(r"The final predicted entity types are:\s*\[([^\]]+)\]", decoded_seq)
            if match:
                print("match#2")
                entity_type = match[-1]
                print("***\nmatch text ", entity_type, "\n***")
                predicted_text_to_return = entity_type #" | ".join(entity_type)
                predicted_text_to_return = predicted_text_to_return.replace(",", "|")
                predicted_text_to_return = predicted_text_to_return.replace("[", "")
                predicted_text_to_return = predicted_text_to_return.replace("]", "")
                
                predicted_text = decoded_seq.split("<pad>", 1)[-1].strip().split("</s>")[0].strip()
            else:
                print("else if ")
                predicted_text = decoded_seq.split("<pad>", 1)[-1].strip().split("</s>")[0].strip()
                predicted_text_to_return = "[no entity]"
                
        else:  
            predicted_text = decoded_seq.split("<pad>", 1)[-1].strip().split("</s>")[0].strip()
            predicted_text_to_return = "[no entity]" #"none"#"[no entity]"
    
    predicted_text_to_return = predicted_text_to_return.lower()
    predicted_text_to_return = predicted_text_to_return.replace("-", "")
    predicted_text_to_return = predicted_text_to_return.replace(":", "|")
    predicted_text_to_return = predicted_text_to_return.replace("**", "").replace("'", "")
    predicted_text_to_return = predicted_text_to_return.replace("list of entity types seperated by |", "[no entity]")
    if predicted_text_to_return == "no entity" or predicted_text_to_return == "noentity" or predicted_text_to_return == "" or predicted_text_to_return == "[]" or predicted_text_to_return == "none" or predicted_text_to_return == "  " or predicted_text_to_return == " " or predicted_text_to_return == " none":
        predicted_text_to_return = "[no entity]"
    
    if "Thefinalpredictedentitytypesare:" in predicted_text_to_return:
        predicted_text_to_return = predicted_text_to_return.replace("Thefinalpredictedentitytypesare:", "")
    if "**Thefinalpredictedentitytypesare:**" in predicted_text_to_return:
        predicted_text_to_return = predicted_text_to_return.replace("**Thefinalpredictedentitytypesare:**", "")
        
    if "list the entity types" in predicted_text_to_return or "none" in predicted_text_to_return:
        print("Predicted text to return contains none")
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
    print(f"Model Prediction:\n{decoded_seq}")
    print("\n")
    # print(f"Clean text: {predicted_text_to_return}")
    # print(f"Model Prediction: {predicted_text} \nConfidence: {avg_confidence_score}")
    print("\n################################################################################")
    
    return predicted_text_to_return, avg_confidence_score

################################################################################################################################

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

is_icl = False
num_examples=3

#
if is_icl:
    print("You are running ICL with 3shot")
else: 
    print("You are running zeroshot.")
    
if is_icl:
    validation_set = CustomMwozDatasetICL(
        tokenizer, 
        data_filename='data/valid.json', 
        mode="eval", 
        num_examples=3,
        is_icl=is_icl
    )
else: 
    validation_set = CustomMwozDatasetICL(
        tokenizer, 
        data_filename='data/valid.json', 
        mode="eval", 
        num_examples=0,
        is_icl=is_icl
    ) 

eval_loader = DataLoader(validation_set[:], batch_size=1)

# Set model to evaluation mode
model.eval()

all_preds = []
all_labels = []
results = []
results_dir = "results"

if is_icl == False:
    generated_results_file = os.path.join(results_dir, "llama_zeroshot.json")
    evaluation_summary_file = "llama_zeroshot_f1.json"
else:
    generated_results_file = os.path.join(results_dir, f"llama_icl_{num_examples}shot.json")
    evaluation_summary_file = f"llama_icl_{num_examples}shot_f1.json"

for batch in tqdm(eval_loader, desc="Evaluating"):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    output_seq = batch['output_seq']  # Ground truth labels

    with torch.no_grad():
        # Generate predictions
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100, #len(input_ids),
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            repetition_penalty=1.1,# Apply a repetition penalty (values > 1 discourage repeats)
        )

        if is_icl:
            predicted_text, confidence_score = decode_with_confidence_score_icl(outputs, tokenizer)
        else:
            predicted_text, confidence_score = decode_with_confidence_score_zero(outputs, tokenizer)
        cleaned_text = predicted_text 
        
        print("\n****")
        print("GT: ", output_seq)
        print(f"Cleaned Predicted Text: {cleaned_text}")
        print(f"Confidence Score: {confidence_score}")
        print("****\n")
        
        data_to_save = {
            "gt": output_seq,
            "out": cleaned_text
        }
        results.append(data_to_save)

    all_preds.append(cleaned_text)
    all_labels.append(output_seq)
    
    with open(generated_results_file, "w") as f:
        json.dump(results, f, indent=4)
        print("generated_results_file ==> ", generated_results_file)


# Compute F1 score and accuracy
precision, recall, f1, accuracy = utils.compute_prediction_scores(all_preds, validation_set, "|")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")


# Save evaluation metrics
evaluation_summary = {
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "accuracy": accuracy
}


with open(evaluation_summary_file, "w") as f:
    json.dump(evaluation_summary, f, indent=4)
    
print("evaluation_summary_file ==> ", evaluation_summary_file)
