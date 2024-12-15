import os
import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import utils
from common.mwoz_data import CustomMwozDataset
import torch.nn.functional as F
from torch.utils.data import Dataset
import re

def construct_icl_prompt(context, entity_types, examples, max_demo=3):
        """
        Construct an in-context learning prompt with `max_demo` demonstrations.
        """
        prompt = "You are given an incomplete dialog between a user and a task-oriented dialog system. As an expert, you should provide the most probable entity types required in the next response. The generated entity types need to be in the format of a list which each entity type is separated by | symbol. The possible values for entity types are address, area, food, name, phone, postcode, pricerange, type, ref, choice, stars, entrance_fee, or none if not applicable. Please do not tend to provide any entity type that does not match the entity types in previous sentence.\n\n"
        prompt = prompt + "**Important Rule**: If the user has clearly indicated that the conversation is over (e.g., 'that's all, thanks', 'Thanks, bye', 'i'm all set. thanks again.', ' that is all. thanks for you help.'), you should not predict any entity type, even if the dialogue context mentions entities. Answer must be [no entity]"
        
        max_demo = 3
        for example in examples[:max_demo]:
            prompt += f"Dialog: {example['context_used']}\n"
            if example['hints']['entity_types']:
                et = ' | '.join(sorted(example['hints']['entity_types']))
            else:
                et = '[no entity]'
            prompt += f"Entity types: {et}\n"
        
        # Add the test case
        prompt += f"Now, based on the following dialog, predict the entity types required for the next response:\n"
        prompt += f"Dialog: {context}\nThe final entity types:" #f"The final entity types:"#Dialog: {context}\nThe final entity types:"
        return prompt

def preprocess_tod_context(context):
    """
    Preprocess Task-Oriented Dialogue context to separate <user> and <sys> turns.
    
    Args:
        context (str): The raw TOD context string with <user> and <sys> tags.
        
    Returns:
        str: Preprocessed and structured dialogue text.
    """
    lines = context.split("\n")
    structured_dialogue = []
    for line in lines:
        if line.startswith("<user>"):
            user_text = line.replace("<user>", "<user>").strip()
            structured_dialogue.append(user_text + "\n\n")
        elif line.startswith("<sys>"):
            sys_text = line.replace("<sys>", "<user>").strip()
            structured_dialogue.append(sys_text+ "\n\n")
    return "\n".join(structured_dialogue)
    
def construct_zeroshot_prompt(context):
    prompt = "You are given a dialog between a user <user> and a task-oriented dialog system <sys>. Your task is to predict entity types required for the system's next response. The possible values for entity types are address, area, food, name, phone, postcode, pricerange, type, ref, choice, stars, entrance_fee, or none if not applicable. Do not attempt to provide any entity type that does not match the entity types: address, area, food, name, phone, postcode, pricerange, type, ref, choice, stars, entrance_fee, or none."#Please ensure that you only provide entity types for the next response that align with the options listed in previous sentence. "#Please do not tend to provide any entity type that does not match the entity types in previous sentence."#"You are given an incomplete dialog between a user and a task-oriented dialog system. As an expert, you should provide the most probable entity types required in the next response." #If you are unsure about any information, kindly ask the user to clarify rather than guessing." # And, do not try to give any entity type that does not match in the previous sentence."#not applicable. "
    prompt = prompt + "\n**Important Rule**: If the user has clearly indicated that the conversation is over (e.g., 'that's all, thanks', 'Thanks, bye', 'i'm all set. thanks again.', ' that is all. thanks for you help.'), you should not predict any entity type, even if the dialogue context mentions entities. Answer must be [no entity]."
    prompt = prompt + "\nOutput Format: "
    
    prompt = prompt + "\n**Answers:**  For the next response, please provide only the most preferable prediction that match with these entity types (address, area, food, name, phone, postcode, pricerange, type, ref, choice, stars, entrance_fee, or none) in the next response as comma-seperated list of entity types within square brackets. Please be careful when user already end the dialogue in that case do not try to generate any entity type. "
    prompt = prompt + "\n**Explaination:** why the next response should include them?"
    
    context_refined = preprocess_tod_context(context)
    
    # print("Refine context ", context_refined)
    input = prompt + '\n<dialog>\n' + context_refined + "</dialog>\n" #+ "\nBased on given context, the most appropriate entity type for the next response:"#+ "<sys> " #+ ". Based on given context the generated entity type can be empty. If any, the generated entity types as list is " #+ " If possible, the generated entity types as list is " # The entity types as a list is : " #+ "\n\n\nThe generated entity types are " #+ "### Expected Output entity_types="
    return input

class CustomMwozDataset(Dataset):
    def __init__(self, tokenizer, data_filename, mode, is_icl=False):       
        self.tokenizer = tokenizer
        self.mode = mode

        with open(data_filename, 'r') as f:
            raw_dataset = json.load(f)

        print(f"Processing: {data_filename} ...")
        self.data = self.process_data(raw_dataset, is_icl=is_icl)
        print("Done.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def process_data(self, raw_dataset, is_icl=False):
        processed_dataset = []
        allowed_entity_types = [
            "address", "area", "food", "name", "phone", "postcode",
            "pricerange", "type", "ref", "choice", "stars", "entrance_fee", "none"
        ]

        for idx, row in enumerate(raw_dataset):
            context = row['context_used']
            examples = raw_dataset[:idx]  # Previous rows as examples for ICL
            
            # check zeroshot or icl
            if is_icl:                
                # Construct the ICL prompt
                prompt = construct_icl_prompt(context, allowed_entity_types, examples, max_demo=3)
            else: 
                prompt = construct_zeroshot_prompt(context=context)
            
            # Ground truth entity types
            et = row['hints']['entity_types']
            if len(et) > 0:
                filtered_et = [entity for entity in et if entity in allowed_entity_types]
                output = ' | '.join(sorted(filtered_et)) 
            else:
                output = '[no entity]'

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
    
import re 
# Function to extract entity types
def extract_entities(text):
    # Regular expression to find text enclosed by ** **
    return re.findall(r"\*\*(.*?)\*\*", text)

def format_entities_as_string(entities):
    # Join the list of entities with '|'
    return '|'.join(entities)

# decode and extract zeroshot gemma
def decode_with_confidence_score_zero(generation_outputs, tokenizer):
    generated_sequence = generation_outputs.sequences[0]
    
    decoded_seq = tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    if "The most probable entity types required in the next response are:".lower() in decoded_seq.lower():
        predicted_text = decoded_seq
        predicted_text_to_return = extract_entities(predicted_text, "The most probable entity types required in the next response are:")
        predicted_text_to_return = format_entities_as_string(predicted_text_to_return)
        predicted_text_to_return = predicted_text_to_return.replace(":", "")
        if "**" in predicted_text_to_return:
            predicted_text_to_return = predicted_text_to_return.replace("**", "")
            
        predicted_text_to_return = predicted_text_to_return.replace("[", "").replace("]", "")
    
    elif "**Answers:**" in decoded_seq:
        matches = re.findall(r"\*\*Answers:\*\*\s*(\[[^\]]+\]|\w+)", decoded_seq)
        print("matches ## ", matches)

        entities = []
        for match in matches[1:]:
            if match.startswith("["):  
                entities.extend(match.strip("[]").split(", "))
            else:  
                entities.append(match)
                
        predicted_text_to_return = " | ".join(entities)
    
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
    print("\n\n")
    print(f"Model Prediction:\n{decoded_seq}")    
    print("\n\n")

    if predicted_text_to_return == "none":
        predicted_text_to_return = "[no entity]"
    
    return predicted_text_to_return, avg_confidence_score

# Computes average confidence score for text-based format outputs using logits scores
def decode_with_confidence_score(generation_outputs, tokenizer):
    generated_sequence = generation_outputs.sequences[0]
    
    # Get predicted text
    decoded_seq = tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    predicted_text_to_return = extract_entity_types(decoded_seq)
    
    # final cleanup
    if "reference number" in predicted_text_to_return:
        predicted_text_to_return = predicted_text_to_return.replace("reference number", "ref")
        
    if predicted_text_to_return == "" or predicted_text_to_return == " " or predicted_text_to_return == "[]" or predicted_text_to_return == "[ ]":
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
    print("\n\n➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️")
    print(f"Model Prediction: \n{decoded_seq}")
    print("➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️\n")

    
    return predicted_text_to_return, avg_confidence_score


def extract_entity_types(llm_output):
    """
    Extract entity types from the given LLM output using multiple regex patterns.
    """
    # First pattern to match the structured format with backticks and square brackets
    match = re.search(r'The final entity types:\s*```\s*Entity types:\s*\[(.*?)\]', llm_output, re.DOTALL)
    if match:
        entity_types = match.group(1).strip()
        if not entity_types or entity_types == '```':
            return '[no entity]'
        return entity_types.replace("**", "").replace(",", " | ")  # Formatting optional

    # Fallback pattern to handle simpler cases
    match = re.search(r'The final entity types:\s*(.+)', llm_output)
    if match:
        entity_types = match.group(1).strip()
        if not entity_types or entity_types in ['```', '']:
            return '[no entity]'
        return entity_types.replace("**", "").replace(",", " | ")  # Formatting optional

    # If no patterns match, return no entity
    return '[no entity]'

# Initialize paths for saving results
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")

# Move model to device (GPU if available)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#
is_icl = True
print("\n\n#########################")
print("Model: Gemma")
if is_icl:
    print(f"You are running fewshot.")
else:
    print("You are running zeroshot.")
print("#########################\n\n")

# Load validation set
validation_set = CustomMwozDataset(tokenizer, data_filename='data/valid.json', mode="eval", is_icl=is_icl)
eval_loader = DataLoader(validation_set, batch_size=1)

# File paths
generated_results_file = os.path.join(results_dir, f"gemma_icl_{is_icl}_json.json")
evaluation_summary_file = os.path.join(results_dir, f"gemma_icl_{is_icl}_f1.json")

# Initialize containers for results
results = []

# Set model to evaluation mode
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
            max_new_tokens=2048,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

        # Decode the generated sequence
        if is_icl:
            predicted_text, confidence_score = decode_with_confidence_score(outputs, tokenizer)
        else:
            predicted_text, confidence_score = decode_with_confidence_score_zero(outputs, tokenizer)
        print("\n****")
        print("GT : ", output_seq)
        print("PRED: ", predicted_text)
        print("****\n")
        cleaned_text = predicted_text
        all_preds.append(cleaned_text)

    # Save each result
    result_entry = {
        "output_seq": output_seq[0],
        "cleaned_text": cleaned_text,
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

