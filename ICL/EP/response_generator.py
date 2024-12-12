from tqdm import tqdm
import argparse
import json
import os

from vllm import LLM, SamplingParams

from common import constants
from common.mwoz_data import CustomSuperICLMwozDataset
from common.utils import compute_prediction_scores

GENERAL_INSTRUCTION = '''You are given a dialog between a USER and an ASSISTANT.   
Refer the dialog, espeically last utterance, generate a list of entity types that should be included in the ASSISTANT's next response.
Your response must consists of the following entity types: address, area, entrance_fee, food, name, phone, postcode, pricerange, type, stars, ref, choice. 

You are provided prediction from another model and corresponding confidence score. Use this information as a reference.

Here is how to approach the task:
1. Analyze the given dialog thoroughly.
2. Review the predictions and confidence scores to guide your decision.
3. Predict the final entity types based on the dialog context and reference.

Here is an example:
<|start_header_id|>user<|end_header_id|> i kind of need some help finding a nice hotel in the north part of town .
<|eot_id|><|start_header_id|>assistant<|end_header_id|> area | choice | pricerange | type

Answer using `|` delimiter without repetitions like given example. If there is no entity needed, just write "[no entity]".
'''.strip()

GENERAL_WO_CONFIDENCE_INSTRUCTION = '''You are given a dialog between a USER and an ASSISTANT.   
Refer the dialog, espeically last utterance, generate a list of entity types that should be included in the ASSISTANT's next response.
Your response must consists of the following entity types: address, area, entrance_fee, food, name, phone, postcode, pricerange, type, stars, ref, choice. 
You are provided prediction from another model. Use this information as a reference.

Here is how to approach the task:
1. Analyze the given dialog thoroughly.
2. Review the predictions to guide your decision.
3. Predict the final entity types based on the dialog context and reference.

Here is an example:
<|start_header_id|>user<|end_header_id|> i kind of need some help finding a nice hotel in the north part of town .
<|start_header_id|>assistant<|end_header_id|> area | choice | pricerange | type

Answer using `|` delimiter without repetitions like given example. If there is no entity needed, just write "[no entity]".
'''.strip()

GENERAL_WO_T5_INSTRUCTION = '''You are given a dialog between a USER and an ASSISTANT.   
Refer the dialog, espeically last utterance, generate a list of entity types that should be included in the ASSISTANT's next response.
Your response must consists of the following entity types: address, area, entrance_fee, food, name, phone, postcode, pricerange, type, stars, ref, choice. 

Here is how to approach the task:
1. Analyze the given dialog thoroughly.
2. Predict the final entity types based on the dialog context.

Here is an example:
<Dialog> USER : i kind of need some help finding a nice hotel in the north part of town .
<Answer> area | choice | pricerange | type

Answer using `|` delimiter without repetitions like given example. If there is no entity needed, just write "[no entity]".
'''.strip()

ANTROPHIC_PROMPT = '''You are an AI assistant tasked with predicting entity types from incomplete task-oriented dialogs. You will be given a dialog, a list of entity types, and additional information to help guide your decision.

Following is the incomplete task-oriented dialog:
{DIALOG}

This is a list of entity types that should be included in the ASSISTANT's next response: address, area, entrance_fee, food, name, phone, postcode, pricerange, type, stars, ref, choice
This is an additional information, including another model's prediction and confidence score: {ADDITIONAL_INFO}
You have to predict which entity types from the given list should be included in the ASSISTANT's next response. Use the additional information to guide your decision.

Consider the dialog context, the given entity types, and the additional information, provide your final answer within [Answer] tags. Remember to use the `|` delimiter without repetitions. If there is no entity needed, just write "[no entity]". '''.strip()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--model_id', type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--mode', type=str,choices=['whole','wo_t5','wo_confidence'], default='whole')
    return parser.parse_args()

def apply_chat_template(tokenizer, dataset):
    new_dataset = []
    for data in dataset:
        instruction = GENERAL_INSTRUCTION + '\n\n' + '[Prediction] ' + data['prediction'] + '\n\n' + '[Confidence] ' + data['confidence'] + '\n\n'
        messages = [{"role":"system", "content":instruction}]
        for idx, turn in enumerate(data['context']):
            if idx % 2 == 0:
                messages.append({"role":"user", "content":turn})
            else:
                messages.append({"role":"assistant", "content":turn})

        messages = tokenizer.apply_chat_template(messages, tokenize=False)
        new_dataset.append(messages)
    return new_dataset

def prompt_gen(dataset):
    new_dataset = []
    for data in dataset:
        instruction = ANTROPHIC_PROMPT.format(DIALOG=data['context'], ADDITIONAL_INFO=f"[Prediction] {data['prediction']} [Confidence] {data['confidence']}")
        new_dataset.append(instruction)
    return new_dataset


def main(args):
    model_id = args.model_id
    results = []
    responses = []

    model = LLM(model_id)
    tokenizer=model.get_tokenizer()
    
    test_set = CustomSuperICLMwozDataset(tokenizer, data_filename=f'{constants.DATA_DIR}test_demon.json', mode='eval')
    # prompt_list = apply_chat_template(tokenizer, test_set, args.mode)
    prompt_list = prompt_gen(test_set)

    print(f"Generated {len(prompt_list)} prompts example : {prompt_list[5]}")
    stop_words = ["/answer", "/Answer", "/ANSWER"]
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.,
        top_p=1.0,
        stop=stop_words
    )
    
    for i in tqdm(range(0, len(prompt_list), args.batch_size), desc=f"Generating responses"):
        batch_prompt = prompt_list[i:i+args.batch_size]
        test_set_slice = test_set[i:i+args.batch_size]
        outputs = model.generate(batch_prompt,sampling_params=sampling_params, use_tqdm=False)
        for idx, response in enumerate(outputs):
            results.append({'uuid':test_set_slice[idx]['uuid'], 'input_seq':test_set_slice[idx]['context'], 'response':response.outputs[0].text, 'label':test_set_slice[idx]['output_seq'], 't5_predictions':test_set_slice[idx]['prediction'], 't5_confidence':test_set_slice[idx]['confidence']})
            responses.append(response.outputs[0].text)

    base_name = os.path.basename(args.model_id)
    output_file = "results/" + base_name +'_'+args.mode + "_superICL.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved at {output_file}")

    prec, rec, f1, acc = compute_prediction_scores(responses, test_set)
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    main(args)