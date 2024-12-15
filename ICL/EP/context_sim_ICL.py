from tqdm import tqdm
import argparse
import json
import os

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from common.retriever import Retriever
from common import constants
from common.mwoz_data import CustomICLMwozDataset
from common.utils import compute_prediction_scores
# from ICL.instruction import EP_FEW_SHOT_CONTEXT_SIMILARITY, EP_ZERO_SHOT_CONTEXT_SIMILARITY, RG_FEW_SHOT_CONTEXT_SIMILARITY, RG_ZERO_SHOT_CONTEXT_SIMILARITY
from ICL.instruction import EP_FEW_SHOT_CONTEXT_SIMILARITY, EP_ZERO_SHOT_CONTEXT_SIMILARITY

def EP_gen_prompt(retriever, text, k=3):
    if k != 0:
        prompt = EP_FEW_SHOT_CONTEXT_SIMILARITY
        prompt += "\n\n"

        top_k_examples = retriever.get_top_k(text, k)
        for idx, examples in enumerate(top_k_examples):
            prompt += f"[Example {idx+1}]\n\n"
            prompt += f"[Dialog] {examples['context']}\n"
            prompt += f"[Entity Types] {examples['output_seq']}\n"
            prompt += "\n"
    else:
        prompt = EP_ZERO_SHOT_CONTEXT_SIMILARITY
        prompt += "\n\n"
    
    prompt += "Now, predict the required entity type list for the next ASSISTANT response refer the example format: \n\n"
    prompt += "### Input \n"
    prompt += f"[Input Dialog] {text}\n\n"
    prompt += "[Entity Types] "

    return prompt

# def RG_gen_prompt(retriever, text, database, k=3):
#     if k != 0:
#         prompt = RG_FEW_SHOT_CONTEXT_SIMILARITY.format(DIALOG=text)
#         prompt += "\n\n"

#         top_k_examples = retriever.get_top_k(text, k)
#         for idx, examples in enumerate(top_k_examples):
#             prompt += f"[Example {idx+1}]\n"
#             prompt += f"[Dialog] {examples['context']}\n\n"
#             prompt += f"[Database] {examples['database']}\n\n"
#             prompt += f"[Output] {examples['output']}\n\n"
#             prompt += "\n"
#     else:
#         prompt = RG_ZERO_SHOT_CONTEXT_SIMILARITY.format(DIALOG=text)
#         prompt += "\n\n"
    
#     prompt += "Now, analyze the provided dialog and database, and generate the ASSISTANT's next response: \n\n"
#     prompt += "### Input \n"
#     prompt += f"[Input Dialog] {text}\n\n"
#     prompt += f"[Database] {database}\n\n"
#     prompt += "[Output] "

#     return prompt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_examples', type=int, default=3)
    parser.add_argument('--mode', type=str,choices=['EP', 'RG'], default='EP')
    parser.add_argument('--gpus', type=str, default='7')
    return parser.parse_args()

def main(args):
    model_id = args.model_id
    results = []
    responses = []

    model = LLM(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    train_set = CustomICLMwozDataset(tokenizer, data_filename=f'{constants.DATA_DIR}train_demon.json', mode='eval')
    test_set = CustomICLMwozDataset(tokenizer, data_filename=f'{constants.DATA_DIR}test_demon.json', mode='eval')

    retriever = Retriever(samples=train_set.data)

    test_context = [x['context'] for x in test_set]
    test_database = [x['database'] for x in test_set]

    if args.mode == "EP":
        prompt_list = [EP_gen_prompt(retriever, x, args.num_examples) for x in test_context]
    # elif args.mode == "RG":
    #     prompt_list = [RG_gen_prompt(retriever, x, test_database[idx], args.num_examples) for idx, x in enumerate(test_context)]

    print(f"Generated {len(prompt_list)} prompts example : {prompt_list[5]}")
    sampling_params = SamplingParams(
        max_tokens=128,
        temperature=0.,
        top_p=1.0,
        stop=['\n\n', 'Dialog', 'Example']
    )

    del retriever

    for i in tqdm(range(0, len(prompt_list), args.batch_size), desc=f"Generating responses"):
        batch_prompt = prompt_list[i:i+args.batch_size]
        test_set_slice = test_set[i:i+args.batch_size]
        outputs = model.generate(batch_prompt,sampling_params=sampling_params, use_tqdm=False)
        for idx, response in enumerate(outputs):
            if args.mode == "EP":
                results.append({'input_seq':test_set_slice[idx]['context'], 'response':response.outputs[0].text, 'label':test_set_slice[idx]['output_seq']})
            elif args.mode == "RG":
                results.append({'input_seq':test_set_slice[idx]['context'], 'response':response.outputs[0].text, 'label':test_set_slice[idx]['output']})
            responses.append(response.outputs[0].text)
    
    basename = os.path.basename(args.model_id)
    file_path = f"ICL/results/context_similarity/{args.num_examples}_context_sim_{basename}.json"
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {file_path}")
    
    if args.mode == "EP":
        prec, rec, f1, acc = compute_prediction_scores(responses, test_set)
        print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")
    elif args.mode == "RG":
        print("Response generation task, no evaluation metrics available")


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    main(args)