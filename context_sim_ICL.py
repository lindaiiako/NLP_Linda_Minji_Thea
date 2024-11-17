from tqdm import tqdm
import argparse
import json

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from common.prompt_gen import Retriever
from common import constants
from common.mwoz_data import CustomMwozDataset
from common.utils import compute_prediction_scores

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_examples', type=int, default=3)
    return parser.parse_args()

def main(args):
    model_id = args.model_id
    results = []
    responses = []

    model = LLM(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    train_set = CustomMwozDataset(tokenizer, data_filename=f'{constants.DATA_DIR}train.json', mode='eval')
    test_set = CustomMwozDataset(tokenizer, data_filename=f'{constants.DATA_DIR}test.json', mode='eval')

    retriever = Retriever(samples=train_set.data)

    test_context = [x['input_seq'] for x in test_set]

    prompt_list = [retriever.gen_prompt(x, args.num_examples) for x in test_context]

    sampling_params = SamplingParams(
        max_tokens=128,
        temperature=0.,
        top_p=1.0,
        stop=['\n\n']
    )

    del retriever

    for i in tqdm(range(0, len(prompt_list), args.batch_size), desc=f"Generating responses"):
        batch_prompt = prompt_list[i:i+args.batch_size]
        test_set_slice = test_set[i:i+args.batch_size]
        outputs = model.generate(batch_prompt,sampling_params=sampling_params)
        for idx, response in enumerate(outputs):
            results.append({'input_seq':test_set_slice[idx]['input_seq'], 'response':response.outputs[0].text, 'label':test_set_slice[idx]['output_seq']})
            responses.append(response.outputs[0].text)
    
    with open('results/context_sim_ICL.json', 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to results/context_sim_ICL.json")

    prec, rec, f1, acc = compute_prediction_scores(responses, test_set)
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")

if __name__ == '__main__':
    args = parse_args()
    main(args)