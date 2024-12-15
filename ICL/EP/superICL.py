from tqdm import tqdm
import argparse
import json
import os

from vllm import LLM, SamplingParams

from common import constants
from common.mwoz_data import CustomSuperICLMwozDataset
from common.utils import compute_prediction_scores
from ICL.instruction import SUPER_ICL_PROMPT, SUPER_ICL_PROMPT_WO_CONFIDENCE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--model_id', type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--mode', type=str,choices=['whole','wo_t5','wo_confidence'], default='whole')
    return parser.parse_args()

def prompt_gen(tokenizer, dataset):
    new_dataset = []
    for data in dataset:
        if args.mode == 'whole':
            prompt = SUPER_ICL_PROMPT
            
            prompt += "Now, predict the required entity type list for the next ASSISTANT response refer the example format: \n\n"
            prompt += "### Input:\n"
            prompt += f"[Dialog] {data['context']}\n"
            prompt += f"[Model Prediction] {data['prediction']}\n"
            prompt += f"[Confidence] {data['confidence']}\n"
            prompt += "[Entity Types] "
        elif args.mode == 'wo_confidence':
            prompt = SUPER_ICL_PROMPT_WO_CONFIDENCE
            prompt += "Now, predict the required entity type list for the next ASSISTANT response refer the example format: \n\n"
            prompt += "### Input: \n"
            prompt += f"[Dialog] {data['context']}\n"
            prompt += f"[Model Prediction] {data['prediction']}\n"
            prompt += "[Entity Types] "
        
        new_dataset.append(prompt)

    return new_dataset

def main(args):
    model_id = args.model_id
    results = []
    responses = []

    model = LLM(model_id)
    tokenizer=model.get_tokenizer()
    
    test_set = CustomSuperICLMwozDataset(tokenizer, data_filename=f'{constants.DATA_DIR}test_demon.json', mode='eval')
    prompt_list = prompt_gen(tokenizer, test_set)

    print(f"Generated {len(prompt_list)} prompts example : {prompt_list[5]}")
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.,
        top_p=1.0,
    )
    
    for i in tqdm(range(0, len(prompt_list), args.batch_size), desc=f"Generating responses"):
        batch_prompt = prompt_list[i:i+args.batch_size]
        test_set_slice = test_set[i:i+args.batch_size]
        outputs = model.generate(batch_prompt,sampling_params=sampling_params, use_tqdm=False)
        for idx, response in enumerate(outputs):
            results.append({'uuid':test_set_slice[idx]['uuid'], 'input_seq':test_set_slice[idx]['context'], 'response':response.outputs[0].text.replace("<|start_header_id|>assistant<|end_header_id|>\n\n",""), 'label':test_set_slice[idx]['output_seq'], 't5_predictions':test_set_slice[idx]['prediction'], 't5_confidence':test_set_slice[idx]['confidence']})
            responses.append(response.outputs[0].text.strip())

    base_name = os.path.basename(args.model_id)
    output_file = "ICL/results/SuperICL/" + args.mode + '_' + base_name +'_'+".json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved at {output_file}")

    prec, rec, f1, acc = compute_prediction_scores(responses, test_set)
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    main(args)