import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from common import constants
from collections import Counter
from common.mwoz_data import CustomMwozDataset
from common.utils import compute_prediction_scores


class Tester():
    # Loads tokenizer and model
    def __init__(self, model_name):
        self.model_name = model_name
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        model_id = constants.MODEL_ID[self.model_name]
        merged_model_path = constants.MERGED_MODEL[self.model_name]
        self.model = AutoModelForCausalLM.from_pretrained(merged_model_path,
                                                          quantization_config=bnb_config,
                                                          low_cpu_mem_usage=True,
                                                          attn_implementation="eager",
                                                          device_map='cuda:7')
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'


    def test(self, is_self_consistency):
        test_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}test.json', model_type=self.model_name, mode='infer', is_self_consistency=is_self_consistency).data

        results = []
        for idx, example in enumerate(test_set):
            counter = Counter()
            formatted_prompts = example['text']
            tokenized_inputs = self.tokenizer(formatted_prompts, return_tensors="pt").to(self.model.device)
            print("TEST INPUT")
            print(formatted_prompts)
            print("CORRECT OUTPUT")
            print(example['output_seq'])

            input_ids = tokenized_inputs["input_ids"]
            if is_self_consistency:
                outputs = self.model.generate(**tokenized_inputs,
                                    max_new_tokens=128,
                                    do_sample=True,
                                    num_return_sequences=3,
                                    top_p=0.2,
                                    temperature=0.1,
                                    )
                generated_ids_batch = outputs[:, input_ids.shape[1]:]
                responses = self.tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=True)

                for response in responses:
                    if 'no entity' in response:
                        response = '[no entity]'
                    print(response)

                    # Ensure unique
                    et_preds = ' | '.join(set(response.split('|'))).split('|')
                    #et_preds = response.split('|')
                    
                    for et_pred in et_preds:
                        counter.update([et_pred.strip()])
                    
                print(counter)
                # Get common preds
                top_items = [key for key, value in counter.items() if value >= 2]
                if len(top_items) == 0:
                    # Heuristic just aggregate
                    top_items = [key for key, value in counter.items() if value == 1]

                print("TOP")
                if '[no entity]' in top_items:
                    answer = '[no entity]'
                else:
                    answer = ' | '.join(sorted(top_items)) 
            else:
                # Beam search
                outputs = self.model.generate(**tokenized_inputs,
                                    max_new_tokens=128,
                                    do_sample=True,
                                    num_beams=3,
                                    )
                response = outputs[0][input_ids.shape[-1]:]
                decoded_response = self.tokenizer.decode(response, skip_special_tokens=True)
                if '[no entity]' in decoded_response:
                    answer = '[no entity]'
                else:
                    answer = decoded_response.replace('[', '').replace(']', '').strip()
                
            print("CLEAN OUT")
            print(answer)
            results.append(answer)

        # Compute scores
        prec, rec, f1, accuracy = compute_prediction_scores(results, test_set)
        
        metrics = dict()
        metrics["test_f1"] = f1
        metrics["test_prec"] = prec
        metrics["test_rec"] = rec
        metrics["test_acc"] = accuracy

        print(f'Evaluation results: {json.dumps(metrics, indent=2)}')

        output = []
        for idx, resp in enumerate(results):
            output.append(dict())
            output[-1]['uuid'] = test_set[idx]['uuid']
            output[-1]['turn_id'] = test_set[idx]['turn_id']
            tetypes = resp.split('|')
            tetypes = [x.strip() for x in tetypes if '[no entity]' != x]
            output[-1]['prediction'] = tetypes
            
        with open(constants.GEMMA_TEST_RESULT_FILE[self.model_type], 'w') as f:
            json.dump(output, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="gemma")
    parser.add_argument('--self_consistency', type=bool, default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = Tester(args.model_name)
    model.test(args.self_consistency)