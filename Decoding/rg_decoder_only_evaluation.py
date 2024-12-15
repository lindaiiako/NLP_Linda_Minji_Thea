import os
import numpy as np
import json
import torch
from typing import List, Dict
import argparse
from tqdm import tqdm

from vllm import LLM, SamplingParams
from common import constants
from common.mwoz_data import CustomMwozDataset
from synctod.text_process import preprocess_text
from synctod.metrics import compute_metrics


class ModelEvaluator:
    def __init__(self, model_path: str, model_type: str):
        """Initialize evaluator with model path and type (gemma/mistral/llama)"""
        self.model_type = model_type
        self.model_path = model_path
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.65,
            trust_remote_code=True
        )
        self.tokenizer = self.model.get_tokenizer()

    def _process_response(self, response: str) -> str:
        """Process response to extract system response"""
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[1].strip()
            
        # Extract response after <sys> token
        if "<sys>" in response:
            response = response.split("<sys>")[1].strip()
            
        # Clean response using SyncTOD's preprocess_text
        return preprocess_text(response)
    
    def _prepare_prompt(self, text: str) -> str:
        """Prepare prompt based on model type"""
        if self.model_type == 'gemma':
            return text + "<start_of_turn>model"
        elif self.model_type == 'llama':
            return text + "[/INST]"
        else:  # mistral
            return text + "[/INST]"
    
    def evaluate_greedy(self, dataset) -> Dict:
        """Evaluate using greedy decoding"""
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=384,
        )
        predictions, outputs = self._evaluate(dataset, sampling_params)
        return self._compute_metrics(predictions, dataset), predictions

    def evaluate_beam_search(self, dataset, num_beams: int, temperature: float) -> Dict:
        """Evaluate using beam search decoding"""
        prompts = [self._prepare_prompt(item['text']) for item in dataset]

        outputs = self.model.beam_search(
            prompts,
            beam_width=num_beams,
            max_tokens=384,
        )
        
        predictions = []
        for output in outputs:
            pred = self._process_response(output.outputs[0].text)
            predictions.append(pred)

        return self._compute_metrics(predictions, dataset), predictions
    
    def evaluate_self_consistency(self, dataset, num_samples: int, temperature: float) -> Dict:
        """Evaluate using self-consistency decoding"""
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=384,  # Response generation 용으로 변경
            n=num_samples,
        )

        all_prompts = [self._prepare_prompt(item['text']) for item in dataset]
        outputs = self.model.generate(all_prompts, sampling_params)

        final_predictions = []
        for prompt_outputs in outputs:
            # Get all predictions for this prompt
            predictions = [
                self._process_response(output.text)
                for output in prompt_outputs
            ]
            
            # Perform majority voting
            prediction_counts = {}
            for pred in predictions:
                if pred in prediction_counts:
                    prediction_counts[pred] += 1
                else:
                    prediction_counts[pred] = 1
            
            # Get majority prediction
            majority_prediction = max(prediction_counts.items(), key=lambda x: x[1])[0]
            final_predictions.append(majority_prediction)
        
        return self._compute_metrics(final_predictions, dataset), final_predictions
    
    def _evaluate(self, dataset, sampling_params: SamplingParams) -> tuple:
        """Common evaluation logic"""
        prompts = [self._prepare_prompt(item['text']) for item in dataset]
        outputs = self.model.generate(prompts, sampling_params)

        predictions = []
        for output in outputs:
            pred = self._process_response(output.outputs[0].text)
            predictions.append(pred)
        
        return predictions, outputs
    
    def _compute_metrics(self, predictions: List[str], dataset) -> Dict:
        """Compute BLEU and Entity F1 metrics for response generation"""
        ground_truth = [entry['output_seq'] for entry in dataset]
        results = compute_metrics(predictions, ground_truth, 'MultiWOZ', f'{constants.DATA_DIR}entities.json')
        
        metrics = {
            "bleu": results["bleu"],
            "entity_f1": results["entity_f1"],
            "entity_precision": results["entity_precision"],
            "entity_recall": results["entity_recall"]
        }
        return metrics
    
    def save_predictions(self, predictions: List[str], dataset, output_path: str):
        """Save predictions to file"""
        output = []
        for idx, pred in enumerate(predictions):
            output.append({
                'uuid': dataset[idx]['uuid'],
                'turn_id': dataset[idx]['turn_id'],
                'ground_resp': dataset[idx]['output_seq'],
                'prediction': pred
            })
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)


def mk_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model')
    parser.add_argument('--model_type', type=str, required=True, choices=['gemma', 'llama', 'mistral'], help='Type of model')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--num_beams', type=int, default=5, help='Number of beams for beam search')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples for self-consistency')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for self-consistency')
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPUs to use')
    return parser.parse_args()


def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        model_type=args.model_type
    )

    # Load test dataset with response_pred=True
    test_dataset = CustomMwozDataset(
        evaluator.tokenizer, 
        data_filename=f'{constants.DATA_DIR}test.json',
        model_type=args.model_type,
        mode='infer',
        response_pred=True  # Response generation task
    ).data
    
    print(f"Evaluating {args.model_type} model from {args.model_path}")

    # Evaluate with different decoding strategies
    print("\nGreedy decoding:")
    greedy_metrics, greedy_preds = evaluator.evaluate_greedy(test_dataset)
    print(json.dumps(greedy_metrics, indent=2))
    evaluator.save_predictions(
        greedy_preds,
        test_dataset,
        os.path.join(args.output_dir, f'{args.model_type}_greedy_predictions.json')
    )

    print(f"\nBeam search decoding (num_beams={args.num_beams}):")
    beam_metrics, beam_preds = evaluator.evaluate_beam_search(test_dataset, args.num_beams, args.temperature)
    print(json.dumps(beam_metrics, indent=2))
    evaluator.save_predictions(
        beam_preds,
        test_dataset,
        os.path.join(args.output_dir, f'{args.model_type}_beam_predictions.json')
    )

    print(f"\nSelf-consistency decoding (samples={args.num_samples}, temp={args.temperature}):")
    sc_metrics, sc_preds = evaluator.evaluate_self_consistency(
        test_dataset,
        args.num_samples,
        args.temperature
    )
    print(json.dumps(sc_metrics, indent=2))
    evaluator.save_predictions(
        sc_preds,
        test_dataset,
        os.path.join(args.output_dir, f'{args.model_type}_sc_predictions.json')
    )

    # Save all metrics
    all_metrics = {
        'greedy': greedy_metrics,
        'beam_search': beam_metrics,
        'self_consistency': sc_metrics
    }
    with open(os.path.join(args.output_dir, f'{args.model_type}_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == '__main__':
    args = mk_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    main(args)