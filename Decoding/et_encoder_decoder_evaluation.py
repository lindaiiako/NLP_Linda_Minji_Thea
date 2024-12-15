import os
import numpy as np
import json
import torch
import argparse
from typing import List, Dict
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from common import constants
from common.mwoz_data import CustomMwozDataset
from common.utils import compute_prediction_scores
from tqdm import tqdm


def mk_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--num_beams', type=int, default=5, help='Number of beams for beam search')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples for self-consistency')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for self-consistency')
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPUs to use')
    return parser.parse_args()


class T5Evaluator:
    def __init__(self, model_path: str, batch_size: int):
        """
        Initialize evaluator with T5 model and tokenizer
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        
    def _process_response(self, response: str) -> str:
        """Process model response to clean the output"""
        cleaned = response.split("<pad>", 1)[-1].strip().split("</s>")[0].strip()
        return cleaned
            
    def evaluate_greedy(self, dataset) -> Dict:
        """Evaluate using greedy decoding"""
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            shuffle=False
        )
        
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Greedy decoding"):
                outputs = self.model.generate(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    max_new_tokens=constants.MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    return_dict_in_generate=True,
                )
                
                responses = self.tokenizer.batch_decode(outputs.sequences, clean_up_tokenization_spaces=False)
                predictions.extend([self._process_response(r) for r in responses])
                
                
        return self._compute_metrics(predictions, dataset), predictions
        
    def evaluate_beam_search(self, dataset, num_beams: int) -> Dict:
        """Evaluate using beam search"""
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            shuffle=False
        )
        
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Beam search"):
                outputs = self.model.generate(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    max_new_tokens=constants.MAX_NEW_TOKENS,
                    num_beams=num_beams,
                    return_dict_in_generate=True,
                    num_return_sequences=1
                )
                
                responses = self.tokenizer.batch_decode(outputs.sequences, clean_up_tokenization_spaces=False)
                predictions.extend([self._process_response(r) for r in responses])
                
                
        return self._compute_metrics(predictions, dataset), predictions
        
    def evaluate_self_consistency(self, dataset, num_samples: int, temperature: float) -> Dict:
        """Evaluate using self-consistency with majority voting"""
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            shuffle=False
        )
        
        final_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Self-consistency"):
                batch_predictions = []
                
                for _ in range(num_samples):
                    outputs = self.model.generate(
                        input_ids=batch['input_ids'].to(self.device),
                        attention_mask=batch['attention_mask'].to(self.device),
                        max_new_tokens=constants.MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=temperature,
                        return_dict_in_generate=True,
                    )
                    
                    responses = self.tokenizer.batch_decode(outputs.sequences, clean_up_tokenization_spaces=False)
                    batch_predictions.append([self._process_response(r) for r in responses])
                    
                
                # Process each example in batch
                for i in range(len(batch['input_ids'])):
                    example_preds = [preds[i] for preds in batch_predictions]
                    
                    pred_counts = {}
                    for pred in example_preds:
                        if pred in pred_counts:
                            pred_counts[pred] += 1
                        else:
                            pred_counts[pred] = 1
                    
                    majority_pred = max(pred_counts.items(), key=lambda x: x[1])[0]
                    
                    final_predictions.append(majority_pred)
                
        return self._compute_metrics(final_predictions, dataset), final_predictions
    
    def _collate_fn(self, batch):
        """Collate function for dataloader"""
        max_input_length = max(len(entry['input_ids']) for entry in batch)
        
        batch_size = len(batch)
        input_token_ids = np.zeros((batch_size, max_input_length), dtype=np.int64)
        attention_mask = np.zeros((batch_size, max_input_length), dtype=np.int64)
        
        for idx, entry in enumerate(batch):
            in_length = len(entry['input_ids'])
            input_token_ids[idx, :in_length] = entry['input_ids']
            attention_mask[idx, :in_length] = entry['attention_mask']
        
        return {
            'input_ids': torch.tensor(input_token_ids),
            'attention_mask': torch.tensor(attention_mask)
        }
        
    def _compute_metrics(self, predictions: List[str], dataset) -> Dict:
        """Compute evaluation metrics"""
        prec, rec, f1, accuracy = compute_prediction_scores(predictions, dataset)
        
        metrics = {
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'accuracy': accuracy
        }
        return metrics
        
    def save_predictions(self, predictions: List[str], dataset, output_path: str):
        """Save predictions to file"""
        output = []
        for idx, pred in enumerate(predictions):
            output.append({
                'prediction': pred,
                'label': dataset[idx]['output_seq']
            })
            
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)


def main():
    args = mk_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize evaluator
    evaluator = T5Evaluator(
        model_path=args.model_path,
        batch_size=args.batch_size
    )
    
    # Load test dataset
    test_dataset = CustomMwozDataset(
        evaluator.tokenizer,
        data_filename=f'{constants.DATA_DIR}test.json',
        model_type='t5',
        mode='test',
        response_pred=False  # Entity type prediction
    ).data
    
    print("Evaluating T5 model...")
    
    # Evaluate with different decoding strategies
    print("\nGreedy decoding:")
    greedy_metrics, greedy_preds= evaluator.evaluate_greedy(test_dataset)
    print(json.dumps(greedy_metrics, indent=2))
    evaluator.save_predictions(
        greedy_preds,
        test_dataset,
        os.path.join(args.output_dir, 'greedy_predictions.json')
    )
    
    print(f"\nBeam search (num_beams={args.num_beams}):")
    beam_metrics, beam_preds = evaluator.evaluate_beam_search(test_dataset, args.num_beams)
    print(json.dumps(beam_metrics, indent=2))
    evaluator.save_predictions(
        beam_preds,
        test_dataset,
        os.path.join(args.output_dir, 'beam_predictions.json')
    )
    
    print(f"\nSelf-consistency (samples={args.num_samples}, temp={args.temperature}):")
    sc_metrics, sc_preds = evaluator.evaluate_self_consistency(
        test_dataset,
        args.num_samples,
        args.temperature
    )
    print(json.dumps(sc_metrics, indent=2))
    evaluator.save_predictions(
        sc_preds,
        test_dataset,
        os.path.join(args.output_dir, 'sc_predictions.json')
    )

    # Save all metrics
    all_metrics = {
        'greedy': greedy_metrics,
        'beam_search': beam_metrics,
        'self_consistency': sc_metrics
    }
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    main()