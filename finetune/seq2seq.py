import torch
import json
from transformers import Trainer
from torch.utils.data import DataLoader
from common.utils import compute_prediction_scores


class Seq2SeqTrainer(Trainer):
    
    # Decodes output with the tokenizer 
    def decode_responses(self, outputs):
        preds = []
        responses = self.tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=False)
        for response in responses:
            preds.append(response.split("<pad>", 1)[-1].strip().split("</s>")[0].strip())

        return preds

    # Obtains predictions with greedy decoding
    def get_preds(self, dataset):
        max_new_tokens = 128
        model = self._wrap_model(self.model, training=False)
        model.eval()

        dataloader = DataLoader(dataset,
                                batch_size=self.args.eval_batch_size,
                                collate_fn=self.data_collator,
                                num_workers=self.args.dataloader_num_workers,
                                sampler=torch.utils.data.SequentialSampler(dataset),        # Iterates data in order
        )

        # Temporarily set to eval mode
        model.eval()

        responses = []
        for inputs in dataloader:
            batch = dict([(k, v.to(self.model.device)) for k, v in inputs.items()])
            # Run predict            
            with torch.no_grad():
                outputs = model.generate(**batch,
                                         use_cache=True,
                                         max_new_tokens=max_new_tokens,
                                         do_sample=False,
                                         num_beams=1,
                                         )
            outputs = outputs.to('cpu')
            responses.extend(self.decode_responses(outputs))

        # Back to train mode
        model.train()
        
        # Ensure correct size
        final_responses = responses[:len(dataset)]

        return final_responses


    # Evaluates entity type prediction using F1 scores (used in validation and test set)
    def evaluate(self, 
                 eval_dataset=None, 
                 ignore_keys=None,
                 metric_key_prefix="eval", 
                 save_results=False, 
                 result_path=None
                 ):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset    # test set

        # Get predictions
        responses = self.get_preds(eval_dataset)

        # Compute scores
        prec, rec, f1, accuracy = compute_prediction_scores(responses, eval_dataset)
        
        metrics = dict()
        metrics[f"{metric_key_prefix}_f1"] = f1
        metrics[f"{metric_key_prefix}_prec"] = prec
        metrics[f"{metric_key_prefix}_rec"] = rec
        metrics[f"{metric_key_prefix}_acc"] = accuracy

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        print(f'Evaluation results: {json.dumps(metrics, indent=2)}')

        if save_results:
            output = []
            for idx, resp in enumerate(responses):
                output.append(dict())
                output[-1]['uuid'] = eval_dataset.raw_data[idx]['uuid']
                output[-1]['turn_id'] = eval_dataset.raw_data[idx]['turn_id']
                tetypes = resp.split('|')
                tetypes = [x.strip() for x in tetypes if '[no entity]' != x]
                output[-1]['prediction'] = tetypes

            with open(result_path, 'w') as f:
                json.dump(output, f, indent=2)

        return metrics
    

