import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import set_seed, Trainer, T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments
from common.mwoz_data import CustomMwozDataset
from common import constants
from common.utils import compute_prediction_scores
from synctod.text_process import preprocess_text
from synctod.metrics import compute_metrics


# Set for reproducibility
np.random.seed(constants.SEED)
torch.manual_seed(constants.SEED)
set_seed(constants.SEED)


class Seq2SeqTrainer(Trainer):
    def __init__(self, is_response_prediction, **kwargs):
        super().__init__(**kwargs)
        self.is_response_prediction = is_response_prediction
        if self.is_response_prediction:
            self.max_new_tokens = 384
        else:
            self.max_new_tokens = constants.MAX_NEW_TOKENS


    # Decodes output with the tokenizer 
    def decode_responses(self, outputs):
        preds = []
        responses = self.tokenizer.batch_decode(outputs.sequences.to('cpu'), clean_up_tokenization_spaces=False)
        for response in responses:
            cleaned = response.split("<pad>", 1)[-1].strip().split("</s>")[0].strip()
            if self.is_response_prediction:
                if "<sys>" in cleaned:
                    preds.append(preprocess_text(cleaned.split("<sys>")[1].strip()))
                else:
                    preds.append(preprocess_text(cleaned))
            else:
                preds.append(cleaned)

        return preds


    # Obtains predictions with greedy decoding
    def get_preds(self, dataset):
        model = self._wrap_model(self.model, training=False)
        # Temporarily set to eval mode
        model.eval()

        dataloader = DataLoader(dataset,
                                batch_size=self.args.eval_batch_size,
                                collate_fn=self.data_collator,
                                sampler=torch.utils.data.SequentialSampler(dataset),
        )

        responses = []
        probs = [] 
        for inputs in dataloader:
            batch = dict([(k, v.to(self.model.device)) for k, v in inputs.items()])
            # Run predict with logits        
            with torch.no_grad():
                outputs = model.generate(**batch,
                                         max_new_tokens=self.max_new_tokens,
                                         do_sample=False,
                                         num_beams=1,
                                         return_dict_in_generate=True,
                                         output_scores=True
                                         )
            responses.extend(self.decode_responses(outputs))

            transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)

            # Obtain scores for selected tokens in greedy search
            probs.extend(np.exp(transition_scores.to('cpu').numpy()))


        # Back to train mode
        model.train()

        return responses, probs


    # Evaluates entity type prediction using F1 scores (used in validation and test set)
    # Or BLEU and Entity F1 for response prediction
    def evaluate(self, 
                 eval_dataset=None, 
                 ignore_keys=None,
                 metric_key_prefix="eval", 
                 save_results=False, 
                 result_path=None
                 ):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        # Get predictions
        responses, probs = self.get_preds(eval_dataset)

        if self.is_response_prediction:
            ground_truth_responses = []
            for entry in eval_dataset.raw_dataset:
                ground_truth_responses.append(entry['output'])
            
            # Compute scores
            results = compute_metrics(responses, ground_truth_responses, 'MultiWOZ', f'{constants.DATA_DIR}entities.json')

            metrics = dict()
            metrics[f"{metric_key_prefix}_bleu"] = results["bleu"]
            metrics[f"{metric_key_prefix}_f1"] = results["entity_f1"]
            metrics[f"{metric_key_prefix}_precision"] = results["entity_precision"]
            metrics[f"{metric_key_prefix}_recall"] = results["entity_recall"]
        else:
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
                output[-1]['uuid'] = eval_dataset.raw_dataset[idx]['uuid']
                output[-1]['turn_id'] = eval_dataset.raw_dataset[idx]['turn_id']
                if self.is_response_prediction:
                    output[-1]['ground_resp'] = ground_truth_responses[idx]
                    output[-1]['prediction'] = resp
                else:
                    tetypes = resp.split('|')
                    tetypes = [x.strip() for x in tetypes if '[no entity]' != x]
                    output[-1]['prediction'] = tetypes
                    # Get average score as proxy for confidence score
                    output[-1]['confidence'] = str(np.mean(probs[idx]))     

            with open(result_path, 'w') as f:
                json.dump(output, f, indent=2)

        return metrics


class T5Trainer():
    # Loads T5 tokenizer and model
    def __init__(self, response_prediction):
        self.is_response_prediction = response_prediction
        if self.is_response_prediction:
            self.dir_key = "t5-response-pred"
        print(f"Predicting response prediction? {self.is_response_prediction}")

        self.tokenizer = T5Tokenizer.from_pretrained(constants.MODEL_ID['t5'], model_max_length=None)
        self.model = T5ForConditionalGeneration.from_pretrained(constants.MODEL_ID['t5'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)


    # Standardizes input and output length per batch
    def seq2seq_collate_fn(self, batch):
        max_input_length = -1
        max_output_length = -1
        train_mode = 'labels' in batch[0]

        # Get max_input_length and max_output_length per batch
        for entry in batch:
            if max_input_length < len(entry['input_ids']):
                max_input_length = len(entry['input_ids'])

            if train_mode and max_output_length < len(entry['labels']):
                max_output_length = len(entry['labels'])

        assert max_input_length > 0

        batch_size = len(batch)
        input_token_ids = np.zeros((batch_size, max_input_length), dtype=np.int64)
        attention_mask = np.zeros((batch_size, max_input_length), dtype=np.int64)

        if train_mode:
            # Marker not to compute loss later on using -100 special value
            labels = np.ones((batch_size, max_output_length), dtype=np.int64) * (-100)

        for idx, entry in enumerate(batch):
            in_length = len(entry['input_ids'])
            input_token_ids[idx, :in_length] = entry['input_ids']
            attention_mask[idx, :in_length] = entry['attention_mask']

            if train_mode:
                out_length = len(entry['labels'])
                labels[idx, :out_length] = entry['labels']

        # Create final data batch
        input_token_ids = torch.tensor(input_token_ids)
        batch_dict = {
            "input_ids": input_token_ids,
            "attention_mask": attention_mask,
        }

        if train_mode:
            batch_dict['labels'] = torch.tensor(labels)

        for k in batch_dict:
            batch_dict[k] = torch.tensor(batch_dict[k])

        return batch_dict


    # Main training procedure
    def train(self):
        # Init dataset
        train_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}train.json', model_type='t5', mode='train', response_pred=self.is_response_prediction)
        validation_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}valid.json', model_type='t5', mode='eval', response_pred=self.is_response_prediction)

        # Init hyperparameters similar to SyncTOD to replicate results
        training_args = Seq2SeqTrainingArguments(
            gradient_accumulation_steps=4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=4,
            num_train_epochs=8,
            tf32=False,
            fp16=False,
            gradient_checkpointing=True,
            learning_rate=1e-4,
            warmup_ratio=0.1,
            output_dir=constants.TRAIN_OUTPUT_DIR[self.dir_key],
            overwrite_output_dir=True,  
            remove_unused_columns=False,
            log_level='warning',
            logging_steps=10,
            save_strategy='steps', 
            save_steps=200,
            seed=constants.SEED,
            eval_strategy='steps',
            eval_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True, 
            metric_for_best_model='f1',
            greater_is_better=True,         
            report_to='tensorboard',
            run_name='t5_exp',
        )

        # Create trainer
        trainer = Seq2SeqTrainer(
            model=self.model, 
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_set, 
            eval_dataset=validation_set,
            data_collator=self.seq2seq_collate_fn,
            is_response_prediction=self.is_response_prediction,
        )

        # Trigger training and evaluation on validation set
        trainer.train()

        # Get test performance
        test_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}test.json', model_type='t5', mode='test', response_pred=self.is_response_prediction)
        trainer.evaluate(test_set, save_results=True, result_path=constants.TEST_RESULT_FILE[self.dir_key], metric_key_prefix='test')