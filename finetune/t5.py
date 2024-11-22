import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import set_seed, Trainer, T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments
from common.mwoz_data import CustomMwozDataset
from common import constants
from common.utils import compute_prediction_scores


# Set for reproducibility
np.random.seed(constants.SEED)
torch.manual_seed(constants.SEED)
set_seed(constants.SEED)


class Seq2SeqTrainer(Trainer):
    
    # Decodes output with the tokenizer 
    def decode_responses(self, outputs):
        preds = []
        responses = self.tokenizer.batch_decode(outputs.sequences.to('cpu'), clean_up_tokenization_spaces=False)
        for response in responses:
            preds.append(response.split("<pad>", 1)[-1].strip().split("</s>")[0].strip())

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
                                         max_new_tokens=constants.MAX_NEW_TOKENS,
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
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained(constants.T5_MODEL_ID, model_max_length=None)
        self.model = T5ForConditionalGeneration.from_pretrained(constants.T5_MODEL_ID)
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
        train_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}train.json', model_type='t5', mode='train')
        validation_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}valid.json', model_type='t5', mode='eval')

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
            output_dir=constants.T5_TRAIN_OUTPUT_DIR,
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
        )

        # Trigger training and evaluation on validation set
        trainer.train()

        # Get test performance
        test_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}test.json', model_type='t5', mode='test')
        test_results = trainer.evaluate(test_set, save_results=True, result_path=constants.T5_TEST_RESULT_FILE, metric_key_prefix='test')
        print(test_results)