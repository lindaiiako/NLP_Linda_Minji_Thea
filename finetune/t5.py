import os
import torch
import numpy as np
from transformers import set_seed
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments
from common.mwoz_data import CustomMwozDataset
from common import constants
from finetune.seq2seq import Seq2SeqTrainer


# Set for reproducibility
np.random.seed(constants.SEED)
torch.manual_seed(constants.SEED)
set_seed(constants.SEED)

class T5Trainer():
    # Loads T5 tokenizer and model
    def __init__(self):
        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(constants.MODEL_NAME, model_max_length=None)
        self.model = T5ForConditionalGeneration.from_pretrained(constants.MODEL_NAME)

        # Push to GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"    # Temporary setting
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
        train_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}train.json', mode='train')
        validation_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}valid.json', mode='eval')

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
            output_dir=constants.TRAIN_OUTPUT_DIR,
            overwrite_output_dir=True,  
            remove_unused_columns=False,
            log_level='warning',
            logging_steps=10,
            save_strategy='steps', 
            save_steps=200,
            seed=constants.SEED,
            evaluation_strategy='steps',
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
            data_collator=self.seq2seq_collate_fn,
            args=training_args,
            train_dataset=train_set, eval_dataset=validation_set,
        )

        # Trigger training and evaluate on validation set
        trainer.train()

        # Get test performance
        test_set = CustomMwozDataset(self.tokenizer, data_filename=f'{constants.DATA_DIR}test.json', mode='eval')
        test_results = trainer.evaluate(test_set, save_results=True, result_path=constants.TEST_RESULT_FILE)
        print(test_results)