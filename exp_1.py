from finetune.t5 import T5Trainer

# Initializer trainer
trainer = T5Trainer()

# Start training (include eval on test set)
trainer.train()