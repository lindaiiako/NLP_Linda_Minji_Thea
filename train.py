import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="t5")
    parser.add_argument('--device', type=str, default="1")
    return parser.parse_args()


def main(args):
    # Set device #
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Initializer trainer
    if args.model_type == 't5':
        from finetune.t5 import T5Trainer
        trainer = T5Trainer()
    else:
        from finetune.llama_it import LlamaTrainer
        trainer = LlamaTrainer()

    # Start training and eval
    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)