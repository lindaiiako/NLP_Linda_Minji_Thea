import argparse
import os
from common import constants


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gemma", required=True)
    parser.add_argument('--model_type', type=str, default="it", required=False)
    parser.add_argument('--device', type=str, default="1", required=True)
    return parser.parse_args()


def main(args):
    # Set device #
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Initializer trainer
    if args.model == 't5':
        from finetune.t5 import T5Trainer
        trainer = T5Trainer()
    #elif args.model == 'llama':
    #   from finetune.llama_it import LlamaTrainer
    #    trainer = LlamaTrainer()
    elif args.model == 'gemma':
        from finetune.gemma import GemmaTrainer
        trainer = GemmaTrainer(args.model_type)
    # Start training and eval
    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)