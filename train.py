import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gemma")
    parser.add_argument('--device', type=str, default="1")
    return parser.parse_args()


def main(args):
    # Set device number
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Initializer trainer
    if args.model == 't5':
        from finetune.t5 import T5Trainer
        trainer = T5Trainer()
    elif args.model == 'llama':
       from finetune.llama import LlamaTrainer
       trainer = LlamaTrainer(args.model)
    elif args.model == 'gemma':
        from finetune.gemma import GemmaTrainer
        trainer = GemmaTrainer(args.model)
    elif args.model == 'mistral':
        from finetune.mistral import MistralTrainer
        trainer = MistralTrainer(args.model)
    else:
        raise NotImplementedError
    
    # Start training and eval
    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)