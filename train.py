import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gemma")
    parser.add_argument('--device', type=str, default="1")
    parser.add_argument('--response_prediction', type=bool, default=False)
    return parser.parse_args()


def main(args):
    # Set device number
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Initializer trainer
    if args.model == 't5':
        from finetune.t5 import T5Trainer
        trainer = T5Trainer(args.response_prediction)
    elif args.model == 'llama':
       from finetune.llama import LlamaTrainer
       trainer = LlamaTrainer(args.model)
    elif args.model == 'gemma':
        if args.response_prediction:
            from finetune.gemma_direct_response import GemmaTrainer
        else:
            from finetune.gemma import GemmaTrainer
        trainer = GemmaTrainer(args.model)
    elif args.model == 'mistral':
        if args.response_prediction:
            from finetune.mistral_direct_response import MistralTrainer
        else:
            from finetune.mistral import MistralTrainer
        trainer = MistralTrainer(args.model)
    else:
        raise NotImplementedError
    
    # Start training and eval
    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)