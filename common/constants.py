
ET_LABELS_KEY_TO_VAL = {0: 'address', 1: 'area', 2: 'food', 3: 'name', 4: 'phone', 5: 'postcode', 6: 'pricerange', 7: 'type', 
                                           8: 'ref', 9: 'choice', 10: 'stars', 11: 'entrance_fee', 12: 'none'}
ET_LABELS_VAL_TO_KEY = {v: k for k, v in ET_LABELS_KEY_TO_VAL.items()}
NUM_CLASSES = len(ET_LABELS_KEY_TO_VAL)

SEED = 43

DATA_DIR = 'data/'


# Configs for finetuning
MAX_NEW_TOKENS = 128
MODEL_ID = {
    "t5": "google/flan-t5-large",
    "gemma": "google/gemma-2-9b-it", 
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3"
}
FT_MODEL = {
    "gemma": "checkpoints/gemma-ft",
    "llama": "checkpoints/llama-ft",
    "mistral": "checkpoints/mistral-ft",
    "gemma-response-pred": "checkpoints/gemma-response-pred-ft",
    "mistral-response-pred": "checkpoints/mistral-response-pred-ft"
}
MERGED_MODEL = {
    "gemma": "checkpoints/gemma-merged",
    "llama": "checkpoints/llama-merged",
    "mistral": "checkpoints/mistral-merged",
    "gemma-response-pred": "checkpoints/gemma-response-pred-merged",
    "mistral-response-pred": "checkpoints/mistral-response-pred-merged"
}
TRAIN_OUTPUT_DIR = {
    "t5": "train_output/t5/",
    "t5-response-pred": "train_output/t5-response-pred/",
    "gemma": "train_output/gemma/",
    "llama": "train_output/llama/",
    "mistral": "train_output/mistral/",
    "gemma-response-pred": "train_output/gemma-response-pred/",
    "mistral-response-pred": "train_output/mistral-response-pred/"
}
TEST_RESULT_FILE = {
    "t5": "predictions/t5.json",
    "t5-response-pred": "predictions/t5-response-pred.json",
    "gemma": "predictions/gemma.json",
    "llama": "predictions/llama.json",
    "mistral": "predictions/mistral.json",
    "gemma-response-pred": "predictions/gemma-response-pred.json",
    "mistral-response-pred": "predictions/mistral-response-pred.json"
}