
ET_LABELS_KEY_TO_VAL = {0: 'address', 1: 'area', 2: 'food', 3: 'name', 4: 'phone', 5: 'postcode', 6: 'pricerange', 7: 'type', 
                                           8: 'ref', 9: 'choice', 10: 'stars', 11: 'entrance_fee', 12: 'none'}
ET_LABELS_VAL_TO_KEY = {v: k for k, v in ET_LABELS_KEY_TO_VAL.items()}
NUM_CLASSES = len(ET_LABELS_KEY_TO_VAL)

SEED = 43

DATA_DIR = 'data/'

MAX_NEW_TOKENS = 128

# T5 config
T5_MODEL_ID = "google/flan-t5-large"
T5_TRAIN_OUTPUT_DIR = 'train_output/t5/'
T5_TEST_RESULT_FILE = 'predictions/t5.json'


# Configs for LLM finetuning
MODEL_ID = {
    "gemma": "google/gemma-2-9b-it", 
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3"
}
FT_MODEL = {
    "gemma": "checkpoints/gemma-ft",
    "llama": "checkpoints/llama-ft",
    "mistral": "checkpoints/mistral-ft",
    "gemma-response-pred": "checkpoints/gemma-response-pred-ft"
}
MERGED_MODEL = {
    "gemma": "checkpoints/gemma-merged",
    "llama": "checkpoints/llama-merged",
    "mistral": "checkpoints/mistral-merged",
    "gemma-response-pred": "checkpoints/gemma-response-pred-merged"
}
TRAIN_OUTPUT_DIR = {
    "gemma": "train_output/gemma/",
    "llama": "train_output/llama/",
    "mistral": "train_output/mistral/",
    "gemma-response-pred": "train_output/gemma-response-pred/"
}
TEST_RESULT_FILE = {
    "gemma": "predictions/gemma.json",
    "llama": "predictions/llama.json",
    "mistral": "predictions/mistral.json",
    "gemma-response-pred": "predictions/gemma-response-pred.json"
}