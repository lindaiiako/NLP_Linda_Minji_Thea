
ET_LABELS_KEY_TO_VAL = {0: 'address', 1: 'area', 2: 'food', 3: 'name', 4: 'phone', 5: 'postcode', 6: 'pricerange', 7: 'type', 
                                           8: 'ref', 9: 'choice', 10: 'stars', 11: 'entrance_fee', 12: 'none'}
ET_LABELS_VAL_TO_KEY = {v: k for k, v in ET_LABELS_KEY_TO_VAL.items()}
NUM_CLASSES = len(ET_LABELS_KEY_TO_VAL)

SEED = 43

DATA_DIR = 'data/'

MAX_NEW_TOKENS = 128

T5_MODEL_ID = "google/flan-t5-large"
T5_TRAIN_OUTPUT_DIR = 'train_output/t5/'
T5_TEST_RESULT_FILE = 'predictions/t5.json'


LLAMA_MODEL_ID = "meta-llama/Llama-3.2-3B" # "meta-llama/Llama-3.2-1B" #"meta-llama/Meta-Llama-3.1-8B"  
LLAMA_IT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_FT_MODEL_NAME = "llama-etpred-ft"
LLAMA_TRAIN_OUTPUT_DIR = 'train_output/llama/'
LLAMA_TEST_RESULT_FILE = 'predictions/llama.json'


GEMMA_MODEL_ID = {
    "base": "google/gemma-2-9b",
    "it": "google/gemma-2-9b-it"
}
GEMMA_FT_MODEL = {
    "base": "checkpoints/gemma-base-ft",
    "it": "checkpoints/gemma-it-ft"
}
GEMMA_MERGED_MODEL = {
    "base": "checkpoints/gemma-base-merged",
    "it": "checkpoints/gemma-it-merged"
}
GEMMA_TRAIN_OUTPUT_DIR = {
    "base": "train_output/gemma-base/",
    "it": "train_output/gemma-it/"
}
GEMMA_TEST_RESULT_FILE = {
    "base": "predictions/gemma-base-ft.json",
    "it": "predictions/gemma-it-ft.json"
}