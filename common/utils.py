import numpy as np
import torch.nn.functional as F
import json
from sklearn.metrics import f1_score, accuracy_score
from common import constants
from collections import Counter


# Encodes list of labels from text format list to one-hot format
def one_hot_encode(labels, delimeter='|'):
    one_hot_label = np.zeros(constants.NUM_CLASSES, dtype=np.float32)

    for label in labels.split(delimeter):
        label_id = constants.ET_LABELS_VAL_TO_KEY.get(label.strip())
        if label_id is not None:
            one_hot_label[label_id] = 1.0

    return one_hot_label


# Flattens kb entries from the json file into a list
def flatten_kb(kb, data_attributes):
    knowledge_seqence = []
    for entry in kb:
        tmp = dict()
        for f in data_attributes:
            tmp[f] = entry.get(f, 'none')
        knowledge_seqence.append(json.dumps(tmp))

    return "[" + ",".join(knowledge_seqence) + "]"


# Returns entity types list as a set with index #
# Sample input: ['food', 'name', 'pricerange', 'type', 'name']
# Sample output: {'food_0', 'name_0', 'name_1', 'pricerange_0', 'type_0'}
def get_indexed_entity_types(entity_types):
    cnts = Counter(entity_types).most_common()
    ret = set()
    for et, cnt in cnts:
        for jj in range(cnt):
            ret.add(f"{et}_{jj}")

    return ret


# Computes F1 score and accuracy for entity type prediction
# Note: entity type is a list (e.g. "area | choice")
def compute_prediction_scores(preds, eval_dataset, delimiter='|'):
    tp, fn, fp = 0, 0, 0
    corr = 0

    for idx in range(len(preds)):
        gold = eval_dataset[idx]['output_seq']
        pred = preds[idx]

        if '[no entity]' in pred:
            pred = '[no entity]'

        entities_gold = get_indexed_entity_types([x.strip() for x in gold.split(delimiter)])
        entities_predicted = get_indexed_entity_types([x.strip() for x in pred.split(delimiter)])

        ttp = len(entities_gold.intersection(entities_predicted))
        tfp = len(entities_predicted - entities_gold)
        tfn = len(entities_gold - entities_predicted)

        if tfp == 0 and tfn == 0:
            corr += 1

        tp += ttp
        fp += tfp
        fn += tfn

    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0
    accuracy = corr / len(eval_dataset)

    return prec, rec, f1, accuracy


# Computes average confidence score for text-based format outputs using logits scores
# Note: assumes input is 1 sample (not batch)
def decode_with_confidence_score(generation_outputs, tokenizer):
    generated_sequence = generation_outputs.sequences[0]
    
    # Get predicted text
    decoded_seq = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=False)
    predicted_text = decoded_seq.split("<pad>", 1)[-1].strip().split("</s>")[0].strip()

    # Get logits for each token
    logits = generation_outputs.scores                    
    
    # Compute confidence scores for each token
    probs = []
    for idx in range(len(logits)):
        token_logits = logits[idx]                        # logits of this token
        token_id = generated_sequence[idx+1].item()       # corresponding token ID in T5

        # TODO: add condition to check only for expected labels?
        if token_id != tokenizer.pad_token_id and token_id != tokenizer.eos_token_id:
            probabilities = F.softmax(token_logits, dim=-1) 
            prob = probabilities[0, token_id]    
            probs.append(prob)

    # Calculate average confidence score for the entire sequence
    if len(probs) > 0:
        avg_confidence_score = sum(probs) / len(probs)
    else:
        avg_confidence_score = "unknown"                # maybe better than 0?

    # Output results
    print(f"Model Prediction: {predicted_text} (Confidence: {avg_confidence_score})")
    
    return predicted_text, avg_confidence_score