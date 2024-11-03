import numpy as np
import torch.nn.functional as F
import json
from sklearn.metrics import f1_score, accuracy_score
from common import constants


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


# Computes accuracy and micro F1 score for entity type prediction
def compute_prediction_scores(preds, labels):
    # Convert text to multilabel class form
    
    if isinstance(preds, str):
        preds_multilabel = one_hot_encode(preds)
        labels_multilabel = one_hot_encode(labels)
    else:                  # for batched data
        preds_multilabel = [one_hot_encode(pred) for pred in preds]
        labels_multilabel = [one_hot_encode(label) for label in labels]

    print(preds_multilabel)
    print(labels_multilabel)

    accuracy = accuracy_score(y_true=labels_multilabel, y_pred=preds_multilabel)
    f1 = f1_score(y_true=labels_multilabel, y_pred=preds_multilabel, average="micro")*100
    
    # Return metrics as dict
    return {'f1': f1, 'accuracy': accuracy}

# TODO: different score computation function for purely text format with repetitions


# Computes average confidence score for text-based format outputs using logits scores
def get_prediction_with_confidence_score(generation_outputs, tokenizer):
    generated_sequence = generation_outputs.sequences[0]
    predicted_text = tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    logits = generation_outputs.scores                    # logits for each token
    
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
