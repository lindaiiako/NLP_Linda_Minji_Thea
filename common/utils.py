import numpy as np
import json
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

        entities_gold = set([x.strip() for x in gold.split(delimiter)])
        entities_predicted = set([x.strip() for x in pred.split(delimiter)])

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


# TODO: Add CoT if testing for self-consistency in decoding
def format_for_llama(prompt, input, output, mode, is_self_consistency):
    text = f"<|im_start|>system\n{prompt}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
    if mode != 'infer':
        text += f"{output}<|im_end|>\n"
    return text

# TODO: Add CoT if testing for self-consistency in decoding
def format_llama_using_chat_template(prompt, input, output, mode, tokenizer, is_self_consistency):
    add_gen_prompt = True
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": input},
        ]
    if mode != 'infer':
        messages.append({"role": "assistant", "content": output})
        add_gen_prompt = False

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=add_gen_prompt, tokenize=False)

    return prompt


def format_for_gemma(prompt, input, output, mode, is_self_consistency):
    if mode == 'infer' and is_self_consistency:
        text = f"<start_of_turn>user {prompt}{input}\nLet's think step-by-step.<end_of_turn>\n<start_of_turn>model\n"
    else:
        text = f"<start_of_turn>user {prompt}{input}<end_of_turn>\n<start_of_turn>model\n"
        if mode != 'infer':
            text += f"{output} <end_of_turn>"
    return text


def format_for_mistral(prompt, input, output, mode, is_self_consistency):
    if mode == 'infer' and is_self_consistency:
        text = f"""<s>[INST]{prompt}{input}\nLet's think step-by-step.[/INST] """
    else:
        text = f"""<s>[INST]{prompt}{input}[/INST] """
        if mode != 'infer':
            text += f"""{output}</s>"""
    return text