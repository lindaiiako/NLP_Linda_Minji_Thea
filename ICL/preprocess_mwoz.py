import json
import argparse
from common import utils

def mk_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    return parser.parse_args()

def main(args):
    data_path = f"./data/{args.mode}.json"
    with open(data_path, "r") as f:
        data_list = json.load(f)

    result_dataset = []
    conversations = []

    for idx, data in enumerate(data_list):
        context = data["context"]
        episode = []
        for idx, turn in enumerate(context):
            if idx % 2 == 1:
                episode.append('ASSISTANT: ' + turn)
            else:
                episode.append('USER: ' + turn)
        conversations.append(episode)

    # entity types
    output_seq = []
    for idx, data in enumerate(data_list):
        entity_types = data['hints']['entity_types']
        if len(entity_types) > 0:
            output = ' | '.join(sorted(entity_types))
        else:
            output = '[no entity]'
        output_seq.append(output)

    responses = [data['output'] for data in data_list]

    # DB
    database = []
    for idx, data in enumerate(data_list):
        kb = data['kb']
        data_attributes = list(kb[0].keys())
        formatted_kb = utils.flatten_kb(kb, data_attributes).rstrip(']').lstrip('[').replace('\"', ' ')
        database.append(formatted_kb)


    for idx in range(len(data_list)):
        result_dataset.append({
            'context': conversations[idx],
            'output': responses[idx],
            'output_seq': output_seq[idx],
            'database': database[idx]
        })

    with open(f'data/{args.mode}_demon.json', 'w') as f:
        json.dump(result_dataset, f, indent=4)

    print(len(result_dataset))

if __name__ == "__main__":
    args = mk_parse()
    main(args)