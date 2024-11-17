import numpy as np
from copy import deepcopy
from tqdm import trange
from tqdm import tqdm
import json

import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

from common.mwoz_data import CustomMwozDataset
from common import constants

GENERAL_INSTRUCTION = """
Assume you are a customer support expert. I will provide you with a [dialog] between a user and a customer service representative, along with a database [kb] of relevant entities. 
Your task is to generate the entity types from the [kb] that should be included in the response to the user based on the dialog. 
The entity types correspond to the keys in the [kb], such as: name, address, area, phone, postcode, food, pricerange, etc.
If no entities from the [kb] are relevant to the response, answer with: [no entity].
Based on the [dialog] and [kb], generate entity types to be included in the response, using the format:
entity_type1 | entity_type2 | ...

Here are the examples: """.strip()

class Retriever:
    def __init__(self, samples=None):
        self.documents = samples
        self.vectors = []
        self.device = 'cuda'
        if torch.cuda.device_count() == 0:
            self.device = 'cpu'
            # TODO : Change the engine to vllm
        self.model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
        self.model.eval()

        contexts = [x['input_seq'] for x in samples]
        self.hint_min = 30

        vectors = np.zeros((len(contexts), 1024)) # batch embedding and save in self.vectors
        for st in tqdm(range(0, len(contexts), 128), desc="Computing embeddings"):
            en = min(len(contexts), st + 128)
            tout = self.tokenizer(contexts[st:en], return_tensors='pt', padding=True, truncation=True)
            tout = dict([(k, v.to(self.device)) for k, v in tout.items()])
            with torch.no_grad():
                ret = self.model(**tout)
            embs = ret[0][:, 0]
            embs = F.normalize(embs, p=2, dim=1)
            vectors[st:en, :] = embs.to("cpu").numpy()

        self.vectors = vectors
        print(f'Total documents', len(vectors), self.vectors.shape)

    def compute_scores(self, text): # query embedding and compute cosine similarity with self.vectors
        tout = self.tokenizer([text], return_tensors='pt', padding=True, truncation=True)
        tout = dict([(k, v.to(self.device)) for k, v in tout.items()])
        with torch.no_grad():
            ret = self.model(**tout)

        embs = ret[0][:, 0]
        embs = F.normalize(embs, p=2, dim=1)
        qvec = embs.to("cpu").numpy()
        scores = np.matmul(self.vectors, qvec.T)[:, 0]

        return scores

    def get_top_k(self, text, k=3): # get top k (can except uuid and etype)
        scores = self.compute_scores(text)

        rets = []
        idxs = np.argsort(scores)[::-1]
        idx_sofar = set()
        for ii in idxs:
            if ii in idx_sofar:
                continue
            idx_sofar.add(ii)
            rets.append(deepcopy(self.documents[ii]))
            if len(rets) == k:
                break
        return rets

    def gen_prompt(self, text, k=3):
        prompt = GENERAL_INSTRUCTION
        prompt += "\n"

        top_k_examples = self.get_top_k(text, k)
        for examples in top_k_examples:
            prompt += f"Query : {examples['input_seq']}\n"
            prompt += f"Response : {examples['output_seq']}\n"
            prompt += "\n"
        
        prompt += f"Query : {text}\n"
        prompt += "Response : "
        return prompt
