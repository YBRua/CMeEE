
import json
import logging
import pickle
import re
from collections import Counter
from itertools import repeat
from os.path import join, exists
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

NER_PAD, NO_ENT = '[PAD]', 'O'

LABEL1 = ['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'dis', 'bod']  # 按照出现频率从低到高排序
LABEL2 = ['sym']

LABEL  = ['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'sym', 'dis', 'bod'] 

# 标签出现频率映射，从低到高
_LABEL_RANK = {L: i for i, L in enumerate(['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'sym', 'dis', 'bod'])}

EE_id2label1 = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL1 for P in ("B", "I")]
EE_id2label2 = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL2 for P in ("B", "I")]
EE_id2label  = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL  for P in ("B", "I")]

EE_label2id1 = {b: a for a, b in enumerate(EE_id2label1)}
EE_label2id2 = {b: a for a, b in enumerate(EE_id2label2)}
EE_label2id  = {b: a for a, b in enumerate(EE_id2label)}

EE_NUM_LABELS1 = len(EE_id2label1)
EE_NUM_LABELS2 = len(EE_id2label2)
EE_NUM_LABELS  = len(EE_id2label)


class InputExample:
    def __init__(self, sentence_id: str, text: str, entities: List[dict] = None):
        self.sentence_id = sentence_id
        self.text = text
        self.entities = entities

    def to_ner_task(self, for_nested_ner: bool = False):    
        '''NOTE: This function is what you need to modify for Nested NER.
        '''
        if self.entities is None:
            return self.sentence_id, self.text
        else:
            if not for_nested_ner:
                label = [NO_ENT] * len(self.text)
            else:
                label1 = [NO_ENT] * len(self.text)
                label2 = [NO_ENT] * len(self.text)

            def _write_label(_label: list, _type: str, _start: int, _end: int):
                for i in range(_start, _end + 1):
                    if i == _start:
                        _label[i] = f"B-{_type}"
                    else:
                        _label[i] = f"I-{_type}"

            for entity in self.entities:
                start_idx = entity["start_idx"]
                end_idx = entity["end_idx"]
                entity_type = entity["type"]

                assert entity["entity"] == self.text[start_idx: end_idx + 1], f"{entity} mismatch: `{self.text}`"

                if not for_nested_ner:
                    _write_label(label, entity_type, start_idx, end_idx)
                else:
                    '''NOTE

                    '''

            if not for_nested_ner:
                return self.sentence_id, self.text, label
            else:
                return self.sentence_id, self.text, label1, label2


class EEDataloader:
    def __init__(self, cblue_root: str):
        self.cblue_root = cblue_root
        self.data_root = join(cblue_root, "CMeEE")

    @staticmethod
    def _load_json(filename: str) -> List[dict]:
        with open(filename, encoding="utf8") as f:
            return json.load(f)

    @staticmethod
    def _parse(cmeee_data: List[dict]) -> List[InputExample]:
        return [InputExample(sentence_id=str(i), **data) for i, data in enumerate(cmeee_data)]

    def get_data(self, mode: str):
        if mode not in ("train", "dev", "test"):
            raise ValueError(f"Unrecognized mode: {mode}")
        return self._parse(self._load_json(join(self.data_root, f"CMeEE_{mode}.json")))


class EEDataset(Dataset):
    def __init__(self, cblue_root: str, mode: str, max_length: int, tokenizer, for_nested_ner: bool):
        self.cblue_root = cblue_root
        self.data_root = join(cblue_root, "CMeEE")
        self.max_length = max_length
        self.for_nested_ner = for_nested_ner

        # This flag is used in CRF
        self.no_decode = mode.lower() == "train"

        # Processed data can vary from using different tokenizers
        _tk_class = re.match("<class '.*\.(.*)'>", str(type(tokenizer))).group(1)
        _head = 2 if for_nested_ner else 1
        cache_file = join(self.data_root, f"cache_{mode}_{max_length}_{_tk_class}_{_head}head.pkl")

        if False and exists(cache_file):
            with open(cache_file, "rb") as f:
                self.examples, self.data = pickle.load(f)
            logger.info(f"Load cached data from {cache_file}")
        else:
            self.examples = EEDataloader(cblue_root).get_data(mode) # get original data
            self.data = self._preprocess(self.examples, tokenizer) # preprocess
            with open(cache_file, 'wb') as f:
                pickle.dump((self.examples, self.data), f)
            logger.info(f"Cache data to {cache_file}")

    def _preprocess(self, examples: List[InputExample], tokenizer) -> list:
        '''NOTE: This function is what you need to modify for Nested NER.
        '''
        is_test = examples[0].entities is None
        data = []

        label2id = EE_label2id

        for example in examples:
            if is_test:
                _sentence_id, text = example.to_ner_task(self.for_nested_ner)
                label = repeat(None, len(text))
            else:
                _sentence_id, text, label = example.to_ner_task(self.for_nested_ner)

            tokens = []
            label_ids = None if is_test else []
            
            for word, L in zip(text, label):
                token = tokenizer.tokenize(word)
                if not token:
                    token = [tokenizer.unk_token]
                tokens.extend(token)

                if not is_test:
                    label_ids.extend([label2id[L]] + [tokenizer.pad_token_id] * (len(token) - 1))

            
            tokens = [tokenizer.cls_token] + tokens[: self.max_length - 2] + [tokenizer.sep_token]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            if not is_test:
                label_ids = [label2id[NO_ENT]] + label_ids[: self.max_length - 2] + [label2id[NO_ENT]]
                
                data.append((token_ids, label_ids))
            else:
                data.append((token_ids,))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.no_decode


class CollateFnForEE:
    def __init__(self, pad_token_id: int, label_pad_token_id: int = EE_label2id[NER_PAD], for_nested_ner: bool = False):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.for_nested_ner = for_nested_ner
       
    def __call__(self, batch) -> dict:
        '''NOTE: This function is what you need to modify for Nested NER.
        '''
        inputs = [x[0] for x in batch]
        no_decode_flag = batch[0][1]

        input_ids = [x[0]  for x in inputs]
        labels    = [x[1]  for x in inputs] if len(inputs[0]) > 1 else None
    
        max_len = max(map(len, input_ids))
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, _ids in enumerate(input_ids):
            attention_mask[i][:len(_ids)] = 1
            _delta_len = max_len - len(_ids)
            input_ids[i] += [self.pad_token_id] * _delta_len
            
            if labels is not None:
                labels[i] += [self.label_pad_token_id] * _delta_len

        if not self.for_nested_ner:
            inputs = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": attention_mask,
                "labels": torch.tensor(labels, dtype=torch.long) if labels is not None else None,
                "no_decode": no_decode_flag
            }
        else:
            inputs = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": attention_mask,
                "labels": None, # modify this
                "labels2": None, # modify this
                "no_decode": no_decode_flag
            }

        return inputs


if __name__ == '__main__':
    import os
    from os.path import expanduser
    from transformers import BertTokenizer

   
    MODEL_NAME = "../bert-base-chinese"
    CBLUE_ROOT = "../data/CBLUEDatasets"

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    dataset = EEDataset(CBLUE_ROOT, mode="dev", max_length=10, tokenizer=tokenizer, for_nested_ner=False)

    batch = [dataset[0], dataset[1], dataset[2]]
    inputs = CollateFnForEE(pad_token_id=tokenizer.pad_token_id, for_nested_ner=False)(batch)
    print(inputs)
