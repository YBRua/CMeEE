
import json
import logging
import pickle
import re
from itertools import repeat
from os.path import join, exists
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

NER_PAD, NO_ENT = '[PAD]', 'O'
MAX_LEN = 256

LABEL1 = ['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'dis', 'bod']  # 按照出现频率从低到高排序
LABEL2 = ['sym']

LABEL  = ['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'sym', 'dis', 'bod']

# 标签出现频率映射，从低到高
_LABEL_RANK = {L: i for i, L in enumerate(['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'sym', 'dis', 'bod'])}

EE_id2label1 = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL1 for P in ("B", "I")]
EE_id2label2 = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL2 for P in ("B", "I")]
EE_id2label  = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL  for P in ("B", "I")]

RAW_ID2LABEL = [L for L in LABEL]
RAW_LABEL2ID = {L: i for i, L in enumerate(RAW_ID2LABEL)}

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

    def to_global_pointer_task(self):
        """将输入实体转换为 Global Pointer 建模方式使用的输入

        Returns:
            - 训练集返回 sentence_id, text, label
                - sentence_id 是 sentence 的唯一标识
                - text 是 sentence 的文本
                - label 是一个 List of Tuple，每个 Tuple 包含实体的起点、终点和实体类型ID
            - 测试集仅返回 sentence_id, text
        """
        if self.entities is None:
            return self.sentence_id, self.text
        else:
            labels = []
            for e in self.entities:
                start, end, label = e['start_idx'], e['end_idx'], e['type']
                if start <= end:
                    labels.append((start, end, RAW_LABEL2ID[label]))
            return self.sentence_id, self.text, labels

    def to_word_pair_task(self):
        pass

    def to_ner_task(self, for_nested_ner: bool = False):    
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
                    # label2 contains and only contains 'sym' classes
                    if 'sym' in entity_type:
                        _write_label(label2, entity_type, start_idx, end_idx)
                    else:
                        _write_label(label1, entity_type, start_idx, end_idx)

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


class GlobalPtrDataset(Dataset):
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
        cache_file = join(self.data_root, f"cache_gp_{mode}_{max_length}_{_tk_class}_{_head}head.pkl")

        self.examples = EEDataloader(cblue_root).get_data(mode) # get original data
        self.data = self._preprocess(self.examples, tokenizer) # preprocess
        with open(cache_file, 'wb') as f:
            pickle.dump((self.examples, self.data), f)
        logger.info(f"Cache data to {cache_file}")

    def _preprocess(self, examples: List[InputExample], tokenizer) -> List[dict]:
        is_test = examples[0].entities is None
        data = []

        for example in examples:
            if is_test:
                _sent_id, text = example.to_global_pointer_task()
                labels = None
            else:
                _sent_id, text, labels = example.to_global_pointer_task()
            
            # tokenization
            token2char_span_mapping = tokenizer(
                text,
                return_offsets_mapping=True,
                max_length=MAX_LEN,
                truncation=True)["offset_mapping"]

            # mapping from character index to token id
            # <https://github.com/xhw205/GlobalPointer_torch/blob/main/data_loader.py>
            char2tok_start = {j[0]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
            char2tok_end = {j[-1] - 1: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
            encoder_txt = tokenizer.encode_plus(text, max_length=MAX_LEN, truncation=True)
            input_ids = encoder_txt["input_ids"]

            # create label matrix
            if is_test:
                label_matrix = None
            else:
                label_matrix = np.zeros((len(RAW_LABEL2ID), MAX_LEN, MAX_LEN))
                for start, end, lb_id in labels:
                    if start in char2tok_start and end in char2tok_end:
                        start = char2tok_start[start]
                        end = char2tok_end[end]
                        labels[lb_id, start, end] = 1
                label_matrix = label_matrix[:, :len(input_ids), :len(input_ids)]
            data.append((input_ids, label_matrix))
        
        return data


class SeqTagDataset(Dataset):
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
        is_test = examples[0].entities is None
        data = []

        if self.for_nested_ner:
            label2id1 = EE_label2id1
            label2id2 = EE_label2id2
        else:
            label2id = EE_label2id

        for example in examples:
            if is_test:
                _sentence_id, text = example.to_ner_task(self.for_nested_ner)
                label = repeat(None, len(text))
                label1 = repeat(None, len(text))
                label2 = repeat(None, len(text))
            else:
                if self.for_nested_ner:
                    # For NestedNER, label = (label1, label2)
                    _sentence_id, text, label1, label2 = example.to_ner_task(self.for_nested_ner)
                else:
                    _sentence_id, text, label = example.to_ner_task(self.for_nested_ner)

            tokens = []
            label_ids = None if is_test else []
            label1_ids = None if is_test else []
            label2_ids = None if is_test else []
            
            if self.for_nested_ner:
                for word, L1, L2 in zip(text, label1, label2):
                    # convert word to tokens
                    token = tokenizer.tokenize(word)
                    if not token:
                        token = [tokenizer.unk_token]
                    tokens.extend(token)

                    # convert label to label_ids
                    if not is_test:
                        label1_ids.extend([label2id1[L1]] + [tokenizer.pad_token_id] * (len(token) - 1))
                        label2_ids.extend([label2id2[L2]] + [tokenizer.pad_token_id] * (len(token) - 1))
            else:
                for word, L in zip(text, label):
                    token = tokenizer.tokenize(word)
                    if not token:
                        token = [tokenizer.unk_token]
                    tokens.extend(token)

                    if not is_test:
                        label_ids.extend([label2id[L]] + [tokenizer.pad_token_id] * (len(token) - 1))

            # add [CLS] and [SEP]
            tokens = [tokenizer.cls_token] + tokens[: self.max_length - 2] + [tokenizer.sep_token]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            # add labels for [CLS] and [SEP]
            if not is_test:
                if self.for_nested_ner:
                    label1_ids = [label2id1[NO_ENT]] + label1_ids[: self.max_length - 2] + [label2id1[NO_ENT]]
                    label2_ids = [label2id2[NO_ENT]] + label2_ids[: self.max_length - 2] + [label2id2[NO_ENT]]
                    data.append((token_ids, (label1_ids, label2_ids)))
                else:
                    label_ids = [label2id[NO_ENT]] + label_ids[: self.max_length - 2] + [label2id[NO_ENT]]
                    data.append((token_ids, label_ids))
            else:
                data.append((token_ids,))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.no_decode


class CollateFnForGlobalPtr:
    def __init__(self, pad_token_id: int, label_pad_token_id: int = EE_label2id[NER_PAD], for_nested_ner: bool = False):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, batch) -> dict:
        # items in batch:
        # input_ids, label_matrix
        # label is None if is not training set
        input_ids = [item[0] for item in batch]
        label_matrix = [item[1] for item in batch]
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        max_len = max(map(len, input_ids))
        # pad input ids and generate attention masks
        for i, _ids in enumerate(input_ids):
            attention_mask[i][:len(_ids)] = 1
            _delta_len = max_len - len(_ids)
            input_ids[i] += [self.pad_token_id] * _delta_len

        # pad labels, note that labels are of shape (n_cls, L, L)
        if label_matrix is not None:
            for i, l_mat in enumerate(label_matrix):
                current_len = l_mat.shape[-1]
                _delta_len = max_len - current_len
                label_matrix[i] = np.pad(
                    l_mat,
                    ((0, 0), (0, _delta_len), (0, _delta_len)),
                    'constant', constant_values=0)

        inputs = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_matrix, dtype=torch.long) if label_matrix is not None else None
        }

        return inputs


class CollateFnForNER:
    def __init__(self, pad_token_id: int, label_pad_token_id: int = EE_label2id[NER_PAD], for_nested_ner: bool = False):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.for_nested_ner = for_nested_ner
       
    def __call__(self, batch) -> dict:
        inputs = [x[0] for x in batch]
        no_decode_flag = batch[0][1]

        input_ids = [x[0]  for x in inputs]
        if self.for_nested_ner:
            labels1 = [x[1][0] for x in inputs] if len(inputs[0]) > 1 else None
            labels2 = [x[1][1] for x in inputs] if len(inputs[0]) > 1 else None
        else:
            labels = [x[1]  for x in inputs] if len(inputs[0]) > 1 else None
    
        max_len = max(map(len, input_ids))
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, _ids in enumerate(input_ids):
            attention_mask[i][:len(_ids)] = 1
            _delta_len = max_len - len(_ids)
            input_ids[i] += [self.pad_token_id] * _delta_len
            
            if self.for_nested_ner:
                if labels1 is not None:
                    labels1[i] += [self.label_pad_token_id] * _delta_len
                if labels2 is not None:
                    labels2[i] += [self.label_pad_token_id] * _delta_len
            else:
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
                "labels": torch.tensor(labels1, dtype=torch.long) if labels1 is not None else None,
                "labels2": torch.tensor(labels2, dtype=torch.long) if labels2 is not None else None,
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
    dataset = GlobalPtrDataset(CBLUE_ROOT, mode="dev", max_length=10, tokenizer=tokenizer, for_nested_ner=False)

    batch = [dataset[0], dataset[1], dataset[2]]
    inputs = CollateFnForGlobalPtr(pad_token_id=tokenizer.pad_token_id, for_nested_ner=False)(batch)
    print(inputs)
