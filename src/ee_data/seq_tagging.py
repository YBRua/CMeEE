import re
import torch
import pickle
from itertools import repeat
from os.path import join, exists
from torch.utils.data import Dataset

from typing import List

from .dataloader import EEDataloader
from .input_example import InputExample
from .common import (
    logger,
    EE_label2id1, EE_label2id2, EE_label2id,
    NO_ENT, NER_PAD
)


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

        # if False and exists(cache_file):
        #     with open(cache_file, "rb") as f:
        #         self.examples, self.data = pickle.load(f)
        #     logger.info(f"Load cached data from {cache_file}")
        # else:
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


class CollateFnForSeqTag:
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
