import re
import torch
import pickle
import numpy as np
from os.path import join
from torch.utils.data import Dataset

from typing import List

from .dataloader import EEDataloader
from .input_example import InputExample
from .common import (
    logger,
    EE_label2id,
    RAW_LABEL2ID,
    NER_PAD
)


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
        """Preprocessor.
        Convert labels to multi-head label matrices of shape (n_cls, L, L)
        Tokenizes and converts words to ids.

        Args:
            examples (List[InputExample]): List of input examples
            tokenizer (_type_): Bert tokenizer

        Returns:
            list: returns the list of data to be stored as the dataset
        """
        is_test = examples[0].entities is None
        data = []

        for example in examples:
            if is_test:
                _sent_id, text = example.to_begin_end_label_tuples()
                labels = None
            else:
                _sent_id, text, labels = example.to_begin_end_label_tuples()
            
            # tokenization
            tokens = []
            for word in text:
                token = tokenizer.tokenize(word)
                if not token:
                    token = [tokenizer.unk_token]
                assert len(token) == 1
                tokens.extend(token)

            # add [CLS] and [SEP]
            tokens = [tokenizer.cls_token] + tokens[: self.max_length - 2] + [tokenizer.sep_token]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            # create label matrix
            if is_test:
                label_matrix = None
            else:
                label_matrix = np.zeros((len(RAW_LABEL2ID), len(token_ids), len(token_ids)))
                for start, end, lid in labels:
                    if start <= end and end < self.max_length - 2:
                        # offset 1 due to [CLS] token at the beginning
                        label_matrix[lid, start + 1, end + 1] = 1
            data.append((token_ids, label_matrix))
        
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CollateFnForGlobalPtr:
    def __init__(self, pad_token_id: int, label_pad_token_id: int = EE_label2id[NER_PAD], for_nested_ner: bool = False):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, batch) -> dict:
        # items in batch:
        # input_ids, label_matrix
        # label is None if is not training set
        input_ids = [item[0] for item in batch]
        label_matrix = [item[1] for item in batch] if batch[0][1] is not None else None
        max_len = max(map(len, input_ids))
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

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
