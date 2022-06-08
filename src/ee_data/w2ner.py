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
    W2_LABEL2ID,
    W2_SUC,
    logger,
    EE_label2id,
    NER_PAD
)


dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class W2NERDataset(Dataset):
    def __init__(self, cblue_root: str, mode: str, max_length: int, tokenizer, for_nested_ner: bool):
        self.cblue_root = cblue_root
        self.data_root = join(cblue_root, "CMeEE")
        self.max_length = max_length
        self.for_nested_ner = for_nested_ner
        self.mode = mode

        # This flag is used in CRF
        self.no_decode = mode.lower() == "train"

        # Processed data can vary from using different tokenizers
        _tk_class = re.match("<class '.*\.(.*)'>", str(type(tokenizer))).group(1)
        _head = 2 if for_nested_ner else 1
        cache_file = join(self.data_root, f"cache_{mode}_{max_length}_{_tk_class}_{_head}head.pkl")

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
                _sent_id, text = example.to_begin_end_label_tuples(mode='w2')
                labels = None
            else:
                _sent_id, text, labels = example.to_begin_end_label_tuples(mode='w2')

            text = text[:self.max_length - 2]

            text_len = len(text)
            wordpair_label = np.zeros((text_len, text_len), dtype=np.int)  # label
            rel_pos = np.zeros((text_len, text_len), dtype=np.int)  # relative position
            grid_mask = np.ones((text_len, text_len), dtype=np.bool)

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

            # create relative positional embedding
            # we use 20 different relative positional embeddings
            # for relative distances at different ranges
            # <https://github.com/ljynlp/W2NER/blob/main/data_loader.py>
            assert text_len < 1000, f"Line too long ({text_len}): {text}"
            for k in range(text_len):
                rel_pos[k, :] += k
                rel_pos[:, k] -= k

            for i in range(text_len):
                for j in range(text_len):
                    if rel_pos[i, j] < 0:
                        rel_pos[i, j] = dis2idx[-rel_pos[i, j]] + 9
                    else:
                        rel_pos[i, j] = dis2idx[rel_pos[i, j]]
            rel_pos[rel_pos == 0] = 19

            # create word-pair matrix
            if is_test:
                wordpair_label = None
            else:
                for start, end, lid in labels:
                    if start <= end and end < self.max_length - 2:
                        # NOTE: do not consider [CLS] here because it will be dropped in the model
                        for idx in range(start, end):
                            wordpair_label[idx, idx + 1] = W2_LABEL2ID[W2_SUC]  # next-word
                        wordpair_label[end, start] = lid  # tail-head-type

            data.append((
                token_ids,
                text_len,
                rel_pos,
                grid_mask,
                wordpair_label
            ))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.no_decode


class CollateFnForW2NER:
    def __init__(
            self,
            pad_token_id: int,
            label_pad_token_id: int = EE_label2id[NER_PAD],
            for_nested_ner: bool = False):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.for_nested_ner = for_nested_ner

    def __call__(self, batch) -> dict:
        inputs = [b[0] for b in batch]
        no_decode_flag = batch[0][1]
        batch_size = len(inputs)

        # token_idss: B, L + 2
        # text_lens: B
        # rel_poss: B, L, L
        # grid_masks: B, L, L
        # w2_labels: B, L, L
        token_idss, text_lens, rel_poss, grid_masks, w2_labels = map(list, zip(*inputs))
        w2_labels = w2_labels if w2_labels[0] is not None else None

        max_text_len = max(text_lens)
        max_input_len = max(len(ids) for ids in token_idss)
        assert max_text_len + 2 == max_input_len, f'{max_text_len} + 2 != {max_input_len}'

        # pad inputs
        attention_mask = torch.zeros((len(batch), max_input_len), dtype=torch.long)
        for id, token_ids in enumerate(token_idss):
            attention_mask[id][:len(token_ids)] = 1
            _delta_len = max_input_len - len(token_ids)
            token_idss[id] += [self.pad_token_id] * _delta_len

        rel_poss = self._batched_pad(rel_poss, 0, max_text_len)
        grid_masks = self._batched_pad(grid_masks, False, max_text_len)
        
        if w2_labels is not None:
            w2_labels = self._batched_pad(w2_labels, self.label_pad_token_id, max_text_len)

        return {
            "input_ids": torch.tensor(token_idss, dtype=torch.long),
            "attention_mask": attention_mask.long(),
            "text_len": torch.tensor(text_lens, dtype=torch.long),
            "rel_pos": torch.tensor(rel_poss, dtype=torch.long),
            "grid_mask": torch.tensor(grid_masks, dtype=torch.bool),
            "labels": torch.tensor(w2_labels, dtype=torch.long) if w2_labels is not None else None,
            "no_decode": no_decode_flag
        }

    def _batched_pad(
            self,
            data: List,
            pad_val: int,
            target_len: int) -> np.ndarray:
        n_dims = len(data[0].shape)
        
        if n_dims < 1:
            raise ValueError('Cannot pad scalar data')

        for idx, d in enumerate(data):
            delta_lens = []
            for j in range(n_dims):
                delta_lens.append((0, target_len - d.shape[j]))
            data[idx] = np.pad(d, delta_lens, 'constant', constant_values=pad_val)
        return data
