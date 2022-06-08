import os
import json
import torch
import numpy as np
from collections import defaultdict

from typing import List, Set, Tuple

from ee_data import RAW_ID2LABEL, NO_ENT
from ee_data import W2_LABEL2ID, W2_SUC
from ee_data.common import W2_ID2LABEL
from metrics import extract_entities_biotagging


def gen_result_bio_tagging(train_args, logger, predictions, test_dataset, for_nested_ner=False):
    assert len(predictions) == len(test_dataset.examples), \
        f"Length mismatch: predictions({len(predictions)}), test examples({len(test_dataset.examples)})"

    if not for_nested_ner:
        pred_entities1 = extract_entities_biotagging(predictions[:, 1:], for_nested_ner=False)
        pred_entities2 = [[]] * len(pred_entities1)
    else:
        pred_entities1 = extract_entities_biotagging(predictions[:, 1:, 0], for_nested_ner=True, first_labels=True)
        pred_entities2 = extract_entities_biotagging(predictions[:, 1:, 1], for_nested_ner=True, first_labels=False)

    final_answer = []

    for p1, p2, example in zip(pred_entities1, pred_entities2, test_dataset.examples):
        text = example.text
        entities = []
        for start_idx, end_idx, entity_type in p1 + p2:
            entities.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "type": entity_type,
                "entity": text[start_idx: end_idx + 1],
            })
        final_answer.append({"text": text, "entities": entities})

    with open(os.path.join(train_args.output_dir, "CMeEE_test.json"), "w", encoding="utf8") as f:
        json.dump(final_answer, f, indent=2, ensure_ascii=False)
        logger.info(f"`CMeEE_test.json` saved")


def gen_result_global_ptr(train_args, logger, predictions, test_dataset, for_nested_ner=False):
    # predictions: B, n_cls, L, L

    assert len(predictions) == len(test_dataset.examples), \
        f"Length mismatch: predictions({len(predictions)}), test examples({len(test_dataset.examples)})"

    final_answer = []

    # pred: n_cls, L, L
    # counter = 0
    for pred, example in zip(predictions, test_dataset.examples):
        # counter += 1
        pred = pred.detach().cpu().numpy()
        text = example.text
        # mask [CLS] and [SEP] token
        pred[:, [0, -1]] -= np.inf
        pred[:, :, [0, -1]] -= np.inf
        entities = []
        for lid, start, end in zip(*np.where(pred > 0)):
        # for start, end, lid in pred:
            entities.append({
                "start_idx": start.item() - 1,  # compensate for [CLS]
                "end_idx": end.item() - 1,
                "type":RAW_ID2LABEL[lid],
                "entity": text[start - 1: end]})
        final_answer.append({
            "text": text,
            "entities": entities
        })
        # assert len(final_answer) == counter, (len(final_answer), counter)

        # if counter % 100 == 0:
        #     print(counter, len(predictions))

    with open(os.path.join(train_args.output_dir, "CMeEE_test.json"), "w", encoding="utf8") as f:
        json.dump(final_answer, f, indent=2, ensure_ascii=False)
        logger.info(f"`CMeEE_test.json` saved")


def gen_result_w2ner(
        train_args,
        logger,
        predictions: List[List[Tuple[int, int, str]]],
        test_dataset,
        for_nested_ner=False):
    assert len(predictions) == len(test_dataset.examples), \
            f"Length mismatch: predictions({len(predictions)}), test examples({len(test_dataset.examples)})"

    final_answer = []
    for pred, example in zip(predictions, test_dataset.examples):
        text = example.text
        entities = []
        for start_idx, end_idx, entity_type in pred:
            entities.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "type": entity_type,
                "entity": text[start_idx: end_idx + 1],
            })
        final_answer.append({
            "text": text,
            "entities": entities
        })

    with open(os.path.join(train_args.output_dir, "CMeEE_test.json"), "w", encoding="utf8") as f:
        json.dump(final_answer, f, indent=2, ensure_ascii=False)
        logger.info(f"`CMeEE_test.json` saved")


def decode_w2matrix(batch_w2matrices: torch.Tensor, batch_lenths: torch.Tensor):
    """Decodes a batch of word-pair matrices.
    Returns a List of List of Tuples
    Each List in List contains Tuples of (start, end, entity_type)
    which indicate entities in current example in current batch

    Args:
        batch_w2matrices (torch.Tensor): B, L, L Predicted word-pair matrix
        batch_lenths (torch.Tensor): B, Lengths of each example in batch

    Returns:
        List[List[Tuple]]: Decoded predictions
    """
    decoded_entities = []
    for idx, (w2matrix, length) in enumerate(zip(batch_w2matrices, batch_lenths)):
        word2nextword = defaultdict(list)  # dict of lists, each list stores keys of next-words
        ht2type = defaultdict()
        head2tail = defaultdict(set)
        
        # build next-words
        for i in range(length):
            for j in range(i + 1, length):
                if w2matrix[i, j] == W2_LABEL2ID[W2_SUC]:
                    word2nextword[i].append(j)
        # build head-tail and tail-head links
        for i in range(length):
            for j in range(i, length):
                if (
                    w2matrix[j, i] != W2_LABEL2ID[NO_ENT]
                    and w2matrix[j, i] != W2_LABEL2ID[W2_SUC]
                ):
                    head2tail[i].add(j)
                    ht2type[(i, j)] = w2matrix[j, i].item()

        # run dfs to find all entities
        predicts = []
        def _lattice_dfs(root: int, tails: Set[int], entity: List[int] = []):
            entity.append(root)
            if root in tails:
                predicts.append(entity.copy())
            if root in word2nextword:
                for next_word in word2nextword[root]:
                    _lattice_dfs(next_word, tails, entity)
            entity.pop()

        for head in word2nextword:
            _lattice_dfs(head, head2tail[head])

        # convert to 3-tuple
        entity_tuples = []
        for pred in predicts:
            start, end = pred[0], pred[-1]
            entity_type = W2_ID2LABEL[ht2type[(start, end)]]
            entity_tuples.append((start, end, entity_type))
        
        decoded_entities.append(entity_tuples)
    return decoded_entities
