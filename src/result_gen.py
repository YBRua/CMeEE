import os
import json
import numpy as np

from ee_data import RAW_ID2LABEL
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
        assert len(final_answer) == counter, (len(final_answer), counter)

        # if counter % 100 == 0:
        #     print(counter, len(predictions))

    with open(os.path.join(train_args.output_dir, "CMeEE_test.json"), "w", encoding="utf8") as f:
        json.dump(final_answer, f, indent=2, ensure_ascii=False)
        logger.info(f"`CMeEE_test.json` saved")