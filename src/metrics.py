import torch
import numpy as np
from collections import defaultdict

from ee_data import NER_PAD, _LABEL_RANK
from ee_data import EE_label2id, EE_label2id1
from ee_data import EE_id2label1, EE_id2label2, EE_id2label
from ee_data import W2_LABEL2ID, W2_SUC, W2_ID2LABEL, NO_ENT

from typing import List, Union, NamedTuple, Tuple, Set


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
    batch_w2matrices = batch_w2matrices.detach().cpu().numpy()
    batch_lenths = batch_lenths.detach().cpu().numpy()
    for idx, (w2matrix, length) in enumerate(zip(batch_w2matrices, batch_lenths)):
        w2matrix = w2matrix[:length, :length]
        word2nextword = defaultdict(list)  # dict of lists, each list stores keys of next-words
        ht2type = defaultdict()  # maps a head-tail pair (entity) to its type
        head2tail = defaultdict(set)  # dict of sets, each set stores tail-words of a head-word
        
        # build next-words
        for i, j in np.argwhere(w2matrix == W2_LABEL2ID[W2_SUC]):
            i, j = i.item(), j.item()
            if i >= j:
                continue
            word2nextword[i].append(j)

        # build head-tail and tail-head links
        for i, j in np.argwhere(w2matrix != W2_LABEL2ID[W2_SUC]):
            i, j = i.item(), j.item()
            if i > j:
                continue
            if w2matrix[j, i] != W2_LABEL2ID[NER_PAD]:
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


def _determine_entity_type(entity: np.ndarray, id2label: dict) -> str:
    """Determines the type (class) of the given `entity`.

    The exact class of an entity is determined by
    the most frequent class label in the entity.

    If more than one candidate exists, the class is determined by
    the global frequency of the candidates.

    Args:
        entity (np.ndarray): A sequence of labels. A 1D array of `str`s.
        id2label (dict): Entity id to label mapping.

    Returns:
        str: The class of the given entity.
    """
    etypes = np.array(
        list(map(lambda x: id2label[x].split('-')[-1], entity)))

    items, counts = np.unique(etypes, return_counts=True)
    candidates = np.nonzero(counts == counts.max())[0]
    if len(candidates) == 1:
        return items[candidates[0]]
    else:
        max_freq = -1
        max_candidate = ''
        for c in items[candidates]:
            freq = _LABEL_RANK[c]
            if freq > max_freq:
                max_freq = freq
                max_candidate = c
        return max_candidate


def _get_entity_boundary(
        sentence: np.ndarray,
        start: int,
        id2label: dict) -> Tuple[int, int]:
    """Determines the boundary of the entity,
    including `start` but excluding `end`.

    Args:
        sentence (np.ndarray): A 1D array of predicted label-ids.
        start (int): Starting point of the entity.
            Should be the index of a token with label `B-xxx`.
        id2label (dict): LabelId to LabelString mapping.

    Returns:
        (start, end): The start and end index of the entity.
    """
    idx = start
    idx += 1

    eeid = sentence[idx]
    label = id2label[eeid]

    while idx < len(sentence) and label.startswith('I'):
        idx += 1
        eeid = sentence[idx]
        label = id2label[eeid]

    return start, idx


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray


class MetricsForGlobalPtr:
    def __init__(self) -> None:
        self.tot_hits = 0
        self.tot_preds = 0
        self.tot_trues = 0

    def accumulate(self, preds, labels):
        preds = torch.gt(preds, 0).float()
        self.tot_hits += (preds * labels).sum().item()
        self.tot_preds += preds.sum().item()
        self.tot_trues += labels.sum().item()

    def _f1(self):
        if self.tot_preds == 0 and self.tot_trues == 0:
            return 0
        return 2 * self.tot_hits / (self.tot_preds + self.tot_trues)

    def _precision(self):
        if self.tot_preds == 0:
            return 0
        return self.tot_hits / self.tot_preds

    def _recall(self):
        if self.tot_trues == 0:
            return 0
        return self.tot_hits / self.tot_trues

    def summary(self):
        return {
            "f1": self._f1(),
            "precision": self._precision(),
            "recall": self._recall()}

    def __call__(self, eval_pred) -> dict:
        predictions, labels = eval_pred
        predictions = torch.gt(predictions, 0).float()
        f1 = 2 * torch.sum(predictions * labels) / torch.sum(predictions + labels)

        return {'f1': f1}


class MetricsForW2NER:
    def __init__(self) -> None:
        self.tot_hits = 0
        self.tot_preds = 0
        self.tot_trues = 0

    def accumulate(self, preds, labels, batch_lengths):
        preds_decoded = decode_w2matrix(preds, batch_lengths)
        labels_decoded = decode_w2matrix(labels, batch_lengths)

        for pes, les in zip(preds_decoded, labels_decoded):
            pes = set(pes)
            les = set(les)
            for pe in pes:
                if pe in les:
                    self.tot_hits += 1
            self.tot_preds += len(pes)
            self.tot_trues += len(les)

    def _f1(self):
        if self.tot_preds == 0 and self.tot_trues == 0:
            return 0
        return 2 * self.tot_hits / (self.tot_preds + self.tot_trues)

    def _precision(self):
        if self.tot_preds == 0:
            return 0
        return self.tot_hits / self.tot_preds

    def _recall(self):
        if self.tot_trues == 0:
            return 0
        return self.tot_hits / self.tot_trues

    def summary(self):
        return {
            "f1": self._f1(),
            "precision": self._precision(),
            "recall": self._recall()}


class MetricsForBIOTagging:  # training_args  `--label_names labels `
    def _f1(self, hit, pred, true):
        if pred == 0 and true == 0:
            return 0
        return 2 * hit / (pred + true)

    def _precision(self, hit, pred):
        if pred == 0:
            return 0
        return hit / pred
    
    def _recall(self, hit, true):
        if true == 0:
            return 0
        return hit / true

    def __call__(self, eval_pred) -> dict:
        predictions, labels = eval_pred
        B, T = predictions.shape

        # -100 ==> [PAD]
        # [batch, seq_len]
        predictions[predictions == -100] = EE_label2id[NER_PAD]
        labels[labels == -100] = EE_label2id[NER_PAD]

        n_hit = 0
        tot_pred = 0
        tot_label = 0
        f1 = 0

        pred_entities = extract_entities_biotagging(predictions)
        label_entities = extract_entities_biotagging(labels)

        for pes, les in zip(pred_entities, label_entities):
            pes = set(pes)
            les = set(les)
            for pe in pes:
                if pe in les:
                    n_hit += 1
            tot_pred += len(pes)
            tot_label += len(les)

        f1 = 2 * n_hit / (tot_pred + tot_label)

        return {
            "f1": self._f1(),
            "precision": self._precision(),
            "recall": self._recall()}


class MetricsForNestedBIOTagging:
    # training_args  `--label_names labels labels2`
    def __call__(self, eval_pred) -> dict:
        predictions, (labels1, labels2) = eval_pred

        # -100 ==> [PAD]
        # [batch, seq_len, 2]
        predictions[predictions == -100] = EE_label2id[NER_PAD]
        predictions1 = predictions[:, :, 0]
        predictions2 = predictions[:, :, 1]
        # [batch, seq_len]
        labels1[labels1 == -100] = EE_label2id[NER_PAD]
        labels2[labels2 == -100] = EE_label2id[NER_PAD]

        n_hit = 0
        tot_pred = 0
        tot_label = 0
        f1 = 0

        n_hit1, n_pred1, n_label1 = self._compute_hits(
            predictions1, labels1, first_label=True)
        n_hit2, n_pred2, n_label2 = self._compute_hits(
            predictions2, labels2, first_label=False)
        n_hit = n_hit1 + n_hit2
        tot_pred = n_pred1 + n_pred2
        tot_label = n_label1 + n_label2

        f1 = 2 * n_hit / (tot_pred + tot_label)

        return {"f1": f1}

    def _compute_hits(
            self,
            predictions: np.ndarray,
            labels: np.ndarray,
            first_label: bool = True):
        true = 0
        tot_pred = 0
        tot_label = 0

        pred_entities = extract_entities_biotagging(predictions, True, first_label)
        label_entities = extract_entities_biotagging(labels, True, first_label)

        for pes, les in zip(pred_entities, label_entities):
            pes = set(pes)
            les = set(les)
            for pe in pes:
                if pe in les:
                    true += 1
            tot_pred += len(pes)
            tot_label += len(les)

        return true, tot_pred, tot_label


def extract_entities_biotagging(
        batch_labels_or_preds: np.ndarray,
        for_nested_ner: bool = False,
        first_labels: bool = True) -> List[List[tuple]]:
    """
    本评测任务采用严格 Micro-F1作为主评测指标, 即要求预测出的 实体的起始、结束下标，实体类型精准匹配才算预测正确。

    Args:
        batch_labels_or_preds: The labels ids or predicted label ids.
        for_nested_ner:        Whether the input label ids is about Nested NER.
        first_labels:          Which kind of labels for NestNER.
    """
    batch_labels_or_preds[batch_labels_or_preds == -100]\
        = EE_label2id1[NER_PAD]  # [batch, seq_len]

    if not for_nested_ner:
        id2label = EE_id2label
    else:
        id2label = EE_id2label1 if first_labels else EE_id2label2

    batch_entities = []  # List[List[(start_idx, end_idx, type)]]
    for sentence in batch_labels_or_preds:
        idx = 0
        sentence_entities = []
        while idx < len(sentence):
            eeid = sentence[idx]
            if eeid == 1:
                idx += 1
                continue
            if eeid == 0:
                break
            label = EE_id2label[eeid]

            if label.startswith('B'):
                start, end = _get_entity_boundary(sentence, idx, id2label)
                entity = sentence[start:end]
                etype = _determine_entity_type(entity, id2label)
                # end excludes the last token, have to fix this manually
                sentence_entities.append((start, end - 1, etype))
                idx = end
                continue
            else:
                idx += 1
        batch_entities.append(sentence_entities)

    return batch_entities


if __name__ == '__main__':

    # Test for ComputeMetricsForNER
    predictions = np.load('../test_files/predictions.npy')
    labels = np.load('../test_files/labels.npy')

    metrics = MetricsForBIOTagging()(EvalPrediction(predictions, labels))

    if metrics["f1"] is None:
        print("f1-score is None")
    elif abs(metrics['f1'] - 0.606179116) < 1e-5:
        print('You passed the test for ComputeMetricsForNER.')
        print(f'Err: {abs(metrics["f1"] - 0.606179116)}')
    else:
        print('The result of ComputeMetricsForNER is not right.')
        print(
            f'Res: {metrics["f1"]}; Err: {abs(metrics["f1"] - 0.606179116)}')

    # Test for ComputeMetricsForNestedNER
    predictions = np.load('../test_files/predictions_nested.npy')
    labels1 = np.load('../test_files/labels1_nested.npy')
    labels2 = np.load('../test_files/labels2_nested.npy')

    metrics = MetricsForNestedBIOTagging()(
        EvalPrediction(predictions, (labels1, labels2)))

    if metrics['f1'] is None:
        print('Result is None. Skipping.')
    elif abs(metrics['f1'] - 0.60333644) < 1e-5:
        print('You passed the test for ComputeMetricsForNestedNER.')
        print(f'Err: {abs(metrics["f1"] - 0.60333644)}')
    else:
        print('The result of ComputeMetricsForNestedNER is not right.')
        print(
            f'Res: {metrics["f1"]}; Err: {abs(metrics["f1"] - 0.60333644)}')
