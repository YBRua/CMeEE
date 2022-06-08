from .common import RAW_LABEL2ID, NO_ENT, W2_LABEL2ID
from typing import List


class InputExample:
    def __init__(self, sentence_id: str, text: str, entities: List[dict] = None):
        self.sentence_id = sentence_id
        self.text = text
        self.entities = entities

    def to_begin_end_label_tuples(self, mode: str = 'gp'):
        """Convert input examples to inputs
        whose labels are (begin, end, type) tuples.

        Args:
            mode (str): should be one of 'gp' and 'w2'
                determines the modeling type
                different modeling types use different label ids

        Returns:
            - Training set returns sentence_id, text, label
                - sentence_id: uid for the sentence
                - text: raw text of the input
                - label: List of Tuple. Each Tuple contains
                    the start and end index, and the label id of the entity
            - Test set returns sentence_id, text
        """
        label2id_mapping = RAW_LABEL2ID if mode == 'gp' else W2_LABEL2ID
        if self.entities is None:
            return self.sentence_id, self.text
        else:
            labels = []
            for e in self.entities:
                start, end, label = e['start_idx'], e['end_idx'], e['type']
                if start <= end:
                    labels.append((start, end, label2id_mapping[label]))
            return self.sentence_id, self.text, labels

    def to_word_pair_task(self):
        pass

    def to_bio_tagged_task(self, for_nested_ner: bool = False):
        """Converts input examples to BIO-Tagging NER input.

        Args:
            for_nested_ner (bool, optional): Whether to support nested NER.

        Returns:
            - Training set returns sentence_id, text, label
                - sentence_id: uid for the sentence
                - text: raw text of the input
                - label: BIO-Tagged labels of the tokens
            - Test set returns sentence_id, text
        """
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
