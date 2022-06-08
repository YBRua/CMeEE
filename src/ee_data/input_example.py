from .common import RAW_LABEL2ID, NO_ENT
from typing import List


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
