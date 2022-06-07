import torch
import torch.nn as nn

from torchcrf import CRF

from .ner_outputs import NEROutputs
from ..ee_data import EE_label2id1, NER_PAD


NER_PAD_ID = EE_label2id1[NER_PAD]


class CRFClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def _pred_labels(self, hidden_states, mask, max_len, label_pad_token_id=NER_PAD_ID):
        decoded = self.crf.decode(hidden_states, mask=mask)
        pred_labels = [
            F.pad(torch.tensor(label),
            (0, max_len - len(label)),
            mode='constant',
            value=label_pad_token_id) for label in decoded]
        return torch.stack(pred_labels).long()

    def forward(
            self,
            hidden_states, attention_mask, labels=None,
            no_decode=False, label_pad_token_id=NER_PAD_ID):
        # hidden_states: B, L, H

        # attention_mask: B, L
        # CRF accepts only ByteTensors
        attention_mask = attention_mask == 1
        bsz, max_len = attention_mask.shape

        # emission: B, L, ntags
        emission = self.dropout(self.linear(hidden_states))
        loss, pred_labels = None, None

        if labels is None:
            pred_labels = self._pred_labels(
                emission, attention_mask, max_len, label_pad_token_id)
        else:
            # use mean reduction, in line with LinearClassifier
            logits = self.crf(
                emission, labels, mask=attention_mask, reduction='mean')

            loss = - logits
            if not no_decode:
                pred_labels = self._pred_labels(
                    emission, attention_mask, max_len, label_pad_token_id)

        return NEROutputs(loss, pred_labels)
