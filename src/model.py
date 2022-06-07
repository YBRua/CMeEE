import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertConfig, BertModel
from transformers.file_utils import ModelOutput
from dataclasses import dataclass

from ee_data import EE_label2id1, NER_PAD
from loss_funcs import GlobalPtrLoss

from typing import Optional


NER_PAD_ID = EE_label2id1[NER_PAD]


@dataclass
class NEROutputs(ModelOutput):
    """
    NOTE: `logits` here is the CRF decoding result.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.LongTensor] = None


class LinearClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout: float):
        super().__init__()
        self.num_labels = num_labels
        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
        self.loss_fct = CrossEntropyLoss()

    def _pred_labels(self, _logits):
        return torch.argmax(_logits, dim=-1)

    def forward(self, hidden_states, labels=None, no_decode=False):
        _logits = self.layers(hidden_states)
        loss, pred_labels = None, None

        if labels is None:
            pred_labels = self._pred_labels(_logits)    
        else:
            loss = self.loss_fct(_logits.view(-1, self.num_labels), labels.view(-1))
            if not no_decode:
                pred_labels = self._pred_labels(_logits)

        return NEROutputs(loss, pred_labels)


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


def _group_ner_outputs(output1: NEROutputs, output2: NEROutputs):
    """ logits: [batch_size, seq_len] ==> [batch_size, seq_len, 2] """
    grouped_loss, grouped_logits = None, None

    if not (output1.loss is None or output2.loss is None):
        grouped_loss = (output1.loss + output2.loss) / 2

    if not (output1.logits is None or output2.logits is None):
        grouped_logits = torch.stack([output1.logits, output2.logits], dim=-1)

    return NEROutputs(grouped_loss, grouped_logits)


class BertForLinearHeadNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)

        self.classifier = LinearClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        output = self.classifier.forward(sequence_output, labels, no_decode=no_decode)
        return output


class BertForLinearHeadNestedNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int, num_labels2: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        
        # Two linear heads for NestedNER
        self.head1 = LinearClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        self.head2 = LinearClassifier(config.hidden_size, num_labels2, config.hidden_dropout_prob)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        output1 = self.head1.forward(sequence_output, labels, no_decode=no_decode)
        output2 = self.head2.forward(sequence_output, labels2, no_decode=no_decode)

        return _group_ner_outputs(output1, output2)


class BertForCRFHeadNestedNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int, num_labels2: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        
        # Two linear heads for NestedNER
        self.head1 = CRFClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        self.head2 = CRFClassifier(config.hidden_size, num_labels2, config.hidden_dropout_prob)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        output1 = self.head1.forward(sequence_output, attention_mask, labels, no_decode=no_decode)
        output2 = self.head2.forward(sequence_output, attention_mask, labels2, no_decode=no_decode)

        return _group_ner_outputs(output1, output2)


class BertForCRFHeadNER(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        self.classifier = CRFClassifier(config.hidden_size, num_labels1, config.hidden_dropout_prob)
        
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        output = self.classifier.forward(sequence_output, attention_mask, labels, no_decode=no_decode)
        
        return output


class BertForGlobalPointer(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, num_labels1: int, proj_dim: int = 64):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        self.global_ptr = GlobalPtrHead(config.hidden_size, num_labels1, proj_dim)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            labels2=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            no_decode=False,
    ):
        # B, L, h
        sequence_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]
        
        output = self.global_ptr.forward(sequence_output, attention_mask, labels, no_decode=no_decode)

        return output


class GlobalPtrHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels:int , proj_dim: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj_dim = proj_dim
        self.num_labels = num_labels

        self.global_ptr_projector = nn.Linear(self.hidden_size, proj_dim * num_labels * 2)
        self.pos_embd = RoPE()
        self.loss_fct = GlobalPtrLoss()
            

    def forward(self, hidden_states, attention_mask, labels=None, no_decode=False):
        B, L = hidden_states.shape[:2]

        # B, L, h -> B, L, h * num_labels * 2
        projected = self.global_ptr_projector(hidden_states)

        # B, L, num_labels, 2 * proj_dim
        projected = projected.reshape(B, L, -1, 2 * self.proj_dim)

        # qs, ks: B, L, num_labels, proj_dim
        qs, ks = projected[..., :self.proj_dim], projected[..., self.proj_dim:]

        qs = self.pos_embd(qs, self.proj_dim)
        ks = self.pos_embd(ks, self.proj_dim)

        logits = torch.einsum('bmhd, bnhd -> bhmn', qs, ks)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(B, self.num_labels, L, L)
        # logits = logits*pad_mask - (1-pad_mask)*1e12
        logits = torch.masked_fill(~pad_mask.bool(), logits, float('-inf'))

        # tril mask
        tril_mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = torch.masked_fill(tril_mask.bool(), logits, float('-inf'))
        logits = logits / self.proj_dim ** 0.5

        loss = None

        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return NEROutputs(loss, logits)


class RoPE(nn.Module):
    # <https://github.com/xhw205/GlobalPointer_torch/blob/main/GlobalPointer.py>
    def __init__(self):
        super().__init__()

    def sinusoidal_position_embedding(
            self,
            batch_size,
            seq_len,
            output_dim) -> torch.Tensor:
        # L, 1
        ks = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        # L, d // 2
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        pe = ks * indices

        # L, d // 2, 2
        pe = torch.stack([torch.sin(pe), torch.cos(pe)], dim=-1)

        # B, L, d // 2, 2
        pe = pe.repeat((batch_size, *([1]*len(pe.shape))))

        # B, L, d
        pe = torch.reshape(pe, (batch_size, seq_len, output_dim))
        
        return pe

    def forward(self, x: torch.Tensor, output_dim: int):
        B, L = x.shape[:2]
        pe = self.sinusoidal_position_embedding(B, L, output_dim)

        # B, L, 1, d
        cos_pos = pe[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pe[..., None, 0::2].repeat_interleave(2, dim=-1)

        x1 = x
        x2 = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).reshape(x1.shape)
        return x1 * cos_pos + x2 * sin_pos
