
from audioop import bias
from typing import Optional
from unicodedata import bidirectional

import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import CrossEntropyLoss
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertConfig, BertModel
from transformers.file_utils import ModelOutput

from ee_data import EE_label2id1, NER_PAD

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

        '''NOTE: This is where to modify for CRF.

        '''

    def _pred_labels(self):
        '''NOTE: This is where to modify for CRF.
        
        You need to finish the code to predict labels.

        You can add input arguments.
        
        '''
        # return pred_labels

    def forward(self, hidden_states, attention_mask, labels=None, no_decode=False, label_pad_token_id=NER_PAD_ID):    
        '''NOTE: This is where to modify for CRF.
        
        You need to finish the code to compute loss and predict labels.


        '''
        # return NEROutputs(loss, pred_labels)


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
        '''NOTE: This is where to modify for Nested NER.

        '''
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

        '''NOTE: This is where to modify for Nested NER.

        Use the above function _group_ner_outputs for combining results.

        '''


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

