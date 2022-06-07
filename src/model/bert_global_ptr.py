from transformers import BertPreTrainedModel, BertConfig, BertModel

from .gp_head import GlobalPtrHead


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
