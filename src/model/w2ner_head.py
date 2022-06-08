from turtle import forward
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .ner_outputs import NEROutputs
from typing import List, Optional


class ConditionalLayerNorm(nn.Module):
    def __init__(
            self,
            in_dim: int,
            cond_dim: int,
            center: bool = True,
            scale: bool = True,
            epsilon: Optional[float] = None,
            conditional: bool = True):
        super().__init__()
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.epsilon = epsilon or 1e-6
        self.input_dim = in_dim
        self.cond_dim = cond_dim

        self.beta = nn.Parameter(torch.zeros(in_dim)) if center else None
        self.gamma = nn.Parameter(torch.ones(in_dim)) if scale else None

        if self.conditional:
            if center:
                self.beta_dense = nn.Linear(cond_dim, in_dim, bias=False)
            if scale:
                self.gamma_dense = nn.Linear(cond_dim, in_dim, bias=False)
        else:
            self.beta_dense = None
            self.gamma_dense = None

        self.initialize_weights()

    def initialize_weights(self):
        with torch.no_grad():
            if self.beta_dense is not None:
                nn.init.constant_(self.beta_dense.weight, 0)
            if self.gamma_dense is not None:
                nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(
            self,
            inputs: torch.Tensor,
            conditions: Optional[torch.Tensor] = None):
        # inputs: B, L, 1, h_in
        # conditions: B, L, h_in

        if self.conditional:
            for _ in range(len(inputs.shape) - len(conditions.shape)):
                # B, 1, L, h_c
                conditions = conditions.unsqueeze(1)
            
            if self.center:
                # B, 1, L, h_in
                beta = self.beta_dense(conditions) + self.beta
            if self.scale:
                # B, 1, L, h_in
                gamma = self.gamma_dense(conditions) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma
        
        # B, L, 1, h_in
        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1, keepdim=True)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1, keepdim=True)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs  # B, L, L, h_in


class ConvLayer(nn.Module):
    def __init__(
            self,
            input_size: int,
            channels: int,
            dilation: List[int],
            dropout: float):
        super().__init__()
        self.channel_align = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU()
        )

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    groups=channels,
                    dilation=d,
                    padding=d
                ),
                nn.GELU()
            )
            for d in dilation])

    def forward(self, x: torch.Tensor):
        # x: B, L, L, C
        
        # B, C, L, L
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.channel_align(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            outputs.append(x)
        
        # B, C', L, L
        outputs = torch.cat(outputs, dim=1)
        # B, L, L, C'
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs


class BiAffine(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int = 1,
            bias_x: bool = True,
            bias_y = True):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.bias_x = bias_x
        self.bias_y = bias_y

        self.weight = nn.Parameter(
            torch.zeros((output_size, input_size + int(bias_x), input_size + int(bias_y))))
        with torch.no_grad():
            nn.init.xavier_normal_(self.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x, y: B, L, H

        if self.bias_x:
            x = torch.cat([x, torch.ones_like(x[...,:1])], dim=-1)
        if self.bias_y:
            y = torch.cat([y, torch.ones_like(y[...,:1])], dim=-1)

        # B, h_out, L, L
        out = torch.einsum('bxi, oij, byj -> boxy', x, self.weight, y)

        # B, L, L, h_out
        out = out.permute(0, 2, 3, 1)

        return out


class Feedforward(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class CoPredictor(nn.Module):
    def __init__(
            self,
            num_labels: int,
            hid_size: int,
            biaffine_size: int,
            ffwd_input_size: int,
            ffwd_hid_size: int,
            dropout: float = 0):
        super().__init__()
        self.biaffine_proj1 = Feedforward(hid_size, biaffine_size, dropout)
        self.biaffine_proj2 = Feedforward(hid_size, biaffine_size, dropout)
        self.biaffine = BiAffine(biaffine_size, num_labels, bias_x=True, bias_y=True)
        self.ffwd_proj = Feedforward(ffwd_input_size, ffwd_hid_size, dropout)
        self.ffwd = nn.Linear(ffwd_hid_size, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            z: torch.Tensor):
        # x: B, L, hx
        # y: B, L, hy
        # z: B, L, L, hz

        # h,t : B, L, h_biaff
        h = self.dropout(self.biaffine_proj1(x))
        t = self.dropout(self.biaffine_proj2(y))

        # B, L, L, h_out
        out1 = self.biaffine(h, t)

        out2 = self.ffwd(self.dropout(self.ffwd_proj(z)))

        return out1 + out2


class W2NERDecoder(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_labels: int,
            dropout: float,
            lstm_hid_size: int = 512,
            dist_emb_size: int = 20,
            type_emb_size: int = 20,
            emb_dropout: float = 0.5,
            conv_out_channels: int = 160,
            conv_dilation: List[int] = [1, 2, 3],
            conv_dropout: float = 0.5,
            biaffine_size: int = 512,
            ffwd_hid_size: int = 288,
            copredictor_dropout: float = 0.33):
        super().__init__()
        self.lstm_hid_size = lstm_hid_size
        self.conv_hid_size = conv_out_channels

        # conditional layernorm
        self.cln = ConditionalLayerNorm(
            in_dim=lstm_hid_size,
            cond_dim=lstm_hid_size)

        # embeddings
        self.relative_pos_embd = nn.Embedding(20, dist_emb_size)
        self.type_embd = nn.Embedding(3, type_emb_size)

        # lstm layer
        lstm_input_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hid_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True)

        # convolutional layers
        conv_input_size = lstm_hid_size + dist_emb_size + type_emb_size
        self.conv_layer = ConvLayer(
            input_size=conv_input_size,
            channels=conv_out_channels,
            dilation=conv_dilation,
            dropout=conv_dropout)

        # co-predictor
        self.dropout = nn.Dropout(emb_dropout)
        self.copredictor = CoPredictor(
            num_labels=num_labels,
            hid_size=lstm_hid_size,
            biaffine_size=biaffine_size,
            ffwd_input_size=conv_out_channels,
            ffwd_hid_size=ffwd_hid_size,
            dropout=copredictor_dropout)

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(
            self,
            hidden_states: torch.Tensor,
            grid_mask: torch.Tensor,
            rel_pos: torch.Tensor,
            lengths: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            no_decode: bool = False) -> NEROutputs:
        # discard CLS and SEP tokens
        hidden_states = hidden_states[:, 1:-1, :]
        assert hidden_states.size(1) == lengths.max()

        # LSTM encoding
        hidden_states = self.dropout(hidden_states)
        packed_inputs = pack_padded_sequence(
            hidden_states, length=lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (lstm_h, _) = self.lstm(packed_inputs)
        hidden_states = pad_packed_sequence(
            lstm_out, total_length=lengths.max(), batch_first=True)

        # conditional layer norm
        # B, L, H -> B, L, L, H
        cln_out = self.cln(hidden_states.unsqueeze(2), hidden_states)

        # embeddings
        rel_pos_embd = self.relative_pos_embd(rel_pos)
        tril_mask = torch.tril(grid_mask.clone().long())
        reg_inputs = tril_mask + grid_mask.clone().long()
        reg_embd = self.type_embd(reg_inputs)

        # B, L, L, h_in
        conv_inputs = torch.cat([cln_out, rel_pos_embd, reg_embd], dim=-1)
        conv_inputs = torch.masked_fill(conv_inputs, grid_mask.eq(0).unsqueeze(-1), 0)
        conv_outputs = self.conv_layer(conv_inputs)
        conv_outputs = torch.masked_fill(conv_outputs, grid_mask.eq(0).unsqueeze(-1), 0)
        outputs = self.copredictor(hidden_states, hidden_states, conv_outputs)

        loss, logits = None, None
        if labels is not None:
            grid_mask_ = grid_mask.clone()
            print(outputs[grid_mask_].shape)
            loss = self.loss_fct(outputs[grid_mask_], labels[grid_mask_])
            if not no_decode:
                # B, L, L
                logits = outputs.argmax(dim=1)
        else:
            logits = outputs.argmax(dim=1)

        return NEROutputs(loss, logits)
