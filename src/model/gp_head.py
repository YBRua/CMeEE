import torch
import torch.nn as nn

from .ner_outputs import NEROutputs
from ee_data import EE_label2id1, NER_PAD
from loss_funcs import GlobalPtrLoss


NER_PAD_ID = EE_label2id1[NER_PAD]


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
        logits = torch.masked_fill(logits, ~pad_mask.bool(), float('-inf'))

        # tril mask
        tril_mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = torch.masked_fill(logits, tril_mask.bool(), float('-inf'))
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
            output_dim,
            device) -> torch.Tensor:
        # L, 1
        ks = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1).to(device)

        # L, d // 2
        indices = torch.arange(0, output_dim // 2, dtype=torch.float).to(device)
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
        pe = self.sinusoidal_position_embedding(B, L, output_dim, x.device)

        # B, L, 1, d
        cos_pos = pe[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pe[..., None, 0::2].repeat_interleave(2, dim=-1)

        x1 = x
        x2 = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).reshape(x1.shape)
        return x1 * cos_pos + x2 * sin_pos
