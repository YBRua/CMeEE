import torch
import torch.nn as nn


class GlobalPtrLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            y_pred: torch.Tensor,
            y_true: torch.Tensor):
        """loss_fn is used for training.
        Args:
            y_pred: Model predcition of shape (B, n_labels, L, L)
            y_true: Ground truth labels of shape (B, n_labels, L, L)
        """
        B, n_labels = y_true.shape[:2]
        y_true = y_true.reshape(B * n_labels, -1)
        y_pred = y_pred.reshape(B * n_labels, -1)
        return self._multilabel_ce(y_true, y_pred)

    def _multilabel_ce(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        # <https://kexue.fm/archives/7359>
        # -1 -> pos classes, 1 -> neg classes
        y_pred = (1 - 2 * y_true) * y_pred

        # mask the pred outputs of pos classes
        y_pred_neg = y_pred - y_true * 1e12

        # mask the pred outputs of neg classes
        y_pred_pos = y_pred - (1 - y_true) * 1e12

        # B * n_labels, 1
        zeros = torch.zeros_like(y_pred[..., :1])

        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)

        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return (neg_loss + pos_loss).mean()
