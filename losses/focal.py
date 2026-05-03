"""Focal loss with per-class alpha."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: torch.Tensor,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha.float())
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        pt = log_pt.exp().clamp(min=1e-8, max=1.0)
        alpha_t = self.alpha.to(logits.device)[targets]
        loss = -alpha_t * (1.0 - pt) ** self.gamma * log_pt
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
