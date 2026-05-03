"""Diversity regulariser — KL(batch mean || uniform)."""
from __future__ import annotations

import torch
import torch.nn as nn


class DiversityLoss(nn.Module):
    """KL divergence between the batch-average prediction and a uniform prior.

    For binary: KL( [p̄_0, p̄_1] || [0.5, 0.5] )
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.num_classes = int(num_classes)

    def forward(self, soft_probs: torch.Tensor) -> torch.Tensor:
        mean_p = soft_probs.mean(dim=0).clamp(min=1e-8)
        uniform = 1.0 / self.num_classes
        # KL(U || p_mean) = sum_c U * log(U / p_mean_c)
        kl = (uniform * (torch.log(torch.tensor(uniform, device=soft_probs.device))
                         - torch.log(mean_p))).sum()
        return kl
