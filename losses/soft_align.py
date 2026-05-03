"""Soft prototype alignment loss (core DIPN innovation)."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftPrototypeAlignmentLoss(nn.Module):
    """Pull each target feature toward the expected source prototype.

    L = (1/N) * sum_i sum_c w_i,c * (1 - cos(f_i, mu_c))

    where w_i,c = corrected soft probability (or uncorrected, if no correction
    is passed). Prototypes are frozen L2-normalised source class means.
    """

    def __init__(self, prototypes: Dict[int, torch.Tensor]) -> None:
        super().__init__()
        if not prototypes:
            raise ValueError("prototypes dict is empty")
        classes = sorted(prototypes.keys())
        proto = torch.stack([F.normalize(prototypes[c].float().view(-1), p=2, dim=0)
                             for c in classes], dim=0)
        self.register_buffer("prototypes", proto)     # [C, D]
        self.num_classes = int(proto.size(0))

    def forward(
        self,
        features: torch.Tensor,                    # [B, D] L2-normalised
        soft_probs: torch.Tensor,                  # [B, C]
        correction_weights: Optional[torch.Tensor] = None,  # [B, C]
    ) -> torch.Tensor:
        weights = correction_weights if correction_weights is not None else soft_probs
        protos = self.prototypes.to(features.device)
        features = F.normalize(features, p=2, dim=1)
        cos_sim = features @ protos.t()            # [B, C]
        dist = 1.0 - cos_sim
        return (weights * dist).sum(dim=1).mean()
