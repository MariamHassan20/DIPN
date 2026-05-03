"""Mixup utilities — source-only, cross-domain, and soft cross-entropy."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def _sample_lambda(alpha: float, device: torch.device) -> float:
    """Sample lambda ~ Beta(alpha, alpha), then lambda = max(l, 1-l) so the
    primary (source) image always dominates.
    """
    if alpha <= 0.0:
        return 1.0
    dist = torch.distributions.Beta(alpha, alpha)
    lam = float(dist.sample().item())
    return max(lam, 1.0 - lam)


def source_mixup(
    images: Tensor,
    labels: Tensor,
    alpha: float = 0.4,
    num_classes: int = 2,
) -> Tuple[Tensor, Tensor]:
    """Source-only Mixup.

    Returns (mixed_images [B,C,H,W], mixed_labels [B,num_classes]).
    The labels are SOFT — use soft_cross_entropy.
    """
    if images.size(0) != labels.size(0):
        raise ValueError("images and labels batch size mismatch")
    device = images.device
    lam = _sample_lambda(alpha, device)
    onehot = F.one_hot(labels.long(), num_classes=num_classes).float()
    perm = torch.randperm(images.size(0), device=device)
    mixed_images = lam * images + (1.0 - lam) * images[perm]
    mixed_labels = lam * onehot + (1.0 - lam) * onehot[perm]
    return mixed_images, mixed_labels


def cross_domain_mixup(
    source_images: Tensor,
    source_labels: Tensor,
    target_images: Tensor,
    target_soft_labels: Tensor,
    alpha: float = 0.4,
    num_classes: int = 2,
) -> Tuple[Tensor, Tensor]:
    """Mix source (hard labels) with target (soft corrected labels).

    Pairs every source image with a randomly-chosen target image (with
    replacement if B_t < B_s). Source always dominates because
    lambda >= 0.5.
    """
    if source_images.size(0) != source_labels.size(0):
        raise ValueError("source images/labels batch size mismatch")
    if target_images.size(0) != target_soft_labels.size(0):
        raise ValueError("target images/soft-labels batch size mismatch")

    device = source_images.device
    Bs = source_images.size(0)
    Bt = target_images.size(0)
    if Bt == 0:
        raise ValueError("Target batch must be non-empty for cross-domain mixup.")

    replace = Bt < Bs
    if replace:
        idx = torch.randint(0, Bt, (Bs,), device=device)
    else:
        idx = torch.randperm(Bt, device=device)[:Bs]

    paired_targets = target_images[idx]
    paired_soft = target_soft_labels[idx]

    lam = _sample_lambda(alpha, device)
    src_onehot = F.one_hot(source_labels.long(), num_classes=num_classes).float()

    mixed_images = lam * source_images + (1.0 - lam) * paired_targets
    mixed_labels = lam * src_onehot + (1.0 - lam) * paired_soft
    return mixed_images, mixed_labels


def soft_cross_entropy(logits: Tensor, soft_labels: Tensor) -> Tensor:
    """Cross-entropy with soft (non-one-hot) targets. Returns scalar mean."""
    if logits.shape != soft_labels.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} != "
            f"soft_labels shape {tuple(soft_labels.shape)}"
        )
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_labels * log_probs).sum(dim=1).mean()
