"""Compute and log frozen source prototypes."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_source_prototypes(
    model: torch.nn.Module,
    source_loader: DataLoader,
    num_classes: int,
    device: str | torch.device,
) -> Dict[int, torch.Tensor]:
    """Compute L2-normalised class prototypes from the FULL source dataset."""
    model.eval()
    feature_dim = getattr(model, "feature_dim", None)
    if feature_dim is None:
        # Fallback: peek at one batch to determine dimension
        for imgs, _ in source_loader:
            feats, _ = model(imgs.to(device), return_features=True)
            feature_dim = feats.size(1)
            break

    sums = torch.zeros(num_classes, feature_dim, device=device)
    counts = torch.zeros(num_classes, device=device)

    for imgs, labels in source_loader:
        imgs = imgs.to(device)
        labels = labels.to(device).long()
        feats, _ = model(imgs, return_features=True)        # already L2-normalised
        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                sums[c] += feats[mask].sum(dim=0)
                counts[c] += mask.sum()

    if torch.any(counts == 0):
        missing = [c for c in range(num_classes) if counts[c] == 0]
        raise RuntimeError(f"No source samples seen for class(es): {missing}")

    protos = sums / counts.unsqueeze(1)
    protos = F.normalize(protos, p=2, dim=1)
    protos = protos.detach()
    protos.requires_grad_(False)
    return {c: protos[c].cpu().clone() for c in range(num_classes)}


def log_prototype_stats(prototypes: Dict[int, torch.Tensor]) -> Dict[str, float]:
    """Return and print per-prototype norms + inter-prototype cosine distance(s)."""
    classes = sorted(prototypes.keys())
    stats: Dict[str, float] = {}
    for c in classes:
        n = float(prototypes[c].norm(p=2).item())
        stats[f"norm_class_{c}"] = n
        print(f"[prototypes] class {c}: L2 norm = {n:.4f}")

    if len(classes) >= 2:
        a = prototypes[classes[0]].view(-1)
        b = prototypes[classes[1]].view(-1)
        cos = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
        stats["inter_cosine"] = cos
        stats["inter_cosine_distance"] = 1.0 - cos
        print(f"[prototypes] class {classes[0]} vs {classes[1]} "
              f"cosine sim = {cos:.4f}  |  distance = {1.0 - cos:.4f}")
    return stats
