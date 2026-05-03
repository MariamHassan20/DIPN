"""Evaluation metrics."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader


def compute_auc(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    y_true = np.asarray(list(y_true))
    y_prob = np.asarray(list(y_prob))
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_sensitivity(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    pos = (y_true == 1)
    if pos.sum() == 0:
        return float("nan")
    return float(((y_pred == 1) & pos).sum() / pos.sum())


def compute_specificity(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    neg = (y_true == 0)
    if neg.sum() == 0:
        return float("nan")
    return float(((y_pred == 0) & neg).sum() / neg.sum())


def compute_f1(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))


# ------------------------------------------------------------------
def _iterate_loader(loader: DataLoader, has_labels: bool = True):
    for batch in loader:
        if has_labels and isinstance(batch, (list, tuple)) and len(batch) == 2:
            yield batch[0], batch[1]
        else:
            # target loader — no labels
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            yield imgs, None


@torch.no_grad()
def compute_all_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str | torch.device,
) -> Dict[str, float]:
    """Evaluate model on a labeled loader. Returns dict of metrics."""
    model.eval()
    all_y, all_p, all_pred = [], [], []
    for imgs, labels in _iterate_loader(loader, has_labels=True):
        if labels is None:
            raise ValueError("compute_all_metrics expects a labeled loader.")
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        all_y.extend(labels.tolist())
        all_p.extend(probs.detach().cpu().tolist())
        all_pred.extend(preds.detach().cpu().tolist())

    y_true = np.asarray(all_y)
    y_pred = np.asarray(all_pred)
    acc = float((y_pred == y_true).mean()) if len(y_true) else float("nan")
    return {
        "auc": compute_auc(all_y, all_p),
        "sensitivity": compute_sensitivity(all_y, all_pred),
        "specificity": compute_specificity(all_y, all_pred),
        "f1": compute_f1(all_y, all_pred),
        "accuracy": acc,
    }


@torch.no_grad()
def compute_prototype_target_alignment(
    model: torch.nn.Module,
    target_loader: DataLoader,
    prototypes: Dict[int, torch.Tensor],
    device: str | torch.device,
) -> float:
    """Mean cosine similarity between target features and their nearest prototype."""
    model.eval()
    classes = sorted(prototypes.keys())
    proto = torch.stack(
        [F.normalize(prototypes[c].float().view(-1), p=2, dim=0)
         for c in classes], dim=0
    ).to(device)

    sims = []
    for batch in target_loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = imgs.to(device)
        feats, _ = model(imgs, return_features=True)
        cos = feats @ proto.t()                # [B, C]
        sims.append(cos.max(dim=1).values)
    if not sims:
        return float("nan")
    return float(torch.cat(sims).mean().item())
