"""Phase 1 — source-only training with focal loss + optional Mixup."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from losses.focal import FocalLoss
from losses.mixup import soft_cross_entropy, source_mixup
from utils.checkpoint import save_checkpoint
from utils.metrics import compute_all_metrics


def _get(cfg: Any, *keys: str, default: Any = None) -> Any:
    cur = cfg
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, default)
        else:
            cur = getattr(cur, k, default)
    return cur


def train_phase1(
    config: Dict[str, Any],
    model: torch.nn.Module,
    source_train_loader: DataLoader,
    source_val_loader: DataLoader,
    device: str | torch.device,
    class_weights: torch.Tensor,
) -> torch.nn.Module:
    """Source-only pretraining with early plateau stop (keeps backbone plastic)."""
    p1 = config["phase1"]
    num_classes = int(config["model"]["num_classes"])

    focal = FocalLoss(alpha=class_weights, gamma=float(p1["focal_gamma"])).to(device)
    optim = torch.optim.Adam(
        model.parameters(), lr=float(p1["lr"]),
        weight_decay=float(p1["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=int(p1["epochs"])
    )

    mixup_cfg = p1.get("mixup", {"enabled": False, "alpha": 0.4, "prob": 0.5})
    mixup_enabled = bool(mixup_cfg.get("enabled", False))
    mixup_alpha = float(mixup_cfg.get("alpha", 0.4))
    mixup_prob = float(mixup_cfg.get("prob", 0.5))

    save_dir = Path(config["checkpoint"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "phase1_best.pt"

    best_auc = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    patience = int(p1["patience"])
    delta = float(p1["plateau_delta"])
    bad_epochs = 0

    print(f"[phase1] starting — epochs={p1['epochs']}  "
          f"mixup={'on' if mixup_enabled else 'off'}")

    for epoch in range(int(p1["epochs"])):
        model.train()
        train_loss_sum, n = 0.0, 0
        mixup_batches = 0
        total_batches = 0
        for imgs, labels in source_train_loader:
            total_batches += 1
            imgs = imgs.to(device)
            labels = labels.to(device).long()
            optim.zero_grad(set_to_none=True)

            use_mixup = (mixup_enabled
                         and torch.rand(1).item() < mixup_prob
                         and imgs.size(0) >= 2)
            if use_mixup:
                mx_imgs, mx_labels = source_mixup(
                    imgs, labels, alpha=mixup_alpha, num_classes=num_classes
                )
                logits = model(mx_imgs)
                loss = soft_cross_entropy(logits, mx_labels)
                mixup_batches += 1
            else:
                logits = model(imgs)
                loss = focal(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            train_loss_sum += float(loss.item()) * imgs.size(0)
            n += imgs.size(0)
        scheduler.step()

        train_loss = train_loss_sum / max(n, 1)
        metrics = compute_all_metrics(model, source_val_loader, device)
        val_auc = metrics["auc"]
        mixup_frac = mixup_batches / max(total_batches, 1)
        print(f"[phase1] epoch {epoch+1:03d}/{p1['epochs']}  "
              f"loss={train_loss:.4f}  val_auc={val_auc:.4f}  "
              f"sens={metrics['sensitivity']:.3f}  "
              f"spec={metrics['specificity']:.3f}  "
              f"mixup_frac={mixup_frac:.2f}", flush=True)

        if not np.isnan(val_auc) and val_auc - best_auc > delta:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(ckpt_path, model, extra={"epoch": epoch, "val_auc": val_auc})
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[phase1] plateau reached (no >{delta} AUC gain for "
                      f"{patience} epochs). Early-stopping at epoch {epoch+1}.")
                break

    model.load_state_dict(best_state)
    print(f"[phase1] best val AUC = {best_auc:.4f}  (checkpoint → {ckpt_path})")
    return model
