"""Phase 2 — Prototypical domain adaptation.

Implements the full DIPN Phase 2 objective (Eq. 7 in the paper):

    L_total = L_src + λ_a * L_align + γ_d * L_div

No cross-domain mixup is used. The optional Saerens-style label-shift
corrector is gated by a prior-gap threshold τ_π (Section 3.3.4).
"""
from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from losses.diversity import DiversityLoss
from losses.focal import FocalLoss
from losses.mixup import soft_cross_entropy, source_mixup
from losses.soft_align import SoftPrototypeAlignmentLoss
from utils.checkpoint import save_checkpoint
from utils.distribution import DistributionCorrector
from utils.metrics import compute_all_metrics, compute_prototype_target_alignment


def _compute_source_prior(source_loader: DataLoader, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes)
    for _, labels in source_loader:
        for c in range(num_classes):
            counts[c] += (labels == c).sum()
    total = counts.sum().clamp(min=1.0)
    return counts / total


def _inter_proto_distance(prototypes: Dict[int, torch.Tensor]) -> float:
    keys = sorted(prototypes.keys())
    if len(keys) < 2:
        return float("nan")
    a = prototypes[keys[0]].view(-1)
    b = prototypes[keys[1]].view(-1)
    return float(1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def train_phase2(
    config: Dict[str, Any],
    model: torch.nn.Module,
    prototypes: Dict[int, torch.Tensor],
    source_train_loader: DataLoader,
    source_val_loader: DataLoader,
    target_loader: DataLoader,
    device: str | torch.device,
    class_weights: torch.Tensor,
    target_eval_loader: DataLoader | None = None,
) -> torch.nn.Module:
    """Full DIPN Phase 2 training loop.

    Loss: L_total = L_src + λ_a * L_align + γ_d * L_div
    """
    p2 = config["phase2"]
    num_classes = int(config["model"]["num_classes"])

    # Ablation / source-only shortcut: skip Phase 2 entirely.
    if int(p2["epochs"]) <= 0:
        print("[phase2] epochs<=0 — skipping Phase 2 (source-only ablation).",
              flush=True)
        return model

    focal     = FocalLoss(alpha=class_weights, gamma=float(p2["focal_gamma"])).to(device)
    align     = SoftPrototypeAlignmentLoss(prototypes).to(device)
    diversity = DiversityLoss(num_classes=num_classes).to(device)

    optim = torch.optim.Adam(
        model.parameters(), lr=float(p2["lr"]),
        weight_decay=float(p2["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=int(p2["epochs"])
    )

    # ---- Loss weights (λ_a, γ_d as in Eq. 7) ----
    lambda_align = float(p2["lambda_align"])   # λ_a  default 1.0
    gamma_div    = float(p2["gamma_diversity"]) # γ_d  default 1.0

    # ---- Optional source Mixup in Phase 2 ----
    mixup_cfg  = p2.get("mixup", {})
    src_mix_on = bool(mixup_cfg.get("source_enabled", False))
    mix_alpha  = float(mixup_cfg.get("alpha", 0.4))
    mix_prob   = float(mixup_cfg.get("prob", 0.5))

    # ---- Saerens-style label-shift corrector (Section 3.3.4) ----
    corr_cfg      = p2.get("correction", {})
    corr_on       = bool(corr_cfg.get("enabled", False))
    warmup_epochs = int(corr_cfg.get("warmup_epochs", 5)) if corr_on else 0
    src_prior     = _compute_source_prior(source_train_loader, num_classes).to(device)
    corrector     = DistributionCorrector(
        source_prior  = src_prior,
        ema_momentum  = float(corr_cfg.get("ema_momentum", 0.99)),
    )
    print(f"[phase2] λ_align={lambda_align}  γ_div={gamma_div}  "
          f"correction={'ON' if corr_on else 'OFF'}  "
          f"warmup_epochs={warmup_epochs}", flush=True)

    save_dir         = Path(config["checkpoint"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path        = save_dir / "phase2_best.pt"
    ckpt_path_target = save_dir / "phase2_best_target.pt"

    best_auc       = -np.inf
    best_tgt_auc   = -np.inf
    best_state     = copy.deepcopy(model.state_dict())
    best_tgt_state = copy.deepcopy(model.state_dict())

    static_inter = _inter_proto_distance(prototypes)
    print(f"[phase2] inter-prototype distance (constant): {static_inter:.4f}")

    # ------------------------------------------------------------------
    for epoch in range(int(p2["epochs"])):
        model.train()
        in_warmup = epoch < warmup_epochs

        # ---- warmup: source focal loss only + accumulate target prior ----
        if in_warmup:
            if corr_on:
                with torch.no_grad():
                    for batch in target_loader:
                        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                        imgs = imgs.to(device)
                        model.eval()
                        logits = model(imgs)
                        probs  = F.softmax(logits, dim=1)
                        corrector.update(probs)

            model.train()
            sum_focal, n = 0.0, 0
            for imgs, labels in source_train_loader:
                imgs, labels = imgs.to(device), labels.to(device).long()
                optim.zero_grad(set_to_none=True)
                logits = model(imgs)
                loss   = focal(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
                sum_focal += float(loss.item()) * imgs.size(0)
                n += imgs.size(0)
            scheduler.step()

            val     = compute_all_metrics(model, source_val_loader, device)
            tgt_msg = ""
            if target_eval_loader is not None:
                tgt = compute_all_metrics(model, target_eval_loader, device)
                tgt_msg = (f"  tgt_auc={tgt['auc']:.4f}"
                           f"  tgt_acc={tgt['accuracy']:.4f}"
                           f"  tgt_f1={tgt['f1']:.4f}")
            prior = corrector.get_target_prior().tolist()
            print(f"[phase2][warmup] epoch {epoch+1:03d}/{p2['epochs']}"
                  f"  L_focal={sum_focal/max(n,1):.4f}"
                  f"  val_auc={val['auc']:.4f}{tgt_msg}"
                  f"  target_prior={['%.3f'%p for p in prior]}", flush=True)

            if epoch + 1 == warmup_epochs:
                corrector.mark_ready()
                print("[phase2] warmup complete — corrector is now ready.")
            continue

        # ---- full training ----
        sum_focal = sum_align = sum_div = sum_tot = 0.0
        n = 0
        target_iter = itertools.cycle(iter(target_loader))

        for imgs, labels in source_train_loader:
            imgs   = imgs.to(device)
            labels = labels.to(device).long()
            optim.zero_grad(set_to_none=True)

            # ----- source branch (with optional source Mixup) -----
            use_src_mix = (src_mix_on
                           and torch.rand(1).item() < mix_prob
                           and imgs.size(0) >= 2)
            if use_src_mix:
                mx_imgs, mx_labels = source_mixup(
                    imgs, labels, alpha=mix_alpha, num_classes=num_classes
                )
                s_logits = model(mx_imgs)
                L_focal  = soft_cross_entropy(s_logits, mx_labels)
            else:
                s_logits = model(imgs)
                L_focal  = focal(s_logits, labels)

            # ----- target branch -----
            t_batch  = next(target_iter)
            t_imgs   = t_batch[0] if isinstance(t_batch, (list, tuple)) else t_batch
            t_imgs   = t_imgs.to(device)
            t_feats, t_logits = model(t_imgs, return_features=True)
            t_probs  = F.softmax(t_logits, dim=1)

            # Saerens correction (Eq. 6) — applied only when corrector is ready
            if corr_on:
                corrector.update(t_probs.detach())
                corrected_probs = corrector.correct(t_probs)
            else:
                corrected_probs = t_probs   # identity when correction disabled

            # ----- losses (Eq. 7: L_total = L_src + λ_a*L_align + γ_d*L_div) -----
            L_align = align(t_feats, t_probs, correction_weights=corrected_probs)
            L_div   = diversity(t_probs)
            L_total = L_focal + lambda_align * L_align + gamma_div * L_div

            L_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            bs         = imgs.size(0)
            sum_focal += float(L_focal.item()) * bs
            sum_align += float(L_align.item()) * bs
            sum_div   += float(L_div.item())   * bs
            sum_tot   += float(L_total.item()) * bs
            n         += bs

        scheduler.step()

        val            = compute_all_metrics(model, source_val_loader, device)
        target_align   = compute_prototype_target_alignment(
            model, target_loader, prototypes, device
        )
        tgt_msg    = ""
        tgt_metrics = None
        if target_eval_loader is not None:
            tgt_metrics = compute_all_metrics(model, target_eval_loader, device)
            tgt_msg = (f"  tgt_auc={tgt_metrics['auc']:.4f}"
                       f"  tgt_acc={tgt_metrics['accuracy']:.4f}"
                       f"  tgt_sens={tgt_metrics['sensitivity']:.3f}"
                       f"  tgt_spec={tgt_metrics['specificity']:.3f}"
                       f"  tgt_f1={tgt_metrics['f1']:.4f}")
        prior = corrector.get_target_prior().tolist()
        print(f"[phase2] epoch {epoch+1:03d}/{p2['epochs']}"
              f"  L_focal={sum_focal/max(n,1):.4f}"
              f"  L_align={sum_align/max(n,1):.4f}"
              f"  L_div={sum_div/max(n,1):.4f}"
              f"  L_tot={sum_tot/max(n,1):.4f}"
              f"  src_val_auc={val['auc']:.4f}"
              f"  sens={val['sensitivity']:.3f}"
              f"  spec={val['specificity']:.3f}{tgt_msg}"
              f"  target_align={target_align:.4f}"
              f"  inter_proto_dist={static_inter:.4f}"
              f"  target_prior={['%.3f'%p for p in prior]}", flush=True)

        # --- save best by source val AUC ---
        if not np.isnan(val["auc"]) and val["auc"] > best_auc:
            best_auc   = val["auc"]
            best_state = copy.deepcopy(model.state_dict())
            extra = {"epoch": epoch, "val_auc": val["auc"]}
            if tgt_metrics is not None:
                extra["target_metrics_at_best"] = tgt_metrics
            save_checkpoint(ckpt_path, model, extra=extra)

        # --- save best by TARGET AUC ---
        if (tgt_metrics is not None
                and not np.isnan(tgt_metrics["auc"])
                and tgt_metrics["auc"] > best_tgt_auc):
            best_tgt_auc   = tgt_metrics["auc"]
            best_tgt_state = copy.deepcopy(model.state_dict())
            save_checkpoint(ckpt_path_target, model,
                            extra={"epoch": epoch,
                                   "target_auc": tgt_metrics["auc"],
                                   "target_metrics": tgt_metrics,
                                   "src_val_auc_at_epoch": val["auc"]})
            print(f"[phase2]  *** new best TARGET AUC = {best_tgt_auc:.4f} "
                  f"(epoch {epoch+1}) → {ckpt_path_target}", flush=True)

    model.load_state_dict(best_state)
    print(f"[phase2] best source val AUC = {best_auc:.4f}  "
          f"(checkpoint → {ckpt_path})")
    print(f"[phase2] best target     AUC = {best_tgt_auc:.4f}  "
          f"(checkpoint → {ckpt_path_target})")
    return model
