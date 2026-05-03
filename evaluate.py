"""Standalone DIPN evaluation — metrics + ROC + t-SNE."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from data.dataset import SourceDataset, TargetDataset
from models.backbone import DIPNModel
from utils.checkpoint import load_checkpoint
from utils.metrics import compute_all_metrics


def _load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_device(req: str) -> torch.device:
    if req.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable → CPU")
        return torch.device("cpu")
    return torch.device(req)


@torch.no_grad()
def _collect_features(model, loader, device, has_labels: bool):
    model.eval()
    feats, labels, probs = [], [], []
    for batch in loader:
        if has_labels:
            imgs, y = batch
        else:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            y = None
        imgs = imgs.to(device)
        f, logits = model(imgs, return_features=True)
        p = F.softmax(logits, dim=1)[:, 1]
        feats.append(f.cpu().numpy())
        probs.append(p.cpu().numpy())
        if y is not None:
            labels.append(np.asarray(y))
    feats = np.concatenate(feats, axis=0) if feats else np.zeros((0,))
    probs = np.concatenate(probs, axis=0) if probs else np.zeros((0,))
    labels = np.concatenate(labels, axis=0) if labels else None
    return feats, probs, labels


def main(
    checkpoint: str,
    config: str,
    source_dir: str | None,
    target_dir: str | None,
    device_req: str,
    out_dir: str,
) -> None:
    cfg = _load_config(config)
    if source_dir:
        cfg["data"]["source_dir"] = source_dir
    if target_dir:
        cfg["data"]["target_dir"] = target_dir
    device = _resolve_device(device_req)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = DIPNModel(
        num_classes=int(cfg["model"]["num_classes"]),
        pretrained=False,
    ).to(device)
    load_checkpoint(checkpoint, model, map_location=device)

    image_size = int(cfg["data"]["image_size"])
    bs = int(cfg["data"]["batch_size"])

    src_val = SourceDataset(
        source_dir=cfg["data"]["source_dir"],
        image_size=image_size,
        split="val",
        val_split=float(cfg["data"]["val_split"]),
        seed=int(cfg.get("seed", 42)),
    )
    tgt = TargetDataset(
        target_dir=cfg["data"]["target_dir"],
        image_size=image_size,
        split="val",
    )
    src_loader = DataLoader(src_val, batch_size=bs, shuffle=False)
    tgt_loader = DataLoader(tgt, batch_size=bs, shuffle=False)

    # --- metrics on source val ---
    src_m = compute_all_metrics(model, src_loader, device)
    print("\n============ METRICS (source val) ============")
    for k, v in src_m.items():
        print(f"{k:<12} = {v:.4f}")
    print("==============================================\n")

    # --- ROC curve ---
    src_feats, src_probs, src_y = _collect_features(model, src_loader, device, True)
    if src_y is not None and len(np.unique(src_y)) >= 2:
        fpr, tpr, _ = roc_curve(src_y, src_probs)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {src_m['auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC — Source Validation")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out / "roc_curve.png", dpi=150)
        plt.close(fig)
        print(f"[evaluate] ROC saved → {out/'roc_curve.png'}")

    # --- t-SNE of source + target features ---
    tgt_feats, _, _ = _collect_features(model, tgt_loader, device, False)
    if len(src_feats) == 0 or len(tgt_feats) == 0:
        print("[evaluate] Not enough features for t-SNE.")
        return
    all_feats = np.concatenate([src_feats, tgt_feats], axis=0)
    domain = np.concatenate([
        np.zeros(len(src_feats), dtype=int),
        np.ones(len(tgt_feats), dtype=int),
    ])
    labels = np.concatenate([
        src_y if src_y is not None else -np.ones(len(src_feats), dtype=int),
        -np.ones(len(tgt_feats), dtype=int),
    ])
    n = all_feats.shape[0]
    perp = min(30, max(5, (n - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perp, n_iter=1000,
                init="pca", random_state=0)
    emb = tsne.fit_transform(all_feats)

    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {0: "o", 1: "X"}
    colors = {0: "#1f77b4", 1: "#d62728", -1: "#888888"}
    for d in [0, 1]:
        for y in sorted(set(labels.tolist())):
            m = (domain == d) & (labels == y)
            if not m.any():
                continue
            lab = f"{'src' if d==0 else 'tgt'}/{'healthy' if y==0 else 'cancer' if y==1 else 'unk'}"
            ax.scatter(emb[m, 0], emb[m, 1],
                       marker=markers[d], s=30, alpha=0.7,
                       c=colors.get(int(y), "#888888"), label=lab,
                       edgecolors="k", linewidths=0.3)
    ax.set_title("t-SNE — source (circle) vs target (X)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "tsne.png", dpi=150)
    plt.close(fig)
    print(f"[evaluate] t-SNE saved → {out/'tsne.png'}")


def cli() -> None:
    p = argparse.ArgumentParser(description="DIPN evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--source_dir", type=str, default=None)
    p.add_argument("--target_dir", type=str, default=None)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, default="results")
    args = p.parse_args()
    main(args.checkpoint, args.config, args.source_dir, args.target_dir,
         args.device, args.out_dir)


if __name__ == "__main__":
    cli()
