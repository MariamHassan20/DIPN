"""DIPN — main training entry point.

Usage:
    python train.py \\
        --config configs/dipn.yaml \\
        --source_dir /path/to/vindr_source \\
        --target_dir /path/to/target_unlabeled \\
        --target_eval_dir /path/to/target_eval \\
        --seed 0 \\
        --save_subdir DIPN_VinDr_to_INbreast/run_0 \\
        --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from data.dataset import SourceDataset, TargetDataset, get_class_weights
from models.backbone import DIPNModel
from training.phase1 import train_phase1
from training.phase2 import train_phase2
from training.prototypes import compute_source_prototypes, log_prototype_stats
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import compute_all_metrics


def _load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(req: str) -> torch.device:
    if req.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] CUDA requested but unavailable → falling back to CPU")
        return torch.device("cpu")
    return torch.device(req)


def build_loaders(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]
    seed = int(cfg.get("seed", 42))

    src_train = SourceDataset(
        source_dir=data_cfg["source_dir"],
        image_size=int(data_cfg["image_size"]),
        split="train",
        val_split=float(data_cfg["val_split"]),
        seed=seed,
    )
    src_val = SourceDataset(
        source_dir=data_cfg["source_dir"],
        image_size=int(data_cfg["image_size"]),
        split="val",
        val_split=float(data_cfg["val_split"]),
        seed=seed,
    )
    tgt = TargetDataset(
        target_dir=data_cfg["target_dir"],
        image_size=int(data_cfg["image_size"]),
        split="train",
    )

    bs = int(data_cfg["batch_size"])
    nw = int(data_cfg["num_workers"])
    src_train_loader = DataLoader(src_train, batch_size=bs, shuffle=True,
                                  num_workers=nw, drop_last=False)
    src_val_loader   = DataLoader(src_val,   batch_size=bs, shuffle=False,
                                  num_workers=nw)
    tgt_loader       = DataLoader(tgt,       batch_size=bs, shuffle=True,
                                  num_workers=nw, drop_last=False)
    return src_train, src_val, tgt, src_train_loader, src_val_loader, tgt_loader


@torch.no_grad()
def _collect_eval_outputs(model, loader, device):
    model.eval()
    all_y, all_p, all_pred = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(dim=1)
        all_y.extend(labels.tolist())
        all_p.extend(probs.detach().cpu().tolist())
        all_pred.extend(preds.detach().cpu().tolist())

    y_true = np.asarray(all_y,    dtype=np.int64)
    y_prob = np.asarray(all_p,    dtype=np.float32)
    y_pred = np.asarray(all_pred, dtype=np.int64)
    out = {"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred}
    if len(np.unique(y_true)) >= 2:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        out["fpr"]        = np.asarray(fpr,        dtype=np.float32)
        out["tpr"]        = np.asarray(tpr,        dtype=np.float32)
        out["thresholds"] = np.asarray(thresholds, dtype=np.float32)
    return out


def main(
    config_path: str,
    source_dir: str | None,
    target_dir: str | None,
    device_req: str,
    target_eval_dir: str | None = None,
    seed: int | None = None,
    save_subdir: str | None = None,
) -> None:
    cfg = _load_config(config_path)
    if source_dir:
        cfg["data"]["source_dir"] = source_dir
    if target_dir:
        cfg["data"]["target_dir"] = target_dir
    if target_eval_dir:
        cfg["data"]["target_eval_dir"] = target_eval_dir
    if seed is not None:
        cfg["seed"] = int(seed)
    if save_subdir:
        cfg["checkpoint"]["save_dir"] = str(
            Path(cfg["checkpoint"]["save_dir"]) / save_subdir
        )

    _set_seed(int(cfg.get("seed", 42)))
    device = _resolve_device(device_req)
    print(f"[main] device = {device}  seed = {cfg.get('seed', 42)}  "
          f"save_dir = {cfg['checkpoint']['save_dir']}", flush=True)

    src_train, src_val, tgt, src_train_loader, src_val_loader, tgt_loader = \
        build_loaders(cfg)
    print(f"[main] source train={len(src_train)}  val={len(src_val)}  "
          f"target={len(tgt)}", flush=True)

    # Optional labeled target evaluation set
    target_eval_loader = None
    target_eval_dir_cfg = cfg["data"].get("target_eval_dir")
    if target_eval_dir_cfg:
        tgt_eval = SourceDataset(
            source_dir=target_eval_dir_cfg,
            image_size=int(cfg["data"]["image_size"]),
            split="val",
            val_split=1.0,
            seed=int(cfg.get("seed", 42)),
        )
        target_eval_loader = DataLoader(
            tgt_eval,
            batch_size=int(cfg["data"]["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["data"]["num_workers"]),
        )
        print(f"[main] labeled target eval set = {len(tgt_eval)} images "
              f"({target_eval_dir_cfg})", flush=True)

    class_weights = get_class_weights(src_train).to(device)
    print(f"[main] class weights = {class_weights.tolist()}")

    backbone = os.environ.get("DIPN_BACKBONE_OVERRIDE") \
               or cfg["model"].get("backbone", "efficientnet_b0")
    print(f"[main] backbone = {backbone}", flush=True)

    model = DIPNModel(
        num_classes=int(cfg["model"]["num_classes"]),
        pretrained=bool(cfg["model"]["pretrained"]),
        backbone=backbone,
    ).to(device)
    print(f"[main] feature_dim = {model.feature_dim}", flush=True)

    # ------------------------------------------------------------------ Phase 1
    model = train_phase1(cfg, model, src_train_loader, src_val_loader,
                         device, class_weights)
    print("Phase 1 complete. Computing source prototypes.")

    # ------------------------------------------------------------------ Prototypes
    protos = compute_source_prototypes(
        model, src_train_loader,
        num_classes=int(cfg["model"]["num_classes"]),
        device=device,
    )
    stats    = log_prototype_stats(protos)
    save_dir = Path(cfg["checkpoint"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    proto_path = save_dir / "prototypes.pt"
    torch.save({int(k): v for k, v in protos.items()}, str(proto_path))
    inter = stats.get("inter_cosine_distance", float("nan"))
    print(f"Prototypes frozen. Inter-class distance: {inter:.4f}  "
          f"(saved → {proto_path})")

    # ------------------------------------------------------------------ Phase 2
    model = train_phase2(cfg, model, protos, src_train_loader, src_val_loader,
                         tgt_loader, device, class_weights,
                         target_eval_loader=target_eval_loader)

    # ------------------------------------------------------------------ Final eval
    src_metrics = compute_all_metrics(model, src_val_loader, device)
    tgt_metrics = (compute_all_metrics(model, target_eval_loader, device)
                   if target_eval_loader is not None else None)

    print("\n============ FINAL METRICS ============", flush=True)
    if tgt_metrics is not None:
        print(f"{'Metric':<12} | {'Source':>7} | {'Target':>7}")
        for k in ("auc", "accuracy", "sensitivity", "specificity", "f1"):
            label = k.capitalize()
            print(f"{label:<12} | {src_metrics[k]:>7.4f} | {tgt_metrics[k]:>7.4f}")
    else:
        print(f"{'Metric':<12} | {'Source':>6}")
        for k in ("auc", "accuracy", "sensitivity", "specificity", "f1"):
            print(f"{k.capitalize():<12} | {src_metrics[k]:>6.4f}")
    print("=======================================\n", flush=True)

    summary = {
        "seed": int(cfg.get("seed", 42)),
        "backbone": getattr(model, "backbone_name", "efficientnet_b0"),
        "feature_dim": int(getattr(model, "feature_dim", -1)),
        "source_dir": cfg["data"]["source_dir"],
        "target_dir": cfg["data"]["target_dir"],
        "target_eval_dir": cfg["data"].get("target_eval_dir"),
        "source_metrics": src_metrics,
        "target_metrics": tgt_metrics,
    }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save raw predictions for later ROC plotting
    src_eval = _collect_eval_outputs(model, src_val_loader, device)
    np.savez_compressed(save_dir / "source_val_roc.npz", split="source_val", **src_eval)
    if target_eval_loader is not None:
        tgt_eval_out = _collect_eval_outputs(model, target_eval_loader, device)
        np.savez_compressed(save_dir / "target_eval_roc.npz",
                            split="target_eval", **tgt_eval_out)

    # Best-target-AUC checkpoint evaluation
    best_tgt_ckpt = save_dir / "phase2_best_target.pt"
    if best_tgt_ckpt.exists() and target_eval_loader is not None:
        tgt_model = DIPNModel(
            num_classes=int(cfg["model"]["num_classes"]),
            pretrained=False,
            backbone=backbone,
        ).to(device)
        load_checkpoint(best_tgt_ckpt, tgt_model, map_location=device)
        best_tgt_metrics  = compute_all_metrics(tgt_model, target_eval_loader, device)
        best_tgt_eval     = _collect_eval_outputs(tgt_model, target_eval_loader, device)
        np.savez_compressed(save_dir / "target_eval_best_target_roc.npz",
                            split="target_eval_best_target", **best_tgt_eval)
        with open(save_dir / "best_target_metrics.json", "w") as f:
            json.dump({"checkpoint": str(best_tgt_ckpt), **best_tgt_metrics}, f, indent=2)
        print(f"[main] best-target-ckpt AUC = {best_tgt_metrics['auc']:.4f}", flush=True)

    final_path = save_dir / "dipn_final.pt"
    save_checkpoint(final_path, model,
                    extra={"config": cfg,
                           "source_metrics": src_metrics,
                           "target_metrics": tgt_metrics})
    print(f"[main] final model saved  → {final_path}", flush=True)
    print(f"[main] summary saved      → {save_dir / 'summary.json'}", flush=True)


def cli() -> None:
    p = argparse.ArgumentParser(description="DIPN training")
    p.add_argument("--config",          type=str, default="configs/dipn.yaml")
    p.add_argument("--source_dir",      type=str, default=None)
    p.add_argument("--target_dir",      type=str, default=None)
    p.add_argument("--target_eval_dir", type=str, default=None,
                   help="Optional labeled target dir (cancer/, healthy/) "
                        "for per-epoch target-AUC tracking.")
    p.add_argument("--seed",            type=int, default=None)
    p.add_argument("--save_subdir",     type=str, default=None,
                   help="Appended to checkpoint.save_dir from the config.")
    p.add_argument("--device",          type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    main(args.config, args.source_dir, args.target_dir, args.device,
         target_eval_dir=args.target_eval_dir,
         seed=args.seed,
         save_subdir=args.save_subdir)


if __name__ == "__main__":
    cli()
