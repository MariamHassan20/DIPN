"""Source and target datasets for DIPN."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((int(image_size * 1.15), int(image_size * 1.15))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class SourceDataset(Dataset):
    """Labeled source dataset with cancer/ and healthy/ subfolders."""

    def __init__(
        self,
        source_dir: str | Path,
        image_size: int = 224,
        split: str = "train",
        val_split: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.image_size = image_size
        self.split = split

        cancer_dir = self.source_dir / "cancer"
        healthy_dir = self.source_dir / "healthy"
        if not cancer_dir.is_dir() or not healthy_dir.is_dir():
            raise FileNotFoundError(
                f"source_dir must contain 'cancer/' and 'healthy/' subfolders. "
                f"Got {self.source_dir}"
            )

        cancer_imgs = _list_images(cancer_dir)
        healthy_imgs = _list_images(healthy_dir)
        if len(cancer_imgs) < 10 or len(healthy_imgs) < 10:
            raise ValueError(
                f"Need >=10 images per class. "
                f"Got cancer={len(cancer_imgs)}, healthy={len(healthy_imgs)}"
            )

        g = torch.Generator().manual_seed(seed)

        def _stratified(imgs: List[Path], label: int) -> List[Tuple[Path, int]]:
            n = len(imgs)
            n_val = max(1, int(round(n * val_split)))
            perm = torch.randperm(n, generator=g).tolist()
            val_idx = set(perm[:n_val])
            items: List[Tuple[Path, int]] = []
            for i, p in enumerate(imgs):
                in_val = i in val_idx
                if split == "train" and not in_val:
                    items.append((p, label))
                elif split == "val" and in_val:
                    items.append((p, label))
            return items

        items = _stratified(cancer_imgs, 1) + _stratified(healthy_imgs, 0)
        if not items:
            raise ValueError(f"No images selected for split='{split}'")
        self.items = items

        self.transform = (_train_transform(image_size)
                          if split == "train"
                          else _eval_transform(image_size))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

    def get_labels(self) -> List[int]:
        return [label for _, label in self.items]


class TargetDataset(Dataset):
    """Unlabeled target dataset (flat folder, no subfolders)."""

    def __init__(
        self,
        target_dir: str | Path,
        image_size: int = 224,
        split: str = "train",
    ) -> None:
        self.target_dir = Path(target_dir)
        if not self.target_dir.is_dir():
            raise FileNotFoundError(f"target_dir does not exist: {self.target_dir}")
        paths = _list_images(self.target_dir)
        if not paths:
            # Maybe the user accidentally passed a nested-folder layout; flatten.
            paths = sorted([p for p in self.target_dir.rglob("*")
                            if p.is_file() and p.suffix.lower() in IMG_EXTS])
        if not paths:
            raise ValueError(f"No images found in target_dir: {self.target_dir}")
        self.paths = paths
        self.transform = (_train_transform(image_size)
                          if split == "train"
                          else _eval_transform(image_size))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def get_class_weights(dataset: SourceDataset) -> Tensor:
    """Return inverse-frequency weights per class (length = num classes)."""
    labels = dataset.get_labels()
    if not labels:
        raise ValueError("Dataset is empty; cannot compute class weights.")
    num_classes = max(labels) + 1
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for y in labels:
        counts[y] += 1
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights / weights.sum() * num_classes  # keep scale ~1
