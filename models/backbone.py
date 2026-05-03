"""DIPN model — multi-backbone factory with L2-normalized features.

The default backbone (`efficientnet_b0`) is constructed exactly as it was
in the very first DIPN release (separate `features`, `pool`, `flatten`,
`classifier` attributes), so checkpoints saved by earlier runs remain
load-compatible with this class. All other backbones are loaded via `timm`
to get a uniform `forward -> pooled vector` interface across architectures
(CNN / ViT / Swin / ConvNeXt).
"""
from __future__ import annotations

from typing import Callable, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


def _timm_loader(timm_name: str) -> Callable[[bool], Tuple[nn.Module, int]]:
    def _load(pretrained: bool) -> Tuple[nn.Module, int]:
        import timm
        m = timm.create_model(
            timm_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        with torch.no_grad():
            d = int(m(torch.zeros(1, 3, 224, 224)).shape[-1])
        return m, d
    return _load


TIMM_REGISTRY = {
    "resnet50":          _timm_loader("resnet50"),
    "densenet121":       _timm_loader("densenet121"),
    "densenet169":       _timm_loader("densenet169"),
    "convnext_tiny":     _timm_loader("convnext_tiny"),
    "vit_b_16":          _timm_loader("vit_base_patch16_224"),
    "mae":               _timm_loader("vit_base_patch16_224.mae"),
    "swin_tiny":         _timm_loader("swin_tiny_patch4_window7_224"),
    "dinovit":           _timm_loader("vit_small_patch16_224.dino"),
    "efficientnet_v2_s": _timm_loader("tf_efficientnetv2_s"),
}

AVAILABLE_BACKBONES = ["efficientnet_b0", *sorted(TIMM_REGISTRY)]


class DIPNModel(nn.Module):
    """Backbone + linear classifier with L2-normalized features.

    forward(x, return_features=False):
        return_features=True  -> (features [B, D] L2-normalized, logits [B, C])
        return_features=False -> logits [B, C]
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        backbone: str = "efficientnet_b0",
    ) -> None:
        super().__init__()
        self.backbone_name: str = backbone

        if backbone == "efficientnet_b0":
            weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            base = tvm.efficientnet_b0(weights=weights)
            self.features: nn.Module = base.features
            self.pool: nn.Module = nn.AdaptiveAvgPool2d(1)
            self.flatten: nn.Module = nn.Flatten()
            self.feature_dim: int = 1280
            self.classifier: nn.Module = nn.Linear(self.feature_dim, num_classes)
            self._uses_efficientnet_path: bool = True
        else:
            if backbone not in TIMM_REGISTRY:
                raise ValueError(
                    f"Unknown backbone '{backbone}'. "
                    f"Available: {AVAILABLE_BACKBONES}"
                )
            loader = TIMM_REGISTRY[backbone]
            self.feature_extractor, self.feature_dim = loader(pretrained)
            self.classifier: nn.Module = nn.Linear(self.feature_dim, num_classes)
            self._uses_efficientnet_path: bool = False

    def extract_features(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Return pooled feature vector.

        normalize=True  (default, DIPN behaviour): L2-normalized features.
        normalize=False: raw pre-normalization features — needed for baselines
        like Deep CORAL whose loss formulation assumes non-unit-norm features.
        """
        if self._uses_efficientnet_path:
            f = self.features(x)
            f = self.pool(f)
            f = self.flatten(f)
        else:
            f = self.feature_extractor(x)
        if normalize:
            return F.normalize(f, p=2, dim=1)
        return f

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        feats_norm = self.extract_features(x)
        logits = self.classifier(feats_norm)
        if return_features:
            return feats_norm, logits
        return logits
