"""Distribution corrector — Bayes prior-shift for target predictions."""
from __future__ import annotations

import torch


class DistributionCorrector:
    """Estimates target class prior via EMA and corrects soft predictions.

    p_corrected(c|x) ∝ p(c|x) * π_s_c / π_t_c
    """

    def __init__(
        self,
        source_prior: torch.Tensor,
        ema_momentum: float = 0.99,
        clamp_min: float = 0.05,
        clamp_max: float = 0.95,
    ) -> None:
        source_prior = source_prior.float().clone()
        if torch.any(source_prior <= 0):
            raise ValueError("source_prior must be strictly positive")
        source_prior = source_prior / source_prior.sum()
        self.source_prior = source_prior
        self.ema_momentum = float(ema_momentum)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        # Initialise target prior to uniform
        C = int(source_prior.numel())
        self.target_prior = torch.full((C,), 1.0 / C)
        self._ready = False

    # ------------------------------------------------------------------
    def _to_device(self, x: torch.Tensor) -> None:
        if self.source_prior.device != x.device:
            self.source_prior = self.source_prior.to(x.device)
        if self.target_prior.device != x.device:
            self.target_prior = self.target_prior.to(x.device)

    # ------------------------------------------------------------------
    def update(self, batch_probs: torch.Tensor) -> None:
        """Update EMA of target class distribution with a batch of probs [B, C]."""
        with torch.no_grad():
            self._to_device(batch_probs)
            batch_mean = batch_probs.mean(dim=0)
            batch_mean = batch_mean / batch_mean.sum().clamp(min=1e-8)
            m = self.ema_momentum
            self.target_prior = m * self.target_prior + (1.0 - m) * batch_mean

    # ------------------------------------------------------------------
    def mark_ready(self) -> None:
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------
    def correct(self, soft_probs: torch.Tensor) -> torch.Tensor:
        """Apply prior correction. During warmup, return soft_probs unchanged."""
        if not self._ready:
            return soft_probs
        self._to_device(soft_probs)
        ratio = self.source_prior / self.target_prior.clamp(min=1e-6)
        corrected = soft_probs * ratio.unsqueeze(0)
        corrected = corrected / corrected.sum(dim=1, keepdim=True).clamp(min=1e-8)
        corrected = corrected.clamp(self.clamp_min, self.clamp_max)
        corrected = corrected / corrected.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return corrected

    # ------------------------------------------------------------------
    def get_target_prior(self) -> torch.Tensor:
        return self.target_prior.detach().cpu().clone()

    def get_source_prior(self) -> torch.Tensor:
        return self.source_prior.detach().cpu().clone()
