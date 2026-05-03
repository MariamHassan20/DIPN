"""Checkpoint helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"model_state": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    payload = torch.load(str(path), map_location=map_location)
    state = payload.get("model_state", payload)
    model.load_state_dict(state)
    return payload
