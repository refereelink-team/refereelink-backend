"""Optional MVFoul-style clip classifier with graceful degradation.

Prefers an external ``offside.foul_model`` (fouls_far) when available.  When the
dependency or checkpoint is missing, classification returns ``None`` so the
pipeline can keep emitting geometry-only foul candidates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FoulPrediction:
    confidence: float
    offence: str = "unknown"
    action: str = "unknown"
    severity: str = "unknown"


def load_mvfoul_model(checkpoint_path: str, device: str = "cpu") -> Any:
    """Load MVFoul weights via fouls_far/offside when present."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"MVFoul checkpoint not found: {checkpoint_path}")
    try:
        from offside.foul_model import load_mvfoul_model as _load
    except ImportError as exc:
        raise ImportError(
            "offside.foul_model unavailable; place fouls_far/offside on PYTHONPATH "
            "or install the foul model package. Geometry foul candidates still work."
        ) from exc
    return _load(checkpoint_path, device=device)


def predict_foul_from_frames(
    frames: np.ndarray,
    model: Any,
    *,
    input_fps: float = 25.0,
    target_fps: float = 17.0,
    device: str = "cpu",
) -> Optional[FoulPrediction]:
    """Run clip classification; map unknown APIs into :class:`FoulPrediction`."""
    try:
        from offside.foul_model import predict_foul_from_frames as _predict
    except ImportError:
        logger.debug("offside.foul_model missing; skip clip classification")
        return None
    raw = _predict(
        frames,
        model,
        input_fps=input_fps,
        target_fps=target_fps,
        device=device,
    )
    if raw is None:
        return None
    if isinstance(raw, FoulPrediction):
        return raw
    confidence = 0.0
    for name in ("confidence", "probability", "score", "offence_confidence"):
        value = getattr(raw, name, None)
        if value is not None:
            try:
                confidence = float(np.clip(float(value), 0.0, 1.0))
                break
            except (TypeError, ValueError):
                pass
    details: dict[str, str] = {}
    for name in ("offence", "action", "label", "severity"):
        value = getattr(raw, name, None)
        if value is not None:
            details[name] = str(value.value if hasattr(value, "value") else value)
    return FoulPrediction(
        confidence=confidence,
        offence=details.get("offence", "unknown"),
        action=details.get("action", details.get("label", "unknown")),
        severity=details.get("severity", "unknown"),
    )
