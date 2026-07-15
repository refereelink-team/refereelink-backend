from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from app.classification.team import TeamClassifier

logger = logging.getLogger(__name__)


class OnlineTeamClassifier:
    """Minimal stub — no warmup, no fitting, no heavy models."""

    def __init__(
        self,
        device: str = "cpu",
        warmup_frames: int = 0,
        warmup_stride: int = 1,
        classifier_path: Optional[str] = None,
    ) -> None:
        self._device = device
        self._classifier = TeamClassifier(device=device)
        self._fitted = True

    def collect_and_predict(
        self,
        frame: np.ndarray,
        player_crops: List[np.ndarray],
    ) -> np.ndarray:
        return self._classifier.predict(player_crops)

    @property
    def ready(self) -> bool:
        return True

    def save(self, path: str) -> None:
        logger.info("Team classifier save is a no-op in lightweight mode")

    def _load_classifier(self, path: str) -> None:
        pass
