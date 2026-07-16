"""Online warmup wrapper for the lightweight team classifier."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from app.classification.team import TeamClassifier

logger = logging.getLogger(__name__)


class OnlineTeamClassifier:
    """Collect crops during warmup, then classify each crop independently.

    The wrapper has no model-runtime dependency.  ``warmup_frames=0`` keeps
    the historical always-ready behaviour but intentionally returns UNKNOWN
    until a classifier is fitted or loaded.  For online unsupervised fitting,
    set a positive warmup frame count and provide crops from both teams.
    """

    def __init__(
        self,
        device: str = "cpu",
        warmup_frames: int = 0,
        warmup_stride: int = 1,
        classifier_path: Optional[str] = None,
        *,
        classifier: Optional[TeamClassifier] = None,
        min_warmup_crops: int = 2,
    ) -> None:
        if warmup_frames < 0:
            raise ValueError("warmup_frames must be non-negative")
        if warmup_stride < 1:
            raise ValueError("warmup_stride must be at least 1")
        if min_warmup_crops < 2:
            raise ValueError("min_warmup_crops must be at least 2")
        self._device = device
        self._classifier = classifier or TeamClassifier(device=device)
        self._warmup_frames = int(warmup_frames)
        self._warmup_stride = int(warmup_stride)
        self._min_warmup_crops = int(min_warmup_crops)
        self._frame_count = 0
        self._warmup_crops: list[np.ndarray] = []
        self._fitted = self._classifier.fitted
        if classifier_path:
            self._load_classifier(classifier_path)

    @property
    def classifier(self) -> TeamClassifier:
        return self._classifier

    @property
    def ready(self) -> bool:
        """Whether the wrapper can serve predictions without blocking."""

        return True

    @property
    def fitted(self) -> bool:
        return self._fitted and self._classifier.fitted

    @property
    def warmup_progress(self) -> float:
        if self._warmup_frames == 0:
            return 1.0 if self.fitted else 0.0
        return float(np.clip(self._frame_count / self._warmup_frames, 0.0, 1.0))

    def collect_and_predict(
        self,
        frame: np.ndarray,
        player_crops: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Collect a stride-selected frame and return team IDs for crops."""

        del frame  # Kept in the API for callers that already pass the frame.
        self._frame_count += 1
        if (
            not self.fitted
            and self._warmup_frames > 0
            and self._frame_count % self._warmup_stride == 0
        ):
            self._warmup_crops.extend(np.asarray(crop) for crop in player_crops)
            if self._frame_count >= self._warmup_frames and len(self._warmup_crops) >= self._min_warmup_crops:
                self.fit(self._warmup_crops)
                self._warmup_crops.clear()
        return self._classifier.predict(player_crops)

    def predict_with_confidence(
        self,
        player_crops: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._classifier.predict_with_confidence(player_crops)

    def fit(
        self,
        player_crops: Sequence[np.ndarray],
        labels: Optional[Sequence[int]] = None,
    ) -> "OnlineTeamClassifier":
        self._classifier.fit(player_crops, labels=labels)
        self._fitted = self._classifier.fitted
        return self

    def save(self, path: str) -> None:
        self._classifier.save(path)
        logger.info("Saved lightweight team classifier to %s", path)

    def _load_classifier(self, path: str) -> None:
        candidate = Path(path)
        if not candidate.exists():
            logger.warning("Team classifier file not found: %s; using UNKNOWN fallback", path)
            return
        try:
            self._classifier.load(candidate)
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Could not load team classifier %s: %s", path, exc)
            return
        self._fitted = self._classifier.fitted
