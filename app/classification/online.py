"""Runtime adapter for the supervised team classifier.

The historical name is retained because the pipeline imports it, but this
class no longer performs online warmup or unsupervised fitting.  Calibration
is an explicit pre-match operation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from app.classification.team import TeamClassifier

logger = logging.getLogger(__name__)


class OnlineTeamClassifier:
    """Strict labelled classifier adapter with UNKNOWN before calibration."""

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
        del device, warmup_frames, warmup_stride, min_warmup_crops
        self._classifier = classifier or TeamClassifier()
        self._fitted = self._classifier.fitted
        if classifier_path:
            self._load_classifier(classifier_path)

    @property
    def classifier(self) -> TeamClassifier:
        return self._classifier

    @property
    def ready(self) -> bool:
        return self.fitted

    @property
    def fitted(self) -> bool:
        return self._fitted and self._classifier.fitted

    @property
    def warmup_progress(self) -> float:
        return 1.0 if self.fitted else 0.0

    def collect_and_predict(
        self,
        frame: np.ndarray,
        player_crops: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Compatibility method; it never mutates the classifier."""

        del frame
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
        if labels is None:
            raise ValueError("labels are required; runtime warmup is disabled")
        self._classifier.fit(player_crops, labels=labels)
        self._fitted = self._classifier.fitted
        return self

    def save(self, path: str) -> None:
        self._classifier.save(path)
        logger.info("Saved supervised team classifier to %s", path)

    def _load_classifier(self, path: str) -> None:
        candidate = Path(path)
        if not candidate.exists():
            logger.warning("Supervised team classifier file not found: %s", path)
            return
        try:
            self._classifier.load(candidate)
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Could not load supervised team classifier %s: %s", path, exc)
            return
        self._fitted = self._classifier.fitted
