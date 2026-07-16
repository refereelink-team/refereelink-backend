"""Compatibility façade for the supervised colour prototype classifier.

The production baseline lives in :mod:`app.classification.team_calibration`.
This small façade keeps the historical ``TeamClassifier`` interface for
callers and for the colour-only experiment, but it deliberately requires
labels.  There is no KMeans or other unsupervised fitting path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from app.classification.team_calibration.color_features import ColorFeatureExtractor


UNKNOWN_TEAM_ID = -1
TEAM_IDS = (0, 1)
_EPS = 1e-8


@dataclass(frozen=True)
class TeamPrediction:
    team_id: int = UNKNOWN_TEAM_ID
    confidence: float = 0.0


class TeamClassifier:
    """Labelled HSV/Lab colour prototype classifier.

    This class is retained as a backwards-compatible adapter.  New code
    should use ``SupervisedPrototypeClassifier`` for the fixed production
    fusion path.
    """

    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 32,
        *,
        color_order: str = "BGR",
        confidence_threshold: float = 0.58,
        upper_body_fraction: float = 0.60,
        min_valid_fraction: float = 0.12,
    ) -> None:
        del device, batch_size, color_order, upper_body_fraction, min_valid_fraction
        self.confidence_threshold = float(np.clip(confidence_threshold, 0.5, 0.99))
        self._extractor = ColorFeatureExtractor(spatial_blocks=1)
        self._prototypes: dict[int, np.ndarray] = {}
        self._fitted = False

    @property
    def feature_dim(self) -> int:
        return self._extractor.feature_dim

    @property
    def fitted(self) -> bool:
        return self._fitted and bool(self._prototypes)

    @property
    def prototypes(self) -> dict[int, np.ndarray]:
        return {team_id: prototype.copy() for team_id, prototype in self._prototypes.items()}

    def fit(
        self,
        crops: Sequence[np.ndarray],
        labels: Optional[Sequence[int]] = None,
    ) -> "TeamClassifier":
        if labels is None:
            raise ValueError("labels are required; unsupervised fitting is disabled")
        features, valid = self._features_for_crops(crops)
        labels_array = np.asarray(labels, dtype=np.int64)
        if labels_array.ndim != 1 or labels_array.size != len(crops):
            raise ValueError("labels must have one value for every crop")
        known = valid & np.isin(labels_array, TEAM_IDS)
        self._prototypes = {
            team_id: features[known & (labels_array == team_id)].mean(axis=0)
            for team_id in TEAM_IDS
            if np.any(known & (labels_array == team_id))
        }
        self._fitted = bool(self._prototypes)
        return self

    def fit_features(
        self,
        features: np.ndarray,
        labels: Optional[Sequence[int]] = None,
    ) -> "TeamClassifier":
        if labels is None:
            raise ValueError("labels are required; unsupervised fitting is disabled")
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (n, {self.feature_dim})")
        labels_array = np.asarray(labels, dtype=np.int64)
        if labels_array.ndim != 1 or labels_array.size != matrix.shape[0]:
            raise ValueError("labels must have one value for every feature row")
        valid = np.isfinite(matrix).all(axis=1)
        self._prototypes = {
            team_id: matrix[valid & (labels_array == team_id)].mean(axis=0)
            for team_id in TEAM_IDS
            if np.any(valid & (labels_array == team_id))
        }
        self._fitted = bool(self._prototypes)
        return self

    def predict(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        team_ids, _ = self.predict_with_confidence(crops)
        return team_ids

    def predict_with_confidence(
        self,
        crops: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(crops) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        unknown_ids = np.full(len(crops), UNKNOWN_TEAM_ID, dtype=np.int64)
        unknown_confidences = np.zeros(len(crops), dtype=np.float32)
        if not self.fitted or any(team_id not in self._prototypes for team_id in TEAM_IDS):
            return unknown_ids, unknown_confidences
        features, valid = self._features_for_crops(crops)
        if not np.any(valid):
            return unknown_ids, unknown_confidences
        distances = np.stack(
            [np.linalg.norm(features - self._prototypes[team_id], axis=1) for team_id in TEAM_IDS],
            axis=1,
        )
        order = np.argsort(distances, axis=1)
        best = distances[np.arange(len(crops)), order[:, 0]]
        second = distances[np.arange(len(crops)), order[:, 1]]
        confidence = np.clip(0.5 + (second - best) / np.maximum(second + best, _EPS), 0.0, 1.0)
        for row in np.flatnonzero(valid):
            if float(confidence[row]) >= self.confidence_threshold:
                unknown_ids[row] = TEAM_IDS[int(order[row, 0])]
                unknown_confidences[row] = float(confidence[row])
        return unknown_ids, unknown_confidences

    def predict_one(self, crop: np.ndarray) -> TeamPrediction:
        team_ids, confidences = self.predict_with_confidence([crop])
        return TeamPrediction(int(team_ids[0]), float(confidences[0]))

    def extract_features(self, crop: np.ndarray) -> np.ndarray:
        return self._extractor.extract(crop)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        prototype_ids = np.asarray(sorted(self._prototypes), dtype=np.int64)
        prototype_matrix = (
            np.stack([self._prototypes[team_id] for team_id in prototype_ids])
            if prototype_ids.size
            else np.empty((0, self.feature_dim), dtype=np.float32)
        )
        with target.open("wb") as handle:
            np.savez(
                handle,
                team_ids=prototype_ids,
                prototypes=prototype_matrix,
                confidence_threshold=np.asarray(self.confidence_threshold),
            )

    def load(self, path: str | Path) -> "TeamClassifier":
        with np.load(Path(path), allow_pickle=False) as data:
            team_ids = np.asarray(data["team_ids"], dtype=np.int64)
            prototypes = np.asarray(data["prototypes"], dtype=np.float32)
            if prototypes.ndim != 2 or prototypes.shape != (team_ids.size, self.feature_dim):
                raise ValueError("invalid supervised colour prototype file")
            self._prototypes = {
                int(team_id): prototypes[index].copy()
                for index, team_id in enumerate(team_ids)
                if int(team_id) in TEAM_IDS
            }
            if "confidence_threshold" in data:
                self.confidence_threshold = float(
                    np.clip(float(data["confidence_threshold"]), 0.5, 0.99)
                )
        self._fitted = bool(self._prototypes)
        return self

    def _features_for_crops(
        self,
        crops: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        features = self._extractor.extract_batch(list(crops))
        valid = np.isfinite(features).all(axis=1) & (np.linalg.norm(features, axis=1) > _EPS)
        return features, valid
