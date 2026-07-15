"""Lightweight, model-free team classification.

The production pipeline currently has no team model.  This module provides a
small colour-prototype classifier that can be trained from player crops and
can be replaced later without changing callers.  It deliberately uses only
NumPy: no OpenCV, Torch, or external model weights are required.

Player crops are expected to be BGR by default because that is the format
used by OpenCV and the rest of the video pipeline.  RGB crops are supported
with ``color_order="RGB"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


UNKNOWN_TEAM_ID = -1
TEAM_IDS = (0, 1)
_EPS = 1e-8


@dataclass(frozen=True)
class TeamPrediction:
    """One classifier output with an explicit UNKNOWN fallback."""

    team_id: int = UNKNOWN_TEAM_ID
    confidence: float = 0.0


class TeamClassifier:
    """Classify teams from robust upper-body colour features.

    ``fit`` accepts optional labels.  With labels, the two team prototypes
    are the mean features for team ``0`` and ``1``.  Without labels, a
    deterministic two-centroid NumPy k-means fit is used for online warmup;
    cluster IDs are assigned deterministically so repeated runs do not swap
    labels arbitrarily.

    An unfitted classifier, an invalid crop, or an ambiguous colour returns
    ``-1``.  ``predict`` preserves the small legacy API and returns only team
    IDs; ``predict_with_confidence`` exposes confidence for trajectory-level
    smoothing.
    """

    feature_dim = 16

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
        self.device = device
        self.batch_size = batch_size
        self.color_order = color_order.upper()
        if self.color_order not in {"BGR", "RGB"}:
            raise ValueError("color_order must be 'BGR' or 'RGB'")
        self.confidence_threshold = float(np.clip(confidence_threshold, 0.5, 0.99))
        self.upper_body_fraction = float(np.clip(upper_body_fraction, 0.2, 1.0))
        self.min_valid_fraction = float(np.clip(min_valid_fraction, 0.0, 1.0))
        self._prototypes: dict[int, np.ndarray] = {}
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted and bool(self._prototypes)

    @property
    def prototypes(self) -> dict[int, np.ndarray]:
        """Return a copy of the learned team prototypes."""

        return {team_id: prototype.copy() for team_id, prototype in self._prototypes.items()}

    def fit(
        self,
        crops: Sequence[np.ndarray],
        labels: Optional[Sequence[int]] = None,
    ) -> "TeamClassifier":
        """Fit team colour prototypes from crops.

        ``labels`` may contain ``0``, ``1`` and ``-1``.  Unknown labels are
        ignored.  If labels are omitted, two clusters are inferred when at
        least two valid crops are available.
        """

        features, valid = self._features_for_crops(crops)
        if labels is not None:
            labels_array = np.asarray(labels, dtype=np.int64)
            if labels_array.ndim != 1 or labels_array.size != len(crops):
                raise ValueError("labels must have one value for every crop")
            known = valid & np.isin(labels_array, TEAM_IDS)
            self._prototypes = {
                team_id: features[known & (labels_array == team_id)].mean(axis=0)
                for team_id in TEAM_IDS
                if np.any(known & (labels_array == team_id))
            }
        else:
            self._prototypes = self._fit_unlabeled(features[valid])

        self._fitted = bool(self._prototypes)
        return self

    def fit_features(
        self,
        features: np.ndarray,
        labels: Optional[Sequence[int]] = None,
    ) -> "TeamClassifier":
        """Fit directly from feature rows, useful for tests and persistence."""

        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (n, {self.feature_dim})")
        valid = np.isfinite(matrix).all(axis=1)
        if labels is not None:
            labels_array = np.asarray(labels, dtype=np.int64)
            if labels_array.ndim != 1 or labels_array.size != matrix.shape[0]:
                raise ValueError("labels must have one value for every feature row")
            self._prototypes = {
                team_id: matrix[valid & (labels_array == team_id)].mean(axis=0)
                for team_id in TEAM_IDS
                if np.any(valid & (labels_array == team_id))
            }
        else:
            self._prototypes = self._fit_unlabeled(matrix[valid])
        self._fitted = bool(self._prototypes)
        return self

    def predict(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Return one team ID per crop, using ``-1`` for unknown."""

        team_ids, _ = self.predict_with_confidence(crops)
        return team_ids

    def predict_with_confidence(
        self,
        crops: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(team_ids, confidences)`` for a sequence of crops."""

        if len(crops) == 0:
            empty_ids = np.empty(0, dtype=np.int64)
            return empty_ids, empty_ids.astype(np.float32)
        if not self.fitted:
            return (
                np.full(len(crops), UNKNOWN_TEAM_ID, dtype=np.int64),
                np.zeros(len(crops), dtype=np.float32),
            )

        features, valid = self._features_for_crops(crops)
        team_ids = np.full(len(crops), UNKNOWN_TEAM_ID, dtype=np.int64)
        confidences = np.zeros(len(crops), dtype=np.float32)
        if not np.any(valid):
            return team_ids, confidences

        available_ids = tuple(sorted(self._prototypes))
        if len(available_ids) < 2:
            # A single observed team is not enough to distinguish the other
            # team; safe fallback is preferable to a false positive label.
            return team_ids, confidences
        distances = np.stack(
            [np.linalg.norm(features - self._prototypes[team_id], axis=1) for team_id in available_ids],
            axis=1,
        )
        # A distance softmax gives a useful confidence and remains stable
        # when only one prototype is available.
        temperature = max(float(np.median(distances[valid])) * 0.5, 0.05)
        logits = -distances / temperature
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), _EPS)
        best_indices = probabilities.argmax(axis=1)
        best_confidences = probabilities[np.arange(len(crops)), best_indices]

        for row in np.flatnonzero(valid):
            confidence = float(best_confidences[row])
            if confidence >= self.confidence_threshold:
                team_ids[row] = available_ids[int(best_indices[row])]
                confidences[row] = confidence
        return team_ids, confidences

    def predict_one(self, crop: np.ndarray) -> TeamPrediction:
        """Convenience wrapper for callers processing one track at a time."""

        team_ids, confidences = self.predict_with_confidence([crop])
        return TeamPrediction(int(team_ids[0]), float(confidences[0]))

    def extract_features(self, crop: np.ndarray) -> np.ndarray:
        """Extract a fixed-size HSV colour descriptor from an upper body."""

        feature, _ = self._extract_feature(crop)
        return feature

    def save(self, path: str | Path) -> None:
        """Persist prototypes without requiring pickle or a model runtime."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        prototype_ids = np.asarray(sorted(self._prototypes), dtype=np.int64)
        prototype_matrix = (
            np.stack([self._prototypes[team_id] for team_id in prototype_ids])
            if prototype_ids.size
            else np.empty((0, self.feature_dim), dtype=np.float32)
        )
        # Opening the file handle prevents NumPy from appending an unwanted
        # .npz suffix when callers use a .joblib/.bin path from configuration.
        with target.open("wb") as handle:
            np.savez(
                handle,
                team_ids=prototype_ids,
                prototypes=prototype_matrix,
                confidence_threshold=np.asarray(self.confidence_threshold),
            )

    def load(self, path: str | Path) -> "TeamClassifier":
        """Load prototypes saved by :meth:`save`."""

        with np.load(Path(path), allow_pickle=False) as data:
            team_ids = np.asarray(data["team_ids"], dtype=np.int64)
            prototypes = np.asarray(data["prototypes"], dtype=np.float32)
            if prototypes.ndim != 2 or prototypes.shape != (team_ids.size, self.feature_dim):
                raise ValueError("invalid team classifier prototype file")
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        features = np.zeros((len(crops), self.feature_dim), dtype=np.float32)
        valid = np.zeros(len(crops), dtype=bool)
        for index, crop in enumerate(crops):
            features[index], valid[index] = self._extract_feature(crop)
        return features, valid

    def _extract_feature(self, crop: np.ndarray) -> Tuple[np.ndarray, bool]:
        array = np.asarray(crop)
        if array.ndim != 3 or array.shape[2] < 3 or array.shape[0] < 2 or array.shape[1] < 2:
            return np.zeros(self.feature_dim, dtype=np.float32), False
        array = array[..., :3]
        height, width = array.shape[:2]
        y_end = max(1, int(round(height * self.upper_body_fraction)))
        x_margin = int(round(width * 0.10))
        x_end = max(x_margin + 1, width - x_margin)
        region = array[:y_end, x_margin:x_end]
        if region.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32), False

        values = region.astype(np.float32)
        if np.issubdtype(array.dtype, np.integer):
            values /= 255.0
        elif float(np.nanmax(values, initial=0.0)) > 1.0:
            values /= 255.0
        if not np.isfinite(values).all():
            return np.zeros(self.feature_dim, dtype=np.float32), False
        values = np.clip(values, 0.0, 1.0)
        if self.color_order == "BGR":
            values = values[..., ::-1]
        hsv = self._rgb_to_hsv(values)
        saturation = hsv[..., 1]
        value = hsv[..., 2]
        valid_pixels = (saturation >= 0.18) & (value >= 0.12)
        valid_fraction = float(valid_pixels.mean())
        if valid_pixels.any():
            hues = hsv[..., 0][valid_pixels]
            weights = saturation[valid_pixels] * (0.5 + 0.5 * value[valid_pixels])
            histogram, _ = np.histogram(
                hues,
                bins=12,
                range=(0.0, 1.0),
                weights=weights,
            )
            histogram = histogram.astype(np.float32)
            histogram /= max(float(histogram.sum()), _EPS)
            mean_saturation = float(saturation[valid_pixels].mean())
            mean_value = float(value[valid_pixels].mean())
        else:
            histogram = np.zeros(12, dtype=np.float32)
            mean_saturation = 0.0
            mean_value = 0.0

        # The last three values make low-information crops distinguishable
        # from a confident colour: mean saturation, value, and valid ratio.
        feature = np.concatenate(
            [histogram, np.asarray([mean_saturation, mean_value, valid_fraction, float(saturation.mean())])]
        ).astype(np.float32)
        return feature, valid_fraction >= self.min_valid_fraction

    @staticmethod
    def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
        maximum = rgb.max(axis=-1)
        minimum = rgb.min(axis=-1)
        delta = maximum - minimum
        hue = np.zeros_like(maximum)
        nonzero = delta > _EPS
        red = (maximum == rgb[..., 0]) & nonzero
        green = (maximum == rgb[..., 1]) & nonzero
        blue = (maximum == rgb[..., 2]) & nonzero
        hue[red] = ((rgb[..., 1][red] - rgb[..., 2][red]) / delta[red]) % 6.0
        hue[green] = (rgb[..., 2][green] - rgb[..., 0][green]) / delta[green] + 2.0
        hue[blue] = (rgb[..., 0][blue] - rgb[..., 1][blue]) / delta[blue] + 4.0
        hue /= 6.0
        saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > _EPS)
        return np.stack([hue, saturation, maximum], axis=-1)

    @staticmethod
    def _fit_unlabeled(features: np.ndarray) -> dict[int, np.ndarray]:
        if features.shape[0] < 2:
            return {}
        # Farthest-point initialization is deterministic and handles the
        # common two-uniform-jersey warmup without random state.
        first = 0
        second = int(np.argmax(np.linalg.norm(features - features[first], axis=1)))
        if second == first:
            return {}
        centroids = np.stack([features[first], features[second]]).astype(np.float32)
        for _ in range(20):
            distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
            assignments = distances.argmin(axis=1)
            updated = np.stack(
                [
                    features[assignments == cluster].mean(axis=0)
                    if np.any(assignments == cluster)
                    else centroids[cluster]
                    for cluster in range(2)
                ]
            )
            if np.allclose(updated, centroids, atol=1e-5):
                centroids = updated
                break
            centroids = updated

        # Make the cluster numbering stable across input ordering.  The
        # feature's hue histogram is the first discriminative component.
        order = np.argsort(np.argmax(centroids[:, :12], axis=1))
        return {team_id: centroids[int(cluster)].copy() for team_id, cluster in enumerate(order)}
