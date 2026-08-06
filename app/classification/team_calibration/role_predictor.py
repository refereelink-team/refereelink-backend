"""Supervised role prediction for calibrated match participants.

The official YOLO person model deliberately has no role information.  This
module turns the tracks manually labelled during the pre-match review into a
small, frozen role classifier.  A role-aware YOLO checkpoint may be supplied
as a fallback, but a missing checkpoint never discards the supervised
calibration prototypes.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from app.classification.team_calibration.appearance_features import AppearanceFeatureExtractor
from app.classification.team_calibration.color_features import ColorFeatureExtractor
from app.classification.team_calibration.quality import CropQualityAssessor
from app.classification.team_calibration.roi import JerseyROIExtractor
from app.classification.team_calibration.types import (
    PlayerRole,
    RolePrediction,
    TeamLabel,
    TeamPrototype,
)
ROLE_ORDER = (PlayerRole.OUTFIELD, PlayerRole.GOALKEEPER, PlayerRole.REFEREE)


def normalize_role(role: object) -> str:
    """Normalize role values without importing the semantic runtime.

    Keeping this small boundary helper local avoids a package import cycle:
    the semantic runtime imports calibration types, while calibration role
    prediction is also used by the semantic runtime.
    """

    if role is None:
        return PlayerRole.UNKNOWN.value
    value = getattr(role, "value", role)
    normalized = str(value).strip().lower().replace("_", "")
    aliases = {
        "player": PlayerRole.OUTFIELD.value,
        "outfield": PlayerRole.OUTFIELD.value,
        "goalkeeper": PlayerRole.GOALKEEPER.value,
        "keeper": PlayerRole.GOALKEEPER.value,
        "referee": PlayerRole.REFEREE.value,
        "staff": PlayerRole.STAFF.value,
        "unknown": PlayerRole.UNKNOWN.value,
    }
    return aliases.get(normalized, PlayerRole.UNKNOWN.value)


class SupervisedRolePrototypeClassifier:
    """Classify roles using frozen, manually labelled appearance prototypes."""

    def __init__(
        self,
        prototypes: Optional[Mapping[tuple[TeamLabel, PlayerRole], TeamPrototype]] = None,
        *,
        color_weight: float = 0.6,
        deep_weight: float = 0.4,
        min_observations: int = 2,
        min_margin: float = 0.04,
        max_distance: float = 0.75,
    ) -> None:
        if color_weight < 0 or deep_weight < 0 or color_weight + deep_weight <= 0:
            raise ValueError("feature weights must be non-negative and not both zero")
        self.prototypes = dict(prototypes or {})
        self.color_weight = float(color_weight)
        self.deep_weight = float(deep_weight)
        self.min_observations = max(1, int(min_observations))
        self.min_margin = float(min_margin)
        self.max_distance = float(max_distance)

    @property
    def ready(self) -> bool:
        return any(
            (team, role) in self.prototypes
            for team, role in self.prototypes
            if role in ROLE_ORDER
        )

    def predict(
        self,
        *,
        color_feature: Optional[np.ndarray],
        deep_feature: Optional[np.ndarray],
        observation_count: int = 0,
    ) -> RolePrediction:
        if observation_count < self.min_observations:
            return self._unknown("observations_insufficient", observation_count)
        if color_feature is None and deep_feature is None:
            return self._unknown("feature_unavailable", observation_count)

        distances: dict[PlayerRole, float] = {}
        for role in ROLE_ORDER:
            candidates = [
                prototype
                for (team, candidate_role), prototype in self.prototypes.items()
                if candidate_role == role
                and (
                    role == PlayerRole.REFEREE
                    and team == TeamLabel.NONE
                    or role != PlayerRole.REFEREE
                    and team in {TeamLabel.HOME, TeamLabel.AWAY}
                )
            ]
            role_distances = [
                self._distance(prototype, color_feature, deep_feature)
                for prototype in candidates
            ]
            finite = [distance for distance in role_distances if np.isfinite(distance)]
            if finite:
                distances[role] = min(finite)

        if len(distances) == 1:
            only_role, only_distance = next(iter(distances.items()))
            if only_role == PlayerRole.OUTFIELD and only_distance <= self.max_distance:
                return RolePrediction(
                    role=only_role,
                    confidence=0.6,
                    best_distance=float(only_distance),
                    observation_count=observation_count,
                )
        if len(distances) < 2:
            return self._unknown("role_prototypes_insufficient", observation_count)
        ordered = sorted(distances.items(), key=lambda item: (item[1], item[0].value))
        best_role, best_distance = ordered[0]
        second_distance = ordered[1][1]
        margin = float(second_distance - best_distance)
        if best_distance > self.max_distance:
            return self._unknown(
                "role_distance_too_large",
                observation_count,
                margin,
                best_distance,
            )
        if margin < self.min_margin:
            return self._unknown(
                "role_margin_too_small",
                observation_count,
                margin,
                best_distance,
            )
        confidence = float(
            np.clip(
                0.5 + margin / max(second_distance + best_distance + 1e-8, 1e-8),
                0.0,
                1.0,
            )
        )
        return RolePrediction(
            role=best_role,
            confidence=confidence,
            margin=margin,
            best_distance=float(best_distance),
            observation_count=observation_count,
        )

    def _distance(
        self,
        prototype: TeamPrototype,
        color_feature: Optional[np.ndarray],
        deep_feature: Optional[np.ndarray],
    ) -> float:
        distances: list[float] = []
        weights: list[float] = []
        if color_feature is not None and prototype.color_feature is not None:
            # Both descriptors are L2-normalized.  Dividing by their maximum
            # Euclidean distance puts the color modality on approximately the
            # same [0, 1] scale as cosine distance.
            color = float(np.linalg.norm(color_feature - prototype.color_feature) / np.sqrt(2.0))
            distances.append(color)
            weights.append(self.color_weight)
        if deep_feature is not None and prototype.deep_feature is not None:
            current = np.asarray(deep_feature, dtype=np.float32)
            norm = float(np.linalg.norm(current))
            if norm > 1e-8:
                current = current / norm
            deep = float(np.clip(1.0 - np.dot(current, prototype.deep_feature), 0.0, 2.0) / 2.0)
            distances.append(deep)
            weights.append(self.deep_weight)
        if not distances:
            return float("inf")
        return float(np.average(distances, weights=weights))

    @staticmethod
    def _unknown(
        reason: str,
        observation_count: int,
        margin: float = 0.0,
        best_distance: float = float("inf"),
    ) -> RolePrediction:
        return RolePrediction(
            role=PlayerRole.UNKNOWN,
            confidence=0.0,
            margin=float(margin),
            best_distance=float(best_distance),
            observation_count=observation_count,
            rejection_reason=reason,
        )


class CalibratedRoleClassifier:
    """Crop adapter combining supervised role prototypes and an optional model."""

    def __init__(
        self,
        *,
        prototypes: Optional[Mapping[tuple[TeamLabel, PlayerRole], TeamPrototype]] = None,
        model: Optional[object] = None,
        color_extractor: Optional[ColorFeatureExtractor] = None,
        appearance_extractor: Optional[AppearanceFeatureExtractor] = None,
        roi_extractor: Optional[JerseyROIExtractor] = None,
        quality_assessor: Optional[CropQualityAssessor] = None,
        prototype_classifier: Optional[SupervisedRolePrototypeClassifier] = None,
    ) -> None:
        self.model = model
        self.color_extractor = color_extractor or ColorFeatureExtractor()
        self.appearance_extractor = appearance_extractor
        self.roi_extractor = roi_extractor or JerseyROIExtractor()
        self.quality_assessor = quality_assessor or CropQualityAssessor()
        self.prototype_classifier = prototype_classifier or SupervisedRolePrototypeClassifier(prototypes)

    @property
    def ready(self) -> bool:
        return self.prototype_classifier.ready or self.model is not None

    def predict_with_confidence(
        self, crops: Sequence[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        if not crops:
            return np.empty(0, dtype=object), np.empty(0, dtype=np.float32)

        roi_crops: list[np.ndarray] = []
        accepted: list[bool] = []
        for crop in crops:
            image = np.asarray(crop)
            roi = self.roi_extractor.extract(
                image,
                [
                    0.0,
                    0.0,
                    float(image.shape[1]) if image.ndim >= 2 else 0.0,
                    float(image.shape[0]) if image.ndim >= 2 else 0.0,
                ],
            )
            assessment = self.quality_assessor.assess(roi, detection_confidence=1.0)
            roi_crops.append(roi)
            accepted.append(assessment.accepted)

        deep_features: list[Optional[np.ndarray]] = [None] * len(crops)
        selected = [crop for crop, is_accepted in zip(roi_crops, accepted) if is_accepted]
        if selected and self.appearance_extractor is not None:
            values = self.appearance_extractor.extract_batch(selected)
            selected_index = 0
            for index, is_accepted in enumerate(accepted):
                if is_accepted:
                    deep_features[index] = values[selected_index]
                    selected_index += 1

        values = np.full(len(crops), PlayerRole.UNKNOWN.value, dtype=object)
        confidences = np.zeros(len(crops), dtype=np.float32)
        prototype_predictions: list[RolePrediction] = []
        for index, (roi, is_accepted) in enumerate(zip(roi_crops, accepted)):
            prediction = (
                self.prototype_classifier.predict(
                    color_feature=self.color_extractor.extract(roi),
                    deep_feature=deep_features[index],
                    observation_count=2,
                )
                if is_accepted
                else RolePrediction(rejection_reason="roi_quality_rejected")
            )
            prototype_predictions.append(prediction)
            if prediction.role != PlayerRole.UNKNOWN:
                values[index] = prediction.role.value
                confidences[index] = prediction.confidence

        if self.model is not None:
            model_values, model_confidences = _model_predictions(self.model, crops)
            for index, prediction in enumerate(prototype_predictions):
                if prediction.role != PlayerRole.UNKNOWN:
                    continue
                model_role = normalize_role(model_values[index])
                if model_role != PlayerRole.UNKNOWN:
                    values[index] = model_role
                    confidences[index] = float(model_confidences[index])
        return values, confidences

    def predict(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        values, _ = self.predict_with_confidence(crops)
        return values


def _model_predictions(model: object, crops: Sequence[np.ndarray]) -> tuple[list[object], list[float]]:
    if hasattr(model, "predict_with_confidence"):
        values, confidences = model.predict_with_confidence(crops)
        return list(np.asarray(values).reshape(-1)), [float(value) for value in np.asarray(confidences).reshape(-1)]
    if hasattr(model, "predict"):
        values = model.predict(crops)
        return list(np.asarray(values).reshape(-1)), [0.5] * len(crops)
    raise AttributeError("role model must provide predict_with_confidence or predict")
