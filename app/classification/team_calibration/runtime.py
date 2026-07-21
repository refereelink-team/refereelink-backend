from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from app.classification.team_calibration.appearance_features import AppearanceFeatureExtractor
from app.classification.team_calibration.color_features import ColorFeatureExtractor
from app.classification.team_calibration.predictor import SupervisedPrototypeClassifier
from app.classification.team_calibration.quality import CropQualityAssessor
from app.classification.team_calibration.roi import JerseyROIExtractor
from app.classification.team_calibration.track_features import TrackFeatureBank
from app.classification.team_calibration.types import (
    PlayerRole,
    TeamLabel,
    TeamPrediction,
    TeamPrototype,
)


class TeamAssignmentService:
    """Extract, aggregate and classify one observation per active Track.

    Feature EMA belongs here; final role/team label hysteresis remains in
    ``TrackSemanticManager``.  The service never changes the prototypes it
    receives.
    """

    def __init__(
        self,
        prototypes: Optional[Mapping[tuple[TeamLabel, PlayerRole], TeamPrototype]] = None,
        *,
        classifier: Optional[SupervisedPrototypeClassifier] = None,
        color_extractor: Optional[ColorFeatureExtractor] = None,
        appearance_extractor: Optional[AppearanceFeatureExtractor] = None,
        roi_extractor: Optional[JerseyROIExtractor] = None,
        quality_assessor: Optional[CropQualityAssessor] = None,
        feature_bank: Optional[TrackFeatureBank] = None,
        require_appearance: bool = True,
    ) -> None:
        self.color_extractor = color_extractor or ColorFeatureExtractor()
        self.appearance_extractor = appearance_extractor
        self.roi_extractor = roi_extractor or JerseyROIExtractor()
        self.quality_assessor = quality_assessor or CropQualityAssessor()
        self.feature_bank = feature_bank or TrackFeatureBank()
        self.classifier = classifier or SupervisedPrototypeClassifier(prototypes)
        self.require_appearance = bool(require_appearance)

    @property
    def ready(self) -> bool:
        if not self.classifier.ready:
            return False
        return not self.require_appearance or (
            self.appearance_extractor is not None and self.appearance_extractor.loaded
        )

    def predict_tracks(
        self,
        crops: Sequence[np.ndarray],
        *,
        track_ids: Sequence[int],
        roles: Sequence[PlayerRole],
        detection_confidences: Optional[Sequence[float]] = None,
        frame_index: int,
    ) -> list[TeamPrediction]:
        if len(crops) != len(track_ids) or len(crops) != len(roles):
            raise ValueError("crops, track_ids and roles must have equal length")
        confidence_values = (
            list(detection_confidences)
            if detection_confidences is not None
            else [1.0] * len(crops)
        )
        roi_crops: list[np.ndarray] = []
        quality_values = []
        accepted = []
        for crop, confidence in zip(crops, confidence_values):
            image = np.asarray(crop)
            roi = self.roi_extractor.extract(
                image,
                [0.0, 0.0, float(image.shape[1]) if image.ndim >= 2 else 0.0,
                 float(image.shape[0]) if image.ndim >= 2 else 0.0],
            )
            assessment = self.quality_assessor.assess(
                roi,
                detection_confidence=float(confidence),
            )
            roi_crops.append(roi)
            quality_values.append(assessment.score if assessment.accepted else 0.0)
            accepted.append(assessment.accepted)

        color_features = [
            self.color_extractor.extract(crop) if is_accepted else None
            for crop, is_accepted in zip(roi_crops, accepted)
        ]
        deep_features: list[Optional[np.ndarray]] = [None] * len(crops)
        if self.appearance_extractor is not None and any(accepted):
            selected = [crop for crop, is_accepted in zip(roi_crops, accepted) if is_accepted]
            selected_features = self.appearance_extractor.extract_batch(selected)
            selected_index = 0
            for index, is_accepted in enumerate(accepted):
                if is_accepted:
                    deep_features[index] = selected_features[selected_index]
                    selected_index += 1

        predictions: list[TeamPrediction] = []
        for index, (track_id, role) in enumerate(zip(track_ids, roles)):
            current = self.feature_bank.update(
                int(track_id),
                team=TeamLabel.UNKNOWN,
                role=role,
                color_feature=color_features[index],
                deep_feature=deep_features[index],
                quality=quality_values[index],
                frame_index=frame_index,
            )
            prediction = self.classifier.predict(
                color_feature=current.color_feature,
                deep_feature=current.deep_feature,
                role=role,
                observation_count=current.observation_count,
            )
            predictions.append(prediction)
        self.feature_bank.remove_stale(frame_index)
        return predictions
