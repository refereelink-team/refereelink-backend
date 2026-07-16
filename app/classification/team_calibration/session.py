from __future__ import annotations

import threading
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.classification.team_calibration.appearance_features import AppearanceFeatureExtractor
from app.classification.team_calibration.bundle import CalibrationBundle
from app.classification.team_calibration.color_features import ColorFeatureExtractor
from app.classification.team_calibration.quality import CropQualityAssessor
from app.classification.team_calibration.roi import JerseyROIExtractor
from app.classification.team_calibration.track_features import TrackFeatureBank
from app.classification.team_calibration.types import (
    CalibrationLabel,
    PlayerRole,
    TeamLabel,
    ValidationReport,
)
from app.classification.team_calibration.validation import CalibrationValidator


class CalibrationState(str, Enum):
    IDLE = "idle"
    CALIBRATING = "calibrating"
    VALIDATING = "validating"
    READY = "ready"
    RUNNING = "running"
    RECALIBRATION_REQUIRED = "recalibration_required"


class TeamCalibrationSession:
    """Thread-safe pre-match Track labelling and sample collection session."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        min_tracks_per_team: int = 3,
        min_samples_per_track: int = 5,
        max_samples_per_track: int = 15,
        require_appearance: bool = True,
        appearance_extractor: Optional[AppearanceFeatureExtractor] = None,
        validator: Optional[CalibrationValidator] = None,
    ) -> None:
        self.device = device
        self.min_samples_per_track = int(min_samples_per_track)
        self.max_samples_per_track = int(max_samples_per_track)
        self.require_appearance = bool(require_appearance)
        self.roi_extractor = JerseyROIExtractor()
        self.quality_assessor = CropQualityAssessor()
        self.color_extractor = ColorFeatureExtractor()
        self.appearance_extractor = appearance_extractor
        self.validator = validator or CalibrationValidator(
            min_tracks_per_team=min_tracks_per_team,
            min_samples_per_track=min_samples_per_track,
        )
        self._lock = threading.RLock()
        self._state = CalibrationState.IDLE
        self._match_id = ""
        self._camera_id = ""
        self._bundle_path: Optional[str] = None
        self._labels: dict[int, CalibrationLabel] = {}
        self._feature_bank = TrackFeatureBank()
        self._report: Optional[ValidationReport] = None
        self._bundle: Optional[CalibrationBundle] = None
        self._last_frame_index: Optional[int] = None
        self._observed_frames = 0

    @property
    def state(self) -> CalibrationState:
        with self._lock:
            return self._state

    @property
    def bundle(self) -> Optional[CalibrationBundle]:
        with self._lock:
            return self._bundle

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._state == CalibrationState.READY and self._bundle is not None

    @property
    def can_run(self) -> bool:
        with self._lock:
            return self._state in {CalibrationState.READY, CalibrationState.RUNNING} and self._bundle is not None

    def start(
        self,
        *,
        match_id: str,
        camera_id: str = "default",
        bundle_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> dict[str, Any]:
        with self._lock:
            if device:
                self.device = device
            self._state = CalibrationState.CALIBRATING
            self._match_id = str(match_id)
            self._camera_id = str(camera_id)
            self._bundle_path = bundle_path or str(
                Path("assets") / "calibration" / "team" / f"{match_id}.npz"
            )
            self._labels.clear()
            self._feature_bank.clear()
            self._report = None
            self._bundle = None
            self._last_frame_index = None
            self._observed_frames = 0
            return self.snapshot()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._state = CalibrationState.IDLE
            self._match_id = ""
            self._camera_id = ""
            self._bundle_path = None
            self._labels.clear()
            self._feature_bank.clear()
            self._report = None
            self._bundle = None
            self._last_frame_index = None
            self._observed_frames = 0
            return self.snapshot()

    def label_track(self, track_id: int, label: CalibrationLabel | str) -> dict[str, Any]:
        with self._lock:
            if self._state not in {
                CalibrationState.CALIBRATING,
                CalibrationState.RECALIBRATION_REQUIRED,
            }:
                raise RuntimeError("calibration session is not accepting labels")
            value = label if isinstance(label, CalibrationLabel) else CalibrationLabel(str(label))
            track_id = int(track_id)
            if value == CalibrationLabel.IGNORE:
                self._labels.pop(track_id, None)
                self._feature_bank.clear_track(track_id)
            else:
                self._labels[track_id] = value
                # A changed semantic label must not retain features sampled
                # under the previous role/team assignment.
                self._feature_bank.clear_track(track_id)
            return self.snapshot()

    def observe_frame(self, frame: np.ndarray, frame_state: Any) -> None:
        """Collect labelled, high-quality samples from one processed frame."""

        with self._lock:
            if self._state not in {
                CalibrationState.CALIBRATING,
                CalibrationState.RECALIBRATION_REQUIRED,
            }:
                return
            frame_index = int(getattr(frame_state, "frame_id", self._observed_frames))
            self._last_frame_index = frame_index
            self._observed_frames += 1
            candidates = []
            for player in getattr(frame_state, "players", []):
                label = self._labels.get(int(player.track_id))
                if label is None or label == CalibrationLabel.IGNORE:
                    continue
                current = self._feature_bank.tracks.get(int(player.track_id))
                if current is not None and current.observation_count >= self.max_samples_per_track:
                    continue
                if player.bbox is None:
                    continue
                full_crop = _crop_from_bbox(frame, player.bbox)
                roi = self.roi_extractor.extract(
                    full_crop,
                    [0.0, 0.0, float(full_crop.shape[1]), float(full_crop.shape[0])],
                )
                quality = self.quality_assessor.assess(
                    roi,
                    detection_confidence=float(player.confidence),
                )
                if not quality.accepted:
                    continue
                candidates.append((player, label, roi, quality.score))

            deep_features: list[Optional[np.ndarray]] = [None] * len(candidates)
            if candidates and self.require_appearance and self.appearance_extractor is None:
                self.appearance_extractor = AppearanceFeatureExtractor(
                    device=self.device,
                    pretrained=True,
                )
            if candidates and self.appearance_extractor is not None:
                values = self.appearance_extractor.extract_batch([item[2] for item in candidates])
                deep_features = [values[index] for index in range(len(candidates))]
            for index, (player, label, roi, quality_score) in enumerate(candidates):
                self._feature_bank.update(
                    int(player.track_id),
                    team=label.team,
                    role=label.role,
                    color_feature=self.color_extractor.extract(roi),
                    deep_feature=deep_features[index],
                    quality=quality_score,
                    frame_index=frame_index,
                )

    def validate(self) -> dict[str, Any]:
        with self._lock:
            if self._state not in {
                CalibrationState.CALIBRATING,
                CalibrationState.RECALIBRATION_REQUIRED,
            }:
                raise RuntimeError("calibration session is not ready for validation")
            self._state = CalibrationState.VALIDATING
            report = self.validator.validate(self._feature_bank.tracks.values())
            reasons = list(report.reasons)
            if self.require_appearance:
                required = [
                    track
                    for track in self._feature_bank.tracks.values()
                    if track.role == PlayerRole.OUTFIELD
                    and track.team in {TeamLabel.HOME, TeamLabel.AWAY}
                ]
                if any(track.deep_feature is None for track in required):
                    reasons.append("appearance_features_unavailable")
            if reasons:
                report = ValidationReport(**{**report.__dict__, "passed": False, "reasons": reasons})
            self._report = report
            if not report.passed:
                self._state = CalibrationState.CALIBRATING
                return self.snapshot()

            from app.classification.team_calibration.prototypes import build_prototypes

            bundle = CalibrationBundle(
                match_id=self._match_id,
                camera_id=self._camera_id,
                prototypes=build_prototypes(self._feature_bank.tracks.values()),
                validation_report=report,
            )
            if self._bundle_path:
                bundle.save(self._bundle_path)
            self._bundle = bundle
            self._state = CalibrationState.READY
            return self.snapshot()

    def mark_running(self) -> dict[str, Any]:
        with self._lock:
            if not self.ready:
                raise RuntimeError("team calibration is not READY")
            self._state = CalibrationState.RUNNING
            return self.snapshot()

    def request_recalibration(self) -> dict[str, Any]:
        with self._lock:
            self._state = CalibrationState.RECALIBRATION_REQUIRED
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            report = self._report.__dict__.copy() if self._report is not None else None
            tracks = []
            for track_id, feature in sorted(self._feature_bank.tracks.items()):
                label = self._labels.get(track_id)
                tracks.append(
                    {
                        "track_id": track_id,
                        "label": label.value if label else None,
                        "team": feature.team.value,
                        "role": feature.role.value,
                        "sample_count": feature.observation_count,
                        "quality_score": round(
                            feature.quality_sum / max(feature.observation_count, 1), 4
                        ),
                        "last_update_frame": feature.last_update_frame,
                    }
                )
            return {
                "type": "team_calibration",
                "state": self._state.value,
                "match_id": self._match_id,
                "camera_id": self._camera_id,
                "bundle_path": self._bundle_path,
                "ready": self.ready,
                "goalkeeper_mapping_ready": bool(
                    self._report.goalkeeper_mapping_ready if self._report else False
                ),
                "observed_frames": self._observed_frames,
                "last_frame_index": self._last_frame_index,
                "tracks": tracks,
                "validation_report": report,
            }


def _crop_from_bbox(frame: np.ndarray, bbox: Any) -> np.ndarray:
    image = np.asarray(frame)
    coordinates = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if image.ndim < 3 or coordinates.size < 4 or not np.isfinite(coordinates[:4]).all():
        return np.empty((0, 0, 3), dtype=np.uint8)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = coordinates[:4]
    left = max(0, min(width, int(np.floor(x1))))
    top = max(0, min(height, int(np.floor(y1))))
    right = max(0, min(width, int(np.ceil(x2))))
    bottom = max(0, min(height, int(np.ceil(y2))))
    if right <= left or bottom <= top:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return image[top:bottom, left:right, :3].copy()
