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
    SOURCE_PREVIEW = "source_preview"
    CLIP_SELECTING = "clip_selecting"
    PROCESSING = "processing"
    REVIEW = "review"
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
        # Manually labelled goalkeepers/referees are often farther away or
        # surrounded by grass than outfield players. Keep structural checks,
        # but avoid losing every sample before a role prototype can be built.
        self.role_quality_assessor = CropQualityAssessor(
            min_width=8,
            min_height=12,
            min_detection_confidence=0.3,
            min_blur_score=10.0,
            min_quality_score=0.25,
            max_green_edge_ratio=1.0,
        )
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
        self._source_url: Optional[str] = None
        self._clip_id: Optional[str] = None
        self._clip_start_ms: Optional[int] = None
        self._clip_end_ms: Optional[int] = None
        self._clip_duration_ms: Optional[int] = None
        self._review_video_url: Optional[str] = None
        self._metadata_url: Optional[str] = None
        self._job_id: Optional[str] = None
        self._processing_progress = 0.0
        self._processing_error: Optional[str] = None
        self._clip_tracks: list[dict[str, Any]] = []

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
            self._source_url = None
            self._clip_id = None
            self._clip_start_ms = None
            self._clip_end_ms = None
            self._clip_duration_ms = None
            self._review_video_url = None
            self._metadata_url = None
            self._job_id = None
            self._processing_progress = 0.0
            self._processing_error = None
            self._clip_tracks = []
            return self.snapshot()

    def begin_source_preview(
        self,
        *,
        match_id: str,
        camera_id: str,
        source_url: str,
        bundle_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> dict[str, Any]:
        """Start a clip-based session without starting the live pipeline."""

        self.start(
            match_id=match_id,
            camera_id=camera_id,
            bundle_path=bundle_path,
            device=device,
        )
        with self._lock:
            self._source_url = source_url
            self._state = CalibrationState.SOURCE_PREVIEW
            return self.snapshot()

    def begin_clip_selecting(self, start_ms: int) -> dict[str, Any]:
        with self._lock:
            if self._state not in {CalibrationState.SOURCE_PREVIEW, CalibrationState.CLIP_SELECTING}:
                raise RuntimeError("source preview is required before selecting a clip")
            self._clip_start_ms = int(start_ms)
            self._clip_end_ms = None
            self._clip_duration_ms = None
            self._state = CalibrationState.CLIP_SELECTING
            return self.snapshot()

    def begin_processing(
        self,
        *,
        clip_id: str,
        job_id: str,
        start_ms: int,
        end_ms: int,
        duration_ms: int,
        review_video_url: str,
        metadata_url: str,
    ) -> dict[str, Any]:
        with self._lock:
            if self._state != CalibrationState.CLIP_SELECTING:
                raise RuntimeError("clip start time has not been selected")
            self._clip_id = str(clip_id)
            self._job_id = str(job_id)
            self._clip_start_ms = int(start_ms)
            self._clip_end_ms = int(end_ms)
            self._clip_duration_ms = int(duration_ms)
            self._review_video_url = review_video_url
            self._metadata_url = metadata_url
            self._processing_progress = 0.0
            self._processing_error = None
            self._clip_tracks = []
            self._observed_frames = 0
            self._last_frame_index = None
            self._state = CalibrationState.PROCESSING
            return self.snapshot()

    def update_processing(self, *, progress: float, observed_frames: Optional[int] = None) -> dict[str, Any]:
        with self._lock:
            self._processing_progress = float(np.clip(progress, 0.0, 1.0))
            if observed_frames is not None:
                self._observed_frames = int(observed_frames)
            return self.snapshot()

    def complete_processing(
        self,
        *,
        tracks: list[dict[str, Any]],
        observed_frames: int,
        last_frame_index: Optional[int],
    ) -> dict[str, Any]:
        with self._lock:
            if self._state != CalibrationState.PROCESSING:
                raise RuntimeError("calibration clip is not processing")
            self._clip_tracks = list(tracks)
            self._observed_frames = int(observed_frames)
            self._last_frame_index = last_frame_index
            self._processing_progress = 1.0
            self._state = CalibrationState.REVIEW
            return self.snapshot()

    def fail_processing(self, error: str) -> dict[str, Any]:
        with self._lock:
            self._processing_error = str(error)
            self._state = CalibrationState.CLIP_SELECTING
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
            self._source_url = None
            self._clip_id = None
            self._clip_start_ms = None
            self._clip_end_ms = None
            self._clip_duration_ms = None
            self._review_video_url = None
            self._metadata_url = None
            self._job_id = None
            self._processing_progress = 0.0
            self._processing_error = None
            self._clip_tracks = []
            return self.snapshot()

    def label_track(
        self,
        track_id: int,
        label: CalibrationLabel | str,
        *,
        samples: Optional[list[tuple[np.ndarray, float, int]]] = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._state not in {
                CalibrationState.CALIBRATING,
                CalibrationState.REVIEW,
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
                if samples:
                    self._add_samples_locked(track_id, value, samples)
            return self.snapshot()

    def _add_samples_locked(
        self,
        track_id: int,
        label: CalibrationLabel,
        samples: list[tuple[np.ndarray, float, int]],
    ) -> None:
        limited = samples[: self.max_samples_per_track]
        if self.require_appearance and self.appearance_extractor is None:
            self.appearance_extractor = AppearanceFeatureExtractor(
                device=self.device,
                pretrained=True,
            )
        crops: list[np.ndarray] = []
        prepared: list[tuple[float, int, np.ndarray]] = []
        assessor = (
            self.role_quality_assessor
            if label in {
                CalibrationLabel.HOME_GOALKEEPER,
                CalibrationLabel.AWAY_GOALKEEPER,
                CalibrationLabel.REFEREE,
            }
            else self.quality_assessor
        )
        for roi, _quality_hint, frame_index in limited:
            quality = assessor.assess(roi, detection_confidence=1.0)
            if not quality.accepted:
                continue
            crops.append(roi)
            prepared.append((quality.score, int(frame_index), roi))
        deep_features: list[Optional[np.ndarray]] = [None] * len(prepared)
        if prepared and self.appearance_extractor is not None:
            values = self.appearance_extractor.extract_batch(crops)
            deep_features = [values[index] for index in range(len(prepared))]
        for index, (quality_score, frame_index, roi) in enumerate(prepared):
            self._feature_bank.update(
                track_id,
                team=label.team,
                role=label.role,
                color_feature=self.color_extractor.extract(roi),
                deep_feature=deep_features[index],
                quality=quality_score,
                frame_index=frame_index,
            )

    def observe_frame(self, frame: np.ndarray, frame_state: Any) -> None:
        """Collect labelled, high-quality samples from one processed frame."""

        with self._lock:
            if self._state not in {
                CalibrationState.CALIBRATING,
                CalibrationState.REVIEW,
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
                CalibrationState.REVIEW,
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
                self._state = (
                    CalibrationState.REVIEW
                    if self._clip_id is not None
                    else CalibrationState.CALIBRATING
                )
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

    def set_clip_tracks(self, tracks: list[dict[str, Any]]) -> None:
        with self._lock:
            self._clip_tracks = list(tracks)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            report = self._report.__dict__.copy() if self._report is not None else None
            tracks = []
            feature_by_id = self._feature_bank.tracks
            for clip_track in self._clip_tracks:
                track_id = int(clip_track["track_id"])
                feature = feature_by_id.get(track_id)
                label = self._labels.get(track_id)
                tracks.append(
                    {
                        **clip_track,
                        "label": label.value if label else None,
                        "team": feature.team.value if feature else TeamLabel.UNKNOWN.value,
                        "role": feature.role.value if feature else PlayerRole.UNKNOWN.value,
                        "sample_count": feature.observation_count if feature else 0,
                        "quality_score": round(
                            feature.quality_sum / max(feature.observation_count, 1), 4
                        ) if feature else 0.0,
                        "last_update_frame": feature.last_update_frame if feature else None,
                    }
                )
            clip_track_ids = {int(track["track_id"]) for track in tracks}
            for track_id, feature in sorted(feature_by_id.items()):
                if track_id in clip_track_ids:
                    continue
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
                "referee_mapping_ready": bool(
                    self._report.referee_mapping_ready if self._report else False
                ),
                "observed_frames": self._observed_frames,
                "last_frame_index": self._last_frame_index,
                "tracks": tracks,
                "validation_report": report,
                "source_url": self._source_url,
                "clip_id": self._clip_id,
                "clip_start_ms": self._clip_start_ms,
                "clip_end_ms": self._clip_end_ms,
                "clip_duration_ms": self._clip_duration_ms,
                "review_video_url": self._review_video_url,
                "metadata_url": self._metadata_url,
                "job_id": self._job_id,
                "processing_progress": self._processing_progress,
                "processing_error": self._processing_error,
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
