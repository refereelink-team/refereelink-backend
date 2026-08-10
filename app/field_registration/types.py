"""Stable contracts shared by pitch perception, tracking, and projection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional, Tuple

import numpy as np


class CameraTrackingStatus(str, Enum):
    INITIALIZING = "initializing"
    RELOCALIZED = "relocalized"
    CORRECTED = "corrected"
    TRACKED = "tracked"
    PREDICTED = "predicted"
    LOST = "lost"


@dataclass(frozen=True)
class PointObservation:
    """A semantic correspondence between an image point and the pitch plane."""

    label: str
    image_xy: Tuple[float, float]
    pitch_xy_m: Tuple[float, float]
    confidence: float = 1.0
    sigma_px: float = 1.0
    source: str = "model"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("point confidence must be between 0 and 1")
        if self.sigma_px <= 0.0:
            raise ValueError("point sigma_px must be positive")
        values = (*self.image_xy, *self.pitch_xy_m)
        if not np.all(np.isfinite(values)):
            raise ValueError("point coordinates must be finite")


@dataclass(frozen=True)
class LineObservation:
    """Sampled pixels belonging to one known pitch line or arc."""

    label: str
    image_points: np.ndarray
    pitch_points_m: np.ndarray
    confidence: float = 1.0
    source: str = "segment"

    def __post_init__(self) -> None:
        image = np.asarray(self.image_points, dtype=np.float64)
        pitch = np.asarray(self.pitch_points_m, dtype=np.float64)
        if image.ndim != 2 or image.shape[1] != 2:
            raise ValueError("image_points must have shape (N, 2)")
        if pitch.ndim != 2 or pitch.shape[1] != 2:
            raise ValueError("pitch_points_m must have shape (M, 2)")
        if image.shape[0] < 2:
            raise ValueError("a line observation needs at least two samples")
        if pitch.shape[0] < 2:
            raise ValueError("a pitch line needs at least two samples")
        if not np.all(np.isfinite(image)) or not np.all(np.isfinite(pitch)):
            raise ValueError("line coordinates must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("line confidence must be between 0 and 1")
        object.__setattr__(self, "image_points", image)
        object.__setattr__(self, "pitch_points_m", pitch)


@dataclass(frozen=True)
class CameraState:
    status: CameraTrackingStatus
    pan_rad: float
    pan_velocity_rad_s: float
    covariance: np.ndarray
    image_to_pitch: Optional[np.ndarray]
    pitch_to_image: Optional[np.ndarray]
    point_inliers: int = 0
    visible_segment_count: int = 0
    spatial_coverage: float = 0.0
    mean_segment_error_px: Optional[float] = None
    p95_segment_error_px: Optional[float] = None
    confidence: float = 0.0
    age_since_semantic_update: int = 0

    def __post_init__(self) -> None:
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (2, 2):
            raise ValueError("camera-state covariance must have shape (2, 2)")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("camera-state covariance must be finite")
        if not 0.0 <= self.spatial_coverage <= 1.0:
            raise ValueError("spatial_coverage must be between 0 and 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("camera-state confidence must be between 0 and 1")
        for name in ("image_to_pitch", "pitch_to_image"):
            value = getattr(self, name)
            if value is not None:
                matrix = np.asarray(value, dtype=np.float64)
                if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                    raise ValueError(f"{name} must be a finite 3x3 matrix")
                object.__setattr__(self, name, matrix)
        object.__setattr__(self, "covariance", covariance)

    @property
    def usable_for_measurement(self) -> bool:
        return (
            self.status
            in {
                CameraTrackingStatus.RELOCALIZED,
                CameraTrackingStatus.CORRECTED,
                CameraTrackingStatus.TRACKED,
            }
            and self.image_to_pitch is not None
            and self.confidence > 0.0
        )


@dataclass(frozen=True)
class PitchCoordinate:
    xy_m: Optional[Tuple[float, float]]
    sigma_m: Optional[float]
    source: Literal["contact_head", "ankles", "bbox_bottom", "none"]
    camera_status: CameraTrackingStatus
    covariance_m2: Optional[np.ndarray] = None


@dataclass(frozen=True)
class FieldRegistrationFrame:
    frame_index: int
    camera_state: CameraState
    point_observations: Tuple[PointObservation, ...] = field(default_factory=tuple)
    line_observations: Tuple[LineObservation, ...] = field(default_factory=tuple)
    refresh_reason: Optional[str] = None
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)
