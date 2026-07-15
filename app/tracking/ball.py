from collections import deque
from dataclasses import dataclass
import time
from typing import Optional

import cv2
import numpy as np
import supervision as sv


@dataclass(frozen=True)
class BallTrackEstimate:
    """Kinematic estimate for the current ball position.

    Coordinates are image-space pixels.  Field-space projection is deliberately
    kept outside this class because it depends on the current homography.
    """

    position: Optional[np.ndarray]
    velocity: Optional[np.ndarray]
    confidence: float
    status: str
    age_frames: int


class BallAnnotator:
    """
    A class to annotate frames with circles of varying radii and colors.

    Attributes:
        radius (int): The maximum radius of the circles to be drawn.
        buffer (deque): A deque buffer to store recent coordinates for annotation.
        color_palette (sv.ColorPalette): A color palette for the circles.
        thickness (int): The thickness of the circle borders.
    """

    def __init__(self, radius: int, buffer_size: int = 5, thickness: int = 2):

        self.color_palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness

    def interpolate_radius(self, i: int, max_i: int) -> int:
        """
        Interpolates the radius between 1 and the maximum radius based on the index.

        Args:
            i (int): The current index in the buffer.
            max_i (int): The maximum index in the buffer.

        Returns:
            int: The interpolated radius.
        """
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Annotates the frame with circles based on detections.

        Args:
            frame (np.ndarray): The frame to annotate.
            detections (sv.Detections): The detections containing coordinates.

        Returns:
            np.ndarray: The annotated frame.
        """
        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int)
        self.buffer.append(xy)
        for i, xy in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))
            for center in xy:
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=interpolated_radius,
                    color=color.as_bgr(),
                    thickness=self.thickness
                )
        return frame


class BallTracker:
    """
    A class used to track a soccer ball's position across video frames.

    The BallTracker class maintains a buffer of recent ball positions and uses this
    buffer to predict the ball's position in the current frame by selecting the
    detection closest to the average position (centroid) of the recent positions.

    Attributes:
        buffer (collections.deque): A deque buffer to store recent ball positions.
    """
    def __init__(self, buffer_size: int = 10):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates the buffer with new detections and returns the detection closest to the
        centroid of recent positions.

        Args:
            detections (sv.Detections): The current frame's ball detections.

        Returns:
            sv.Detections: The detection closest to the centroid of recent positions.
            If there are no detections, returns the input detections.
        """
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)

        if len(detections) == 0:
            return detections

        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = np.argmin(distances)
        return detections[[index]]


class KinematicBallTracker:
    """Small constant-velocity tracker for a single football.

    The detector is allowed to run at a lower frequency than the video.  When
    a detection is missing, this tracker predicts only for a bounded number of
    frames and then returns ``unavailable`` rather than manufacturing a stale
    ball coordinate.
    """

    def __init__(
        self,
        max_prediction_frames: int = 8,
        velocity_smoothing: float = 0.65,
        max_dt_seconds: float = 0.25,
    ) -> None:
        self.max_prediction_frames = max(int(max_prediction_frames), 0)
        self.velocity_smoothing = float(np.clip(velocity_smoothing, 0.0, 1.0))
        self.max_dt_seconds = max(float(max_dt_seconds), 1e-3)
        self._position: Optional[np.ndarray] = None
        self._velocity = np.zeros(2, dtype=np.float32)
        self._last_timestamp: Optional[float] = None
        self._age_frames = 0
        self._confidence = 0.0

    @property
    def age_frames(self) -> int:
        return self._age_frames

    def reset(self) -> None:
        self._position = None
        self._velocity = np.zeros(2, dtype=np.float32)
        self._last_timestamp = None
        self._age_frames = 0
        self._confidence = 0.0

    def update(
        self,
        position: Optional[np.ndarray],
        confidence: float = 0.0,
        timestamp_s: Optional[float] = None,
    ) -> BallTrackEstimate:
        now = float(timestamp_s) if timestamp_s is not None else time.monotonic()
        if self._last_timestamp is None:
            dt = 1.0 / 25.0
        else:
            dt = float(np.clip(now - self._last_timestamp, 1e-3, self.max_dt_seconds))
        self._last_timestamp = now

        observed = None
        if position is not None:
            candidate = np.asarray(position, dtype=np.float32).reshape(-1)
            if candidate.size >= 2 and np.isfinite(candidate[:2]).all():
                observed = candidate[:2].copy()

        if observed is not None:
            if self._position is not None:
                measured_velocity = (observed - self._position) / dt
                self._velocity = (
                    (1.0 - self.velocity_smoothing) * self._velocity
                    + self.velocity_smoothing * measured_velocity
                )
            else:
                self._velocity.fill(0.0)
            self._position = observed
            self._age_frames = 0
            self._confidence = float(np.clip(confidence, 0.0, 1.0))
            return BallTrackEstimate(
                position=self._position.copy(),
                velocity=self._velocity.copy(),
                confidence=self._confidence,
                status="fresh",
                age_frames=0,
            )

        if self._position is None or self._age_frames >= self.max_prediction_frames:
            self._age_frames += 1
            return BallTrackEstimate(
                position=None,
                velocity=None,
                confidence=0.0,
                status="unavailable",
                age_frames=self._age_frames,
            )

        self._position = self._position + self._velocity * dt
        self._age_frames += 1
        confidence = self._confidence * max(
            0.0, 1.0 - self._age_frames / max(self.max_prediction_frames, 1)
        )
        return BallTrackEstimate(
            position=self._position.copy(),
            velocity=self._velocity.copy(),
            confidence=float(confidence),
            status="predicted",
            age_frames=self._age_frames,
        )
