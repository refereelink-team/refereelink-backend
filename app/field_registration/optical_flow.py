"""Masked sparse LK flow for propagating field geometry between keyframes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from app.field_registration.geometry import transform_points
from app.field_registration.pitch_model import PitchModel


@dataclass(frozen=True)
class OpticalFlowConfig:
    maximum_corners: int = 300
    quality_level: float = 0.01
    minimum_distance_px: float = 8.0
    block_size: int = 7
    pyramid_levels: int = 3
    window_size_px: int = 21
    forward_backward_threshold_px: float = 1.5
    minimum_tracks: int = 12
    excluded_top_ratio: float = 0.12
    dynamic_box_expansion_ratio: float = 0.12


@dataclass(frozen=True)
class OpticalFlowResult:
    previous_xy: np.ndarray
    current_xy: np.ndarray
    forward_backward_error_px: np.ndarray
    valid_ratio: float

    @property
    def count(self) -> int:
        return int(self.previous_xy.shape[0])

    @property
    def mean_error_px(self) -> Optional[float]:
        if self.forward_backward_error_px.size == 0:
            return None
        return float(np.mean(self.forward_backward_error_px))


class MaskedSparseOpticalFlow:
    def __init__(self, config: OpticalFlowConfig | None = None) -> None:
        self.config = config or OpticalFlowConfig()

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3 and frame.shape[2] >= 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raise ValueError("frame must be grayscale or BGR")

    def build_static_mask(
        self,
        frame_shape: Sequence[int],
        dynamic_boxes_xyxy: Optional[np.ndarray] = None,
        field_polygon_xy: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        mask = np.full((height, width), 255, dtype=np.uint8)
        mask[: int(round(height * self.config.excluded_top_ratio))] = 0
        if field_polygon_xy is not None:
            polygon = np.asarray(field_polygon_xy, dtype=np.int32)
            polygon_mask = np.zeros_like(mask)
            if polygon.ndim == 2 and polygon.shape[0] >= 3 and polygon.shape[1] == 2:
                cv2.fillPoly(polygon_mask, [polygon], 255)
                mask = cv2.bitwise_and(mask, polygon_mask)
        if dynamic_boxes_xyxy is not None:
            for box in np.asarray(dynamic_boxes_xyxy, dtype=np.float64).reshape(-1, 4):
                x_min, y_min, x_max, y_max = box
                expand_x = (x_max - x_min) * self.config.dynamic_box_expansion_ratio
                expand_y = (y_max - y_min) * self.config.dynamic_box_expansion_ratio
                left = int(np.clip(np.floor(x_min - expand_x), 0, width))
                top = int(np.clip(np.floor(y_min - expand_y), 0, height))
                right = int(np.clip(np.ceil(x_max + expand_x), 0, width))
                bottom = int(np.clip(np.ceil(y_max + expand_y), 0, height))
                mask[top:bottom, left:right] = 0
        return mask

    def track(
        self,
        previous_frame: np.ndarray,
        current_frame: np.ndarray,
        dynamic_boxes_xyxy: Optional[np.ndarray] = None,
        field_polygon_xy: Optional[np.ndarray] = None,
    ) -> OpticalFlowResult:
        previous_gray = self._gray(previous_frame)
        current_gray = self._gray(current_frame)
        if previous_gray.shape != current_gray.shape:
            raise ValueError("optical-flow frames must have the same shape")
        mask = self.build_static_mask(
            previous_gray.shape,
            dynamic_boxes_xyxy=dynamic_boxes_xyxy,
            field_polygon_xy=field_polygon_xy,
        )
        previous_points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=self.config.maximum_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.minimum_distance_px,
            mask=mask,
            blockSize=self.config.block_size,
        )
        if previous_points is None or len(previous_points) == 0:
            return OpticalFlowResult(
                np.empty((0, 2)), np.empty((0, 2)), np.empty(0), 0.0
            )
        window = (self.config.window_size_px, self.config.window_size_px)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        )
        current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            previous_points,
            None,
            winSize=window,
            maxLevel=self.config.pyramid_levels,
            criteria=criteria,
        )
        if current_points is None or forward_status is None:
            return OpticalFlowResult(
                np.empty((0, 2)), np.empty((0, 2)), np.empty(0), 0.0
            )
        backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            previous_gray,
            current_points,
            None,
            winSize=window,
            maxLevel=self.config.pyramid_levels,
            criteria=criteria,
        )
        if backward_points is None or backward_status is None:
            return OpticalFlowResult(
                np.empty((0, 2)), np.empty((0, 2)), np.empty(0), 0.0
            )
        previous_xy = previous_points.reshape(-1, 2)
        current_xy = current_points.reshape(-1, 2)
        backward_xy = backward_points.reshape(-1, 2)
        errors = np.linalg.norm(backward_xy - previous_xy, axis=1)
        valid = (
            forward_status.reshape(-1).astype(bool)
            & backward_status.reshape(-1).astype(bool)
            & np.all(np.isfinite(current_xy), axis=1)
            & (errors <= self.config.forward_backward_threshold_px)
        )
        return OpticalFlowResult(
            previous_xy=previous_xy[valid].astype(np.float64),
            current_xy=current_xy[valid].astype(np.float64),
            forward_backward_error_px=errors[valid].astype(np.float64),
            valid_ratio=float(np.count_nonzero(valid) / max(len(valid), 1)),
        )

    @staticmethod
    def field_correspondences(
        flow: OpticalFlowResult,
        previous_image_to_pitch: np.ndarray,
        pitch_model: PitchModel,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if flow.count == 0:
            return np.empty((0, 2)), np.empty((0, 2))
        pitch_xy = transform_points(flow.previous_xy, previous_image_to_pitch)
        valid = np.array(
            [
                np.all(np.isfinite(point))
                and pitch_model.contains((float(point[0]), float(point[1])), margin_m=0.25)
                for point in pitch_xy
            ],
            dtype=bool,
        )
        return pitch_xy[valid], flow.current_xy[valid]
