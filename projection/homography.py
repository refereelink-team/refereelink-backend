# -*- coding: utf-8 -*-
"""
Homography interfaces and quality utilities.

This module now supports two coordinate systems for homography outputs:
- map_pixel: projected points are in field_map pixel coordinates (legacy mode)
- meter_center: projected points are in metric pitch coordinates with center origin
  x in [-52.5, 52.5], y in [-34, 34]
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from projection.coords import (
    FIELD_MAP_HEIGHT,
    FIELD_MAP_WIDTH,
    HALF_PITCH_LENGTH_M,
    HALF_PITCH_WIDTH_M,
    PITCH_BOTTOM,
    PITCH_LEFT,
    PITCH_LENGTH_M,
    PITCH_RIGHT,
    PITCH_TOP,
    PITCH_WIDTH_M,
    field_meter_center_to_map_pixel,
    field_meter_center_to_top_left,
    map_pixel_to_field_meter_center,
)

COORD_SYSTEM_MAP_PIXEL = "map_pixel"
COORD_SYSTEM_METER_CENTER = "meter_center"

# field_map.png pitch bounds
PITCH_WIDTH = PITCH_RIGHT - PITCH_LEFT
PITCH_HEIGHT = PITCH_BOTTOM - PITCH_TOP

PENALTY_BOX_DEPTH_M = 16.5
PENALTY_BOX_WIDTH_M = 40.32
HALF_PENALTY_BOX_WIDTH_M = PENALTY_BOX_WIDTH_M / 2.0


def _meters_top_left_to_map(x_m: float, y_m: float) -> Tuple[float, float]:
    """Map top-left-origin pitch meters to field_map pixels."""
    x = PITCH_LEFT + (x_m / PITCH_LENGTH_M) * PITCH_WIDTH
    y = PITCH_TOP + (y_m / PITCH_WIDTH_M) * PITCH_HEIGHT
    return x, y


# Legacy target points (field_map pixels)
DEFAULT_DST_PTS = np.array(
    [
        [PITCH_LEFT, PITCH_BOTTOM],
        [PITCH_RIGHT, PITCH_BOTTOM],
        [PITCH_RIGHT, PITCH_TOP],
        [PITCH_LEFT, PITCH_TOP],
    ],
    dtype=np.float32,
)

# Metric target points (center-origin meters)
DEFAULT_DST_PTS_METER_CENTER = np.array(
    [
        [-HALF_PITCH_LENGTH_M, HALF_PITCH_WIDTH_M],   # bottom-left
        [HALF_PITCH_LENGTH_M, HALF_PITCH_WIDTH_M],    # bottom-right
        [HALF_PITCH_LENGTH_M, -HALF_PITCH_WIDTH_M],   # top-right
        [-HALF_PITCH_LENGTH_M, -HALF_PITCH_WIDTH_M],  # top-left
    ],
    dtype=np.float32,
)

# Default source points for a generic 1280x720 frame
DEFAULT_SRC_PTS = np.array(
    [
        [320, 540],
        [960, 540],
        [960, 180],
        [320, 180],
    ],
    dtype=np.float32,
)


PITCH_KEYPOINT_TEMPLATE_TOP_LEFT: Dict[str, Tuple[float, float]] = {
    "top_left_corner": (0.0, 0.0),
    "top_right_corner": (PITCH_LENGTH_M, 0.0),
    "bottom_right_corner": (PITCH_LENGTH_M, PITCH_WIDTH_M),
    "bottom_left_corner": (0.0, PITCH_WIDTH_M),
    "penalty_top_left": (
        PENALTY_BOX_DEPTH_M,
        (PITCH_WIDTH_M / 2.0) - HALF_PENALTY_BOX_WIDTH_M,
    ),
    "penalty_top_right": (
        PITCH_LENGTH_M - PENALTY_BOX_DEPTH_M,
        (PITCH_WIDTH_M / 2.0) - HALF_PENALTY_BOX_WIDTH_M,
    ),
    "penalty_bottom_right": (
        PITCH_LENGTH_M - PENALTY_BOX_DEPTH_M,
        (PITCH_WIDTH_M / 2.0) + HALF_PENALTY_BOX_WIDTH_M,
    ),
    "penalty_bottom_left": (
        PENALTY_BOX_DEPTH_M,
        (PITCH_WIDTH_M / 2.0) + HALF_PENALTY_BOX_WIDTH_M,
    ),
    "mid_top": (PITCH_LENGTH_M / 2.0, 0.0),
    "mid_bottom": (PITCH_LENGTH_M / 2.0, PITCH_WIDTH_M),
    "mid_left": (0.0, PITCH_WIDTH_M / 2.0),
    "mid_right": (PITCH_LENGTH_M, PITCH_WIDTH_M / 2.0),
}


def template_to_dst_points(keypoint_id: str) -> Optional[np.ndarray]:
    if keypoint_id not in PITCH_KEYPOINT_TEMPLATE_TOP_LEFT:
        return None
    x_m, y_m = PITCH_KEYPOINT_TEMPLATE_TOP_LEFT[keypoint_id]
    x, y = _meters_top_left_to_map(x_m, y_m)
    return np.array([x, y], dtype=np.float32)


def template_to_image_points() -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for keypoint_id in PITCH_KEYPOINT_TEMPLATE_TOP_LEFT:
        pt = template_to_dst_points(keypoint_id)
        if pt is not None:
            result[keypoint_id] = pt
    return result


def template_to_field_points_center() -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for keypoint_id, (x_tl, y_tl) in PITCH_KEYPOINT_TEMPLATE_TOP_LEFT.items():
        x_c = x_tl - HALF_PITCH_LENGTH_M
        y_c = y_tl - HALF_PITCH_WIDTH_M
        result[keypoint_id] = np.array([x_c, y_c], dtype=np.float32)
    return result


class HomographyAdapter:
    """Homography adapter with explicit output coordinate semantics."""

    def __init__(
        self,
        src_pts: Optional[np.ndarray] = None,
        dst_pts: Optional[np.ndarray] = None,
        H_matrix: Optional[np.ndarray] = None,
        coord_system: str = COORD_SYSTEM_MAP_PIXEL,
    ):
        self.coord_system = coord_system
        if H_matrix is not None:
            self.H = np.array(H_matrix, dtype=np.float32)
        elif src_pts is not None and dst_pts is not None:
            src_pts = np.array(src_pts, dtype=np.float32)
            dst_pts = np.array(dst_pts, dtype=np.float32)
            H, _ = cv2.findHomography(src_pts, dst_pts)
            if H is None:
                raise ValueError("Failed to estimate homography from points.")
            self.H = H.astype(np.float32)
        else:
            raise ValueError("Provide either (src_pts, dst_pts) or H_matrix.")

    def pixel_to_map(self, u: float, v: float) -> Tuple[float, float]:
        """Project pixel to adapter output coordinate system."""
        pt = np.array([[[u, v]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.H)[0][0]
        return float(mapped[0]), float(mapped[1])

    def pixel_points_to_map(self, points: np.ndarray) -> np.ndarray:
        """Batch project pixels to adapter output coordinate system."""
        if points.ndim == 1:
            points = points.reshape(1, -1)
        pts = points.astype(np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.H)
        return out.reshape(-1, 2)

    def pixel_to_field_meters(self, u: float, v: float) -> Tuple[float, float]:
        """Project pixel to center-origin field meters."""
        x, y = self.pixel_to_map(u, v)
        if self.coord_system == COORD_SYSTEM_METER_CENTER:
            return x, y
        return map_pixel_to_field_meter_center(x, y)

    def pixel_points_to_field_meters(self, points: np.ndarray) -> np.ndarray:
        """Batch project pixels to center-origin field meters."""
        mapped = self.pixel_points_to_map(points)
        if self.coord_system == COORD_SYSTEM_METER_CENTER:
            return mapped
        converted = np.zeros_like(mapped, dtype=np.float32)
        for i, (x_px, y_px) in enumerate(mapped):
            x_m, y_m = map_pixel_to_field_meter_center(float(x_px), float(y_px))
            converted[i, 0] = x_m
            converted[i, 1] = y_m
        return converted

    def field_meters_to_map_pixel(self, x_m: float, y_m: float) -> Tuple[float, float]:
        return field_meter_center_to_map_pixel(x_m, y_m)

    def set_homography(self, H: np.ndarray) -> None:
        self.H = np.array(H, dtype=np.float32)

    def get_matrix(self) -> np.ndarray:
        return self.H.copy()


@dataclass
class HomographyState:
    H: np.ndarray
    inliers: int = 0
    inlier_ratio: float = 0.0
    reproj_err_px_med: float = 0.0
    reproj_err_px_p90: float = 0.0
    fails: int = 0
    mode: str = "TRACK"
    last_good_ts: float = 0.0

    @staticmethod
    def default() -> "HomographyState":
        H = build_default_homography().get_matrix()
        return HomographyState(H=H, mode="TRACK")


@dataclass
class HomographyQuality:
    inliers: int
    inlier_ratio: float
    reproj_err_px_med: float
    reproj_err_px_p90: float


class HomographyEstimator:
    """RANSAC-based homography estimator with quality metrics."""

    def __init__(
        self,
        ransac_reproj_threshold: float = 5.0,
        max_iters: int = 1000,
        confidence: float = 0.995,
    ):
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.max_iters = max_iters
        self.confidence = confidence

    def estimate(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[HomographyQuality]]:
        if len(src_pts) < 4 or len(dst_pts) < 4:
            return None, None, None

        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)

        try:
            H, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                self.ransac_reproj_threshold,
                maxIters=self.max_iters,
                confidence=self.confidence,
            )
        except Exception:
            return None, None, None

        if H is None:
            return None, None, None

        quality = self._evaluate_quality(src_pts, dst_pts, H, mask)
        return H, mask, quality

    def _evaluate_quality(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        H: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> HomographyQuality:
        if mask is not None:
            inliers = mask.flatten() == 1
            inlier_count = int(np.sum(inliers))
            inlier_ratio = inlier_count / len(src_pts) if len(src_pts) > 0 else 0.0
        else:
            inliers = np.ones(len(src_pts), dtype=bool)
            inlier_count = len(src_pts)
            inlier_ratio = 1.0

        src_pts_np = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(src_pts_np, H)
        errors = np.linalg.norm(projected.reshape(-1, 2) - dst_pts, axis=1)
        inlier_errors = errors[inliers] if np.any(inliers) else errors
        reproj_err_med = float(np.median(inlier_errors)) if len(inlier_errors) > 0 else 0.0
        reproj_err_p90 = float(np.percentile(inlier_errors, 90)) if len(inlier_errors) > 0 else 0.0

        return HomographyQuality(
            inliers=inlier_count,
            inlier_ratio=inlier_ratio,
            reproj_err_px_med=reproj_err_med,
            reproj_err_px_p90=reproj_err_p90,
        )


class HomographyQualityGate:
    def __init__(
        self,
        min_inliers: int = 6,
        min_inlier_ratio: float = 0.4,
        max_reproj_err_px: float = 10.0,
    ):
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.max_reproj_err_px = max_reproj_err_px

    def accept(self, quality: Optional[HomographyQuality]) -> bool:
        if quality is None:
            return False
        if quality.inliers < self.min_inliers:
            return False
        if quality.inlier_ratio < self.min_inlier_ratio:
            return False
        if quality.reproj_err_px_p90 > self.max_reproj_err_px:
            return False
        return True

    def get_rejection_reason(self, quality: Optional[HomographyQuality]) -> str:
        if quality is None:
            return "quality is None"
        if quality.inliers < self.min_inliers:
            return f"inliers {quality.inliers} < {self.min_inliers}"
        if quality.inlier_ratio < self.min_inlier_ratio:
            return f"inlier_ratio {quality.inlier_ratio:.2f} < {self.min_inlier_ratio}"
        if quality.reproj_err_px_p90 > self.max_reproj_err_px:
            return f"reproj_err_p90 {quality.reproj_err_px_p90:.1f} > {self.max_reproj_err_px}"
        return "accepted"


def load_calibration(calibration_path: str) -> HomographyAdapter:
    with open(calibration_path, encoding="utf-8") as f:
        data = json.load(f)
    coord_system = data.get("coord_system", COORD_SYSTEM_MAP_PIXEL)
    return HomographyAdapter(H_matrix=np.array(data["H"]), coord_system=coord_system)


def build_default_homography() -> HomographyAdapter:
    """Legacy default homography (pixel -> map pixel)."""
    return HomographyAdapter(
        src_pts=DEFAULT_SRC_PTS,
        dst_pts=DEFAULT_DST_PTS,
        coord_system=COORD_SYSTEM_MAP_PIXEL,
    )


def build_default_meter_homography() -> HomographyAdapter:
    """Default homography in metric center-origin space."""
    return HomographyAdapter(
        src_pts=DEFAULT_SRC_PTS,
        dst_pts=DEFAULT_DST_PTS_METER_CENTER,
        coord_system=COORD_SYSTEM_METER_CENTER,
    )


class HomographySmoother:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.H_prev: Optional[np.ndarray] = None

    def update(self, H_new: np.ndarray) -> np.ndarray:
        H_new = np.array(H_new, dtype=np.float32)
        if self.H_prev is None:
            self.H_prev = H_new
            return H_new.copy()

        H_smooth = self.alpha * H_new + (1.0 - self.alpha) * self.H_prev
        self.H_prev = H_smooth
        return H_smooth

    def reset(self) -> None:
        self.H_prev = None

