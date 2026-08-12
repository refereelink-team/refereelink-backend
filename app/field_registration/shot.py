"""Lightweight hard-cut and field-view diagnostics for broadcast video."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np


class FieldView(str, Enum):
    WIDE_FIELD = "wide_field"
    PARTIAL_FIELD = "partial_field"
    NON_FIELD = "non_field"


@dataclass(frozen=True)
class ShotBoundaryConfig:
    histogram_threshold: float = 0.62
    soft_histogram_threshold: float = 0.35
    maximum_feature_inlier_ratio: float = 0.12
    minimum_feature_matches: int = 12
    resize_width: int = 320
    minimum_green_ratio: float = 0.035
    wide_field_green_ratio: float = 0.18


@dataclass(frozen=True)
class ShotDecision:
    is_cut: bool
    score: float
    histogram_distance: float
    feature_inlier_ratio: Optional[float]
    field_view: FieldView
    green_ratio: float


class ShotBoundaryDetector:
    """Detect hard broadcast cuts without introducing another neural model.

    A histogram jump is sufficient for an unambiguous cut. Borderline jumps
    additionally require the two frames to have almost no geometrically
    consistent ORB correspondences. The detector intentionally does not try to
    classify dissolves; confidence loss in the tracker handles those safely.
    """

    def __init__(self, config: ShotBoundaryConfig | None = None) -> None:
        self.config = config or ShotBoundaryConfig()
        self._previous_histogram: Optional[np.ndarray] = None
        self._previous_gray: Optional[np.ndarray] = None
        self._orb = cv2.ORB_create(nfeatures=450, fastThreshold=12)

    def reset(self) -> None:
        self._previous_histogram = None
        self._previous_gray = None

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= self.config.resize_width:
            return frame
        scale = self.config.resize_width / float(width)
        return cv2.resize(
            frame,
            (self.config.resize_width, max(int(round(height * scale)), 1)),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _histogram(frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
        return cv2.normalize(histogram, None, norm_type=cv2.NORM_L1).reshape(-1)

    @staticmethod
    def _green_ratio(frame: np.ndarray) -> float:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = (
            (hsv[..., 0] >= 28)
            & (hsv[..., 0] <= 95)
            & (hsv[..., 1] >= 35)
            & (hsv[..., 2] >= 25)
        )
        return float(np.mean(green))

    def _field_view(self, green_ratio: float) -> FieldView:
        if green_ratio >= self.config.wide_field_green_ratio:
            return FieldView.WIDE_FIELD
        if green_ratio >= self.config.minimum_green_ratio:
            return FieldView.PARTIAL_FIELD
        return FieldView.NON_FIELD

    def _feature_inlier_ratio(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
    ) -> Optional[float]:
        previous_keypoints, previous_descriptors = self._orb.detectAndCompute(
            previous_gray, None
        )
        current_keypoints, current_descriptors = self._orb.detectAndCompute(
            current_gray, None
        )
        if previous_descriptors is None or current_descriptors is None:
            return None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(previous_descriptors, current_descriptors, k=2)
        matches = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < 0.75 * second.distance
        ]
        if len(matches) < self.config.minimum_feature_matches:
            return 0.0
        previous_xy = np.float32(
            [previous_keypoints[match.queryIdx].pt for match in matches]
        )
        current_xy = np.float32(
            [current_keypoints[match.trainIdx].pt for match in matches]
        )
        _, mask = cv2.findHomography(
            previous_xy,
            current_xy,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if mask is None:
            return 0.0
        return float(np.count_nonzero(mask) / max(mask.size, 1))

    def update(self, frame: np.ndarray) -> ShotDecision:
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("shot detector expects a BGR frame")
        resized = self._resize(frame[..., :3])
        histogram = self._histogram(resized)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        green_ratio = self._green_ratio(resized)
        field_view = self._field_view(green_ratio)
        if self._previous_histogram is None or self._previous_gray is None:
            self._previous_histogram = histogram
            self._previous_gray = gray
            return ShotDecision(False, 0.0, 0.0, None, field_view, green_ratio)

        histogram_distance = float(
            cv2.compareHist(
                self._previous_histogram.astype(np.float32),
                histogram.astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA,
            )
        )
        feature_ratio: Optional[float] = None
        if histogram_distance >= self.config.soft_histogram_threshold:
            feature_ratio = self._feature_inlier_ratio(self._previous_gray, gray)
        is_cut = histogram_distance >= self.config.histogram_threshold or (
            histogram_distance >= self.config.soft_histogram_threshold
            and feature_ratio is not None
            and feature_ratio <= self.config.maximum_feature_inlier_ratio
        )
        feature_penalty = 1.0 - float(feature_ratio or 0.0)
        score = float(np.clip(0.75 * histogram_distance + 0.25 * feature_penalty, 0, 1))
        self._previous_histogram = histogram
        self._previous_gray = gray
        return ShotDecision(
            is_cut,
            score,
            histogram_distance,
            feature_ratio,
            field_view,
            green_ratio,
        )
