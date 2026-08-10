"""Robust point-based initializer with MAGSAC and explicit degeneracy gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from app.field_registration.geometry import (
    convex_hull_coverage,
    normalized_homography_condition,
    transform_points,
)
from app.field_registration.types import PointObservation


@dataclass(frozen=True)
class HomographyInitializerConfig:
    min_points: int = 4
    min_inliers: int = 4
    min_inlier_ratio: float = 0.35
    min_confidence: float = 0.20
    threshold_diagonal_ratio: float = 0.004
    min_threshold_px: float = 1.5
    min_image_coverage: float = 0.003
    min_world_coverage: float = 0.003
    min_second_singular_ratio: float = 0.015
    max_normalized_condition: float = 1e5
    max_mean_reprojection_error_px: float = 8.0
    max_p95_reprojection_error_px: float = 15.0
    confidence_iterations: int = 10000


@dataclass(frozen=True)
class HomographyInitialization:
    image_to_pitch: Optional[np.ndarray]
    pitch_to_image: Optional[np.ndarray]
    inlier_indices: Tuple[int, ...]
    mean_reprojection_error_px: Optional[float]
    p95_reprojection_error_px: Optional[float]
    image_coverage: float
    world_coverage: float
    normalized_condition: Optional[float]
    method: str
    rejection_reason: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.image_to_pitch is not None


class HomographyInitializer:
    def __init__(self, config: HomographyInitializerConfig | None = None) -> None:
        self.config = config or HomographyInitializerConfig()

    @staticmethod
    def _direction_ratio(points: np.ndarray) -> float:
        centred = points - np.mean(points, axis=0)
        singular_values = np.linalg.svd(centred, compute_uv=False)
        if singular_values.size < 2 or singular_values[0] < 1e-12:
            return 0.0
        return float(singular_values[1] / singular_values[0])

    def _failure(
        self,
        reason: str,
        image_coverage: float = 0.0,
        world_coverage: float = 0.0,
        condition: Optional[float] = None,
        method: str = "USAC_MAGSAC",
    ) -> HomographyInitialization:
        return HomographyInitialization(
            image_to_pitch=None,
            pitch_to_image=None,
            inlier_indices=(),
            mean_reprojection_error_px=None,
            p95_reprojection_error_px=None,
            image_coverage=image_coverage,
            world_coverage=world_coverage,
            normalized_condition=condition,
            method=method,
            rejection_reason=reason,
        )

    def estimate(
        self,
        observations: Sequence[PointObservation],
        image_size: Tuple[int, int],
        pitch_size_m: Tuple[float, float],
    ) -> HomographyInitialization:
        candidates = [
            (index, observation)
            for index, observation in enumerate(observations)
            if observation.confidence >= self.config.min_confidence
        ]
        candidates.sort(key=lambda item: item[1].confidence, reverse=True)
        if len(candidates) < self.config.min_points:
            return self._failure("insufficient_points")

        original_indices = np.array([item[0] for item in candidates], dtype=np.int64)
        image_points = np.asarray([item[1].image_xy for item in candidates], dtype=np.float64)
        world_points = np.asarray(
            [item[1].pitch_xy_m for item in candidates], dtype=np.float64
        )
        image_coverage = convex_hull_coverage(image_points, image_size)
        pitch_length, pitch_width = pitch_size_m
        world_area = max(float(pitch_length * pitch_width), 1e-12)
        world_hull = cv2.convexHull(world_points.astype(np.float32).reshape(-1, 1, 2))
        world_coverage = float(np.clip(cv2.contourArea(world_hull) / world_area, 0.0, 1.0))
        if image_coverage < self.config.min_image_coverage:
            return self._failure("insufficient_image_coverage", image_coverage, world_coverage)
        if world_coverage < self.config.min_world_coverage:
            return self._failure("insufficient_world_coverage", image_coverage, world_coverage)
        if (
            self._direction_ratio(image_points) < self.config.min_second_singular_ratio
            or self._direction_ratio(world_points) < self.config.min_second_singular_ratio
        ):
            return self._failure("near_collinear_points", image_coverage, world_coverage)

        width, height = image_size
        threshold = max(
            self.config.min_threshold_px,
            float(np.hypot(width, height) * self.config.threshold_diagonal_ratio),
        )
        method_value = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        method_name = "USAC_MAGSAC" if hasattr(cv2, "USAC_MAGSAC") else "RANSAC"

        def fit(method: int) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            try:
                # Fit pitch -> image so OpenCV's robust threshold is expressed
                # in pixels.  Fitting image -> pitch would silently interpret
                # this as metres and make the threshold unit-dependent.
                return cv2.findHomography(
                    world_points,
                    image_points,
                    method=method,
                    ransacReprojThreshold=threshold,
                    maxIters=self.config.confidence_iterations,
                    confidence=0.999,
                )
            except cv2.error:
                return None, None

        pitch_to_image, mask = fit(method_value)
        minimum_consensus = max(
            self.config.min_inliers,
            int(np.ceil(len(candidates) * self.config.min_inlier_ratio)),
        )
        # Some OpenCV builds have numerically fragile USAC behavior for highly
        # projective sports views.  Never accept a tiny, internally consistent
        # four/five-point model from a much larger observation set.  A guarded
        # RANSAC fallback is safer than returning that false-positive matrix.
        if (
            method_value != cv2.RANSAC
            and (
                pitch_to_image is None
                or mask is None
                or int(np.count_nonzero(mask)) < minimum_consensus
            )
        ):
            pitch_to_image, mask = fit(cv2.RANSAC)
            method_name = "RANSAC_FALLBACK"
        if pitch_to_image is None or mask is None:
            return self._failure(
                "estimation_failed", image_coverage, world_coverage, method=method_name
            )

        inlier_mask = mask.reshape(-1).astype(bool)
        if int(np.count_nonzero(inlier_mask)) < minimum_consensus:
            return self._failure(
                "insufficient_inliers", image_coverage, world_coverage, method=method_name
            )
        inlier_image = image_points[inlier_mask]
        inlier_world = world_points[inlier_mask]
        try:
            image_to_pitch = np.linalg.inv(pitch_to_image)
            condition = normalized_homography_condition(
                image_to_pitch, inlier_image, inlier_world
            )
        except (np.linalg.LinAlgError, ValueError):
            return self._failure(
                "singular_homography", image_coverage, world_coverage, method=method_name
            )
        if condition > self.config.max_normalized_condition:
            return self._failure(
                "ill_conditioned_homography",
                image_coverage,
                world_coverage,
                condition,
                method_name,
            )

        reprojected = transform_points(inlier_world, pitch_to_image)
        errors = np.linalg.norm(reprojected - inlier_image, axis=1)
        mean_error = float(np.mean(errors))
        p95_error = float(np.percentile(errors, 95))
        if mean_error > self.config.max_mean_reprojection_error_px:
            return self._failure(
                "mean_reprojection_error",
                image_coverage,
                world_coverage,
                condition,
                method_name,
            )
        if p95_error > self.config.max_p95_reprojection_error_px:
            return self._failure(
                "p95_reprojection_error",
                image_coverage,
                world_coverage,
                condition,
                method_name,
            )

        image_to_pitch = image_to_pitch / image_to_pitch[2, 2]
        pitch_to_image = pitch_to_image / pitch_to_image[2, 2]
        inlier_indices = tuple(int(value) for value in original_indices[inlier_mask])
        return HomographyInitialization(
            image_to_pitch=image_to_pitch,
            pitch_to_image=pitch_to_image,
            inlier_indices=inlier_indices,
            mean_reprojection_error_px=mean_error,
            p95_reprojection_error_px=p95_error,
            image_coverage=image_coverage,
            world_coverage=world_coverage,
            normalized_condition=condition,
            method=method_name,
        )
