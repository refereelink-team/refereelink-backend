"""Dependency-free point/line optimization under a fixed-rig pan constraint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from app.field_registration.camera_model import CameraRigProfile
from app.field_registration.geometry import transform_points
from app.field_registration.types import LineObservation, PointObservation


@dataclass(frozen=True)
class PanRefinerConfig:
    global_grid_steps: int = 181
    local_search_half_width_rad: float = 0.08
    local_grid_steps: int = 33
    refinement_iterations: int = 24
    huber_delta: float = 3.0
    minimum_point_inliers: int = 4
    minimum_line_samples: int = 12
    point_inlier_threshold_px: float = 8.0
    max_mean_point_error_px: float = 10.0
    max_p95_line_error_px: float = 14.0
    prior_sigma_rad: float = 0.04


@dataclass(frozen=True)
class PanRefinementResult:
    success: bool
    pan_rad: Optional[float]
    confidence: float
    point_inliers: int
    mean_point_error_px: Optional[float]
    mean_line_error_px: Optional[float]
    p95_line_error_px: Optional[float]
    objective: Optional[float]
    rejection_reason: Optional[str] = None


class PanOnlyPointLineRefiner:
    def __init__(
        self,
        rig_profile: CameraRigProfile,
        config: PanRefinerConfig | None = None,
    ) -> None:
        self.rig_profile = rig_profile
        self.config = config or PanRefinerConfig()

    @staticmethod
    def _huber(values: np.ndarray, delta: float) -> np.ndarray:
        absolute = np.abs(values)
        return np.where(
            absolute <= delta,
            0.5 * values**2,
            delta * (absolute - 0.5 * delta),
        )

    def _errors(
        self,
        pan_rad: float,
        points: Sequence[PointObservation],
        lines: Sequence[LineObservation],
        image_size: Tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        homography = self.rig_profile.pitch_to_image_homography(pan_rad, image_size)
        point_errors: list[float] = []
        point_weights: list[float] = []
        point_sigmas: list[float] = []
        for observation in points:
            predicted = transform_points(
                np.asarray([observation.pitch_xy_m]), homography
            )[0]
            error = float(np.linalg.norm(predicted - np.asarray(observation.image_xy)))
            if np.isfinite(error):
                point_errors.append(error)
                point_weights.append(observation.confidence)
                point_sigmas.append(observation.sigma_px)

        width, height = image_size
        line_errors: list[float] = []
        line_weights: list[float] = []
        for observation in lines:
            projected = transform_points(observation.pitch_points_m, homography)
            visible = (
                np.all(np.isfinite(projected), axis=1)
                & (projected[:, 0] >= -8.0)
                & (projected[:, 0] <= width + 8.0)
                & (projected[:, 1] >= -8.0)
                & (projected[:, 1] <= height + 8.0)
            )
            projected = projected[visible]
            if projected.size == 0:
                continue
            differences = projected[:, None, :] - observation.image_points[None, :, :]
            squared = np.sum(differences**2, axis=2)
            model_to_observed = np.sqrt(np.min(squared, axis=1))
            observed_to_model = np.sqrt(np.min(squared, axis=0))
            nearest = np.concatenate((model_to_observed, observed_to_model))
            finite = nearest[np.isfinite(nearest)]
            line_errors.extend(finite.tolist())
            line_weights.extend([observation.confidence] * finite.size)
        return (
            np.asarray(point_errors, dtype=np.float64),
            np.asarray(point_weights, dtype=np.float64),
            np.asarray(point_sigmas, dtype=np.float64),
            np.asarray(line_errors, dtype=np.float64),
        )

    def _objective(
        self,
        pan_rad: float,
        points: Sequence[PointObservation],
        lines: Sequence[LineObservation],
        image_size: Tuple[int, int],
        prior_pan_rad: Optional[float],
    ) -> float:
        point_errors, point_weights, point_sigmas, line_errors = self._errors(
            pan_rad, points, lines, image_size
        )
        terms: list[float] = []
        if point_errors.size:
            normalized_point_errors = point_errors / np.maximum(point_sigmas, 1e-6)
            terms.append(
                float(
                    np.sum(
                        self._huber(normalized_point_errors, self.config.huber_delta)
                        * point_weights
                    )
                    / max(np.sum(point_weights), 1e-12)
                )
            )
        if line_errors.size:
            terms.append(
                float(np.mean(self._huber(line_errors, self.config.huber_delta)))
            )
        if not terms:
            return float("inf")
        cost = float(np.mean(terms))
        if prior_pan_rad is not None:
            difference = float(
                (pan_rad - prior_pan_rad + np.pi) % (2.0 * np.pi) - np.pi
            )
            cost += 0.5 * (difference / self.config.prior_sigma_rad) ** 2
        return cost

    def refine(
        self,
        points: Sequence[PointObservation],
        lines: Sequence[LineObservation],
        image_size: Tuple[int, int],
        initial_pan_rad: Optional[float] = None,
        global_search: bool = False,
    ) -> PanRefinementResult:
        if len(points) < self.config.minimum_point_inliers and not lines:
            return PanRefinementResult(
                False, None, 0.0, 0, None, None, None, None, "insufficient_observations"
            )
        lower_limit, upper_limit = self.rig_profile.pan_limits_rad
        if global_search or initial_pan_rad is None:
            lower, upper = lower_limit, upper_limit
            steps = self.config.global_grid_steps
            prior = None
        else:
            lower = max(lower_limit, initial_pan_rad - self.config.local_search_half_width_rad)
            upper = min(upper_limit, initial_pan_rad + self.config.local_search_half_width_rad)
            steps = self.config.local_grid_steps
            prior = initial_pan_rad
        grid = np.linspace(lower, upper, max(steps, 3))
        costs = np.asarray(
            [self._objective(value, points, lines, image_size, prior) for value in grid]
        )
        if not np.any(np.isfinite(costs)):
            return PanRefinementResult(
                False, None, 0.0, 0, None, None, None, None, "optimization_failed"
            )
        best_index = int(np.nanargmin(costs))
        left = grid[max(best_index - 1, 0)]
        right = grid[min(best_index + 1, grid.size - 1)]
        # Golden-section refinement is robust for the one-dimensional state
        # and avoids adding SciPy solely for this tiny constrained problem.
        ratio = (np.sqrt(5.0) - 1.0) / 2.0
        x1 = right - ratio * (right - left)
        x2 = left + ratio * (right - left)
        f1 = self._objective(x1, points, lines, image_size, prior)
        f2 = self._objective(x2, points, lines, image_size, prior)
        for _ in range(self.config.refinement_iterations):
            if f1 <= f2:
                right, x2, f2 = x2, x1, f1
                x1 = right - ratio * (right - left)
                f1 = self._objective(x1, points, lines, image_size, prior)
            else:
                left, x1, f1 = x1, x2, f2
                x2 = left + ratio * (right - left)
                f2 = self._objective(x2, points, lines, image_size, prior)
        pan = float((left + right) / 2.0)
        objective = self._objective(pan, points, lines, image_size, prior)
        point_errors, _, _, line_errors = self._errors(pan, points, lines, image_size)
        point_inliers = int(
            np.count_nonzero(point_errors <= self.config.point_inlier_threshold_px)
        )
        mean_point = float(np.mean(point_errors)) if point_errors.size else None
        mean_line = float(np.mean(line_errors)) if line_errors.size else None
        p95_line = float(np.percentile(line_errors, 95)) if line_errors.size else None
        if point_errors.size and point_inliers < min(
            self.config.minimum_point_inliers, len(points)
        ):
            return PanRefinementResult(
                False,
                pan,
                0.0,
                point_inliers,
                mean_point,
                mean_line,
                p95_line,
                objective,
                "insufficient_point_inliers",
            )
        if mean_point is not None and mean_point > self.config.max_mean_point_error_px:
            return PanRefinementResult(
                False,
                pan,
                0.0,
                point_inliers,
                mean_point,
                mean_line,
                p95_line,
                objective,
                "point_reprojection_error",
            )
        if p95_line is not None and p95_line > self.config.max_p95_line_error_px:
            return PanRefinementResult(
                False,
                pan,
                0.0,
                point_inliers,
                mean_point,
                mean_line,
                p95_line,
                objective,
                "line_reprojection_error",
            )
        point_score = point_inliers / max(len(points), 1) if points else 0.0
        line_score = (
            min(line_errors.size / max(self.config.minimum_line_samples, 1), 1.0)
            if line_errors.size
            else 0.0
        )
        error_score = float(np.exp(-max(mean_point or 0.0, mean_line or 0.0) / 8.0))
        confidence = float(np.clip(0.45 * point_score + 0.25 * line_score + 0.30 * error_score, 0, 1))
        return PanRefinementResult(
            True,
            pan,
            confidence,
            point_inliers,
            mean_point,
            mean_line,
            p95_line,
            objective,
        )
