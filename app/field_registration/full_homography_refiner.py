"""Robust eight-DoF point/line refinement for unconstrained broadcast views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from app.field_registration.geometry import transform_points, validate_homography
from app.field_registration.types import LineObservation, PointObservation


@dataclass(frozen=True)
class FullHomographyRefinerConfig:
    iterations: int = 10
    huber_delta_px: float = 4.0
    initial_damping: float = 1e-2
    finite_difference_step: float = 2e-4
    maximum_samples_per_line: int = 36
    minimum_point_inliers: int = 4
    point_inlier_threshold_px: float = 8.0
    max_mean_point_error_px: float = 9.0
    max_p95_line_error_px: float = 14.0
    maximum_parameter_step: float = 0.35


@dataclass(frozen=True)
class FullHomographyRefinement:
    success: bool
    pitch_to_image: Optional[np.ndarray]
    image_to_pitch: Optional[np.ndarray]
    confidence: float
    point_inliers: int
    visible_segment_count: int
    mean_point_error_px: Optional[float]
    mean_line_error_px: Optional[float]
    p95_line_error_px: Optional[float]
    objective: Optional[float]
    rejection_reason: Optional[str] = None


class FullHomographyPointLineRefiner:
    """Refine an initialized homography with sparse points and semantic lines.

    It uses a small damped IRLS solver and numerical Jacobians, keeping SciPy
    out of the realtime dependency set. This is intentionally a correction
    stage: robust global initialization remains the responsibility of MAGSAC.
    """

    def __init__(self, config: FullHomographyRefinerConfig | None = None) -> None:
        self.config = config or FullHomographyRefinerConfig()

    @staticmethod
    def _sample(points: np.ndarray, maximum: int) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.shape[0] <= maximum:
            return values
        indices = np.linspace(0, values.shape[0] - 1, maximum).astype(np.int64)
        return values[indices]

    @staticmethod
    def _parameter_scales(initial: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        width, height = image_size
        lower_bounds = np.asarray(
            [0.25, 0.25, width * 0.05, 0.25, 0.25, height * 0.05, 1e-3, 1e-3],
            dtype=np.float64,
        )
        values = initial.reshape(-1)[:8]
        return np.maximum(np.abs(values), lower_bounds)

    @staticmethod
    def _matrix(initial: np.ndarray, scales: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        values = initial.reshape(-1)[:8] + scales * parameters
        return np.asarray((*values, 1.0), dtype=np.float64).reshape(3, 3)

    def _residuals(
        self,
        homography: np.ndarray,
        points: Sequence[PointObservation],
        lines: Sequence[LineObservation],
        image_size: Tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        residuals: list[np.ndarray] = []
        groups: list[np.ndarray] = []
        point_errors: list[float] = []
        line_errors: list[float] = []
        visible_segments = 0
        group_id = 0
        for observation in points:
            predicted = transform_points(
                np.asarray([observation.pitch_xy_m], dtype=np.float64), homography
            )[0]
            if not np.all(np.isfinite(predicted)):
                continue
            difference = (predicted - np.asarray(observation.image_xy)) / max(
                observation.sigma_px, 0.5
            )
            difference *= np.sqrt(max(observation.confidence, 1e-3))
            residuals.append(difference)
            groups.append(np.full(2, group_id, dtype=np.int64))
            point_errors.append(float(np.linalg.norm(predicted - observation.image_xy)))
            group_id += 1

        width, height = image_size
        for observation in lines:
            pitch_samples = self._sample(
                observation.pitch_points_m, self.config.maximum_samples_per_line
            )
            image_samples = self._sample(
                observation.image_points, self.config.maximum_samples_per_line
            )
            projected = transform_points(pitch_samples, homography)
            visible = (
                np.all(np.isfinite(projected), axis=1)
                & (projected[:, 0] >= -12.0)
                & (projected[:, 0] <= width + 12.0)
                & (projected[:, 1] >= -12.0)
                & (projected[:, 1] <= height + 12.0)
            )
            projected = projected[visible]
            if projected.shape[0] < 2 or image_samples.shape[0] < 2:
                continue
            visible_segments += 1
            differences = projected[:, None, :] - image_samples[None, :, :]
            squared = np.sum(differences**2, axis=2)
            nearest_observed = image_samples[np.argmin(squared, axis=1)]
            nearest_projected = projected[np.argmin(squared, axis=0)]
            forward = projected - nearest_observed
            backward = image_samples - nearest_projected
            weight = np.sqrt(max(observation.confidence, 1e-3))
            for difference in (forward, backward):
                residuals.append((difference * weight).reshape(-1))
                count = difference.shape[0]
                groups.append(np.repeat(np.arange(group_id, group_id + count), 2))
                group_id += count
            line_errors.extend(
                np.concatenate(
                    (
                        np.sqrt(np.min(squared, axis=1)),
                        np.sqrt(np.min(squared, axis=0)),
                    )
                ).tolist()
            )
        if not residuals:
            return np.empty(0), np.empty(0), np.empty(0), 0
        return (
            np.concatenate(residuals),
            np.asarray(point_errors, dtype=np.float64),
            np.asarray(line_errors, dtype=np.float64),
            visible_segments,
        )

    def _weights(self, residuals: np.ndarray) -> np.ndarray:
        absolute = np.abs(residuals)
        delta = self.config.huber_delta_px
        return np.where(absolute <= delta, 1.0, delta / np.maximum(absolute, 1e-9))

    def refine(
        self,
        initial_pitch_to_image: np.ndarray,
        points: Sequence[PointObservation],
        lines: Sequence[LineObservation],
        image_size: Tuple[int, int],
    ) -> FullHomographyRefinement:
        if len(points) < self.config.minimum_point_inliers and not lines:
            return FullHomographyRefinement(
                False, None, None, 0.0, 0, 0, None, None, None, None,
                "insufficient_observations",
            )
        try:
            initial = validate_homography(initial_pitch_to_image)
        except ValueError:
            return FullHomographyRefinement(
                False, None, None, 0.0, 0, 0, None, None, None, None,
                "invalid_initial_homography",
            )
        initial = initial / initial[2, 2]
        scales = self._parameter_scales(initial, image_size)
        parameters = np.zeros(8, dtype=np.float64)
        damping = self.config.initial_damping

        def evaluate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
            try:
                matrix = validate_homography(self._matrix(initial, scales, values))
                return self._residuals(matrix, points, lines, image_size)
            except (ValueError, np.linalg.LinAlgError):
                return np.empty(0), np.empty(0), np.empty(0), 0

        residuals, _, _, _ = evaluate(parameters)
        if residuals.size == 0:
            return FullHomographyRefinement(
                False, None, None, 0.0, 0, 0, None, None, None, None,
                "no_finite_residuals",
            )
        objective = float(np.mean(self._weights(residuals) * residuals**2))
        for _ in range(self.config.iterations):
            jacobian = np.empty((residuals.size, 8), dtype=np.float64)
            step_size = self.config.finite_difference_step
            valid_jacobian = True
            for index in range(8):
                shifted = parameters.copy()
                shifted[index] += step_size
                shifted_residuals, _, _, _ = evaluate(shifted)
                if shifted_residuals.shape != residuals.shape:
                    valid_jacobian = False
                    break
                jacobian[:, index] = (shifted_residuals - residuals) / step_size
            if not valid_jacobian:
                break
            weights = self._weights(residuals)
            weighted_jacobian = jacobian * np.sqrt(weights[:, None])
            weighted_residuals = residuals * np.sqrt(weights)
            normal = weighted_jacobian.T @ weighted_jacobian
            gradient = weighted_jacobian.T @ weighted_residuals
            try:
                step = -np.linalg.solve(normal + damping * np.eye(8), gradient)
            except np.linalg.LinAlgError:
                break
            norm = float(np.linalg.norm(step))
            if norm > self.config.maximum_parameter_step:
                step *= self.config.maximum_parameter_step / norm
            candidate = parameters + step
            candidate_residuals, _, _, _ = evaluate(candidate)
            if candidate_residuals.shape != residuals.shape:
                damping *= 4.0
                continue
            candidate_objective = float(
                np.mean(self._weights(candidate_residuals) * candidate_residuals**2)
            )
            if candidate_objective < objective:
                parameters = candidate
                residuals = candidate_residuals
                objective = candidate_objective
                damping = max(damping * 0.5, 1e-6)
                if norm < 1e-5:
                    break
            else:
                damping *= 4.0

        try:
            pitch_to_image = validate_homography(
                self._matrix(initial, scales, parameters)
            )
            image_to_pitch = validate_homography(np.linalg.inv(pitch_to_image))
        except (ValueError, np.linalg.LinAlgError):
            return FullHomographyRefinement(
                False, None, None, 0.0, 0, 0, None, None, None, objective,
                "singular_refinement",
            )
        _, point_errors, line_errors, visible_segments = self._residuals(
            pitch_to_image, points, lines, image_size
        )
        point_inliers = int(
            np.count_nonzero(point_errors <= self.config.point_inlier_threshold_px)
        )
        mean_point = float(np.mean(point_errors)) if point_errors.size else None
        mean_line = float(np.mean(line_errors)) if line_errors.size else None
        p95_line = float(np.percentile(line_errors, 95)) if line_errors.size else None
        if point_errors.size and point_inliers < min(
            self.config.minimum_point_inliers, len(points)
        ):
            reason = "insufficient_point_inliers"
        elif mean_point is not None and mean_point > self.config.max_mean_point_error_px:
            reason = "point_reprojection_error"
        elif p95_line is not None and p95_line > self.config.max_p95_line_error_px:
            reason = "line_reprojection_error"
        else:
            reason = None
        if reason is not None:
            return FullHomographyRefinement(
                False, pitch_to_image, image_to_pitch, 0.0, point_inliers,
                visible_segments, mean_point, mean_line, p95_line, objective, reason,
            )
        point_score = point_inliers / max(len(points), 1) if points else 0.0
        line_score = min(visible_segments / 3.0, 1.0)
        error_score = float(np.exp(-max(mean_point or 0.0, mean_line or 0.0) / 8.0))
        confidence = float(
            np.clip(0.45 * point_score + 0.25 * line_score + 0.30 * error_score, 0, 1)
        )
        return FullHomographyRefinement(
            True, pitch_to_image, image_to_pitch, confidence, point_inliers,
            visible_segments, mean_point, mean_line, p95_line, objective,
        )
