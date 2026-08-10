"""Accuracy, stability, and safety metrics for field registration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from app.field_registration.geometry import transform_points, validate_homography


@dataclass(frozen=True)
class ErrorSummary:
    count: int
    mean: Optional[float]
    median: Optional[float]
    p90: Optional[float]
    p95: Optional[float]
    maximum: Optional[float]

    @classmethod
    def from_values(cls, values: Iterable[float]) -> "ErrorSummary":
        array = np.asarray(list(values), dtype=np.float64)
        array = array[np.isfinite(array)]
        if array.size == 0:
            return cls(0, None, None, None, None, None)
        return cls(
            count=int(array.size),
            mean=float(np.mean(array)),
            median=float(np.median(array)),
            p90=float(np.percentile(array, 90)),
            p95=float(np.percentile(array, 95)),
            maximum=float(np.max(array)),
        )

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


@dataclass(frozen=True)
class RegistrationMetrics:
    image_reprojection_px: ErrorSummary
    pitch_projection_m: ErrorSummary
    grid_projection_m: ErrorSummary
    symmetric_transfer: ErrorSummary
    valid_grid_ratio: float


def correspondence_errors(
    image_points: np.ndarray,
    pitch_points_m: np.ndarray,
    image_to_pitch: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image_points, dtype=np.float64)
    pitch = np.asarray(pitch_points_m, dtype=np.float64)
    if image.shape != pitch.shape or image.ndim != 2 or image.shape[1] != 2:
        raise ValueError("image and pitch correspondences must both have shape (N, 2)")
    matrix = validate_homography(image_to_pitch)
    pitch_to_image = np.linalg.inv(matrix)
    predicted_pitch = transform_points(image, matrix)
    predicted_image = transform_points(pitch, pitch_to_image)
    return (
        np.linalg.norm(predicted_image - image, axis=1),
        np.linalg.norm(predicted_pitch - pitch, axis=1),
    )


def grid_projection_errors(
    predicted_image_to_pitch: np.ndarray,
    ground_truth_image_to_pitch: np.ndarray,
    image_size: Tuple[int, int],
    x_steps: int = 20,
    y_steps: int = 12,
) -> Tuple[np.ndarray, float]:
    width, height = image_size
    x_values = np.linspace(0.0, width - 1.0, x_steps)
    y_values = np.linspace(0.0, height - 1.0, y_steps)
    grid = np.array(np.meshgrid(x_values, y_values)).reshape(2, -1).T
    predicted = transform_points(grid, predicted_image_to_pitch)
    ground_truth = transform_points(grid, ground_truth_image_to_pitch)
    valid = np.all(np.isfinite(predicted), axis=1) & np.all(np.isfinite(ground_truth), axis=1)
    errors = np.linalg.norm(predicted[valid] - ground_truth[valid], axis=1)
    return errors, float(np.count_nonzero(valid) / max(grid.shape[0], 1))


def symmetric_transfer_errors(
    predicted_image_to_pitch: np.ndarray,
    ground_truth_image_to_pitch: np.ndarray,
    image_points: np.ndarray,
    pitch_points_m: np.ndarray,
    pitch_scale_px_per_m: float = 10.0,
) -> np.ndarray:
    predicted = validate_homography(predicted_image_to_pitch)
    truth = validate_homography(ground_truth_image_to_pitch)
    image = np.asarray(image_points, dtype=np.float64)
    pitch = np.asarray(pitch_points_m, dtype=np.float64)
    predicted_pitch = transform_points(image, predicted)
    truth_pitch = transform_points(image, truth)
    predicted_image = transform_points(pitch, np.linalg.inv(predicted))
    truth_image = transform_points(pitch, np.linalg.inv(truth))
    pitch_error_px_equivalent = (
        np.linalg.norm(predicted_pitch - truth_pitch, axis=1) * pitch_scale_px_per_m
    )
    image_error = np.linalg.norm(predicted_image - truth_image, axis=1)
    return np.sqrt(image_error**2 + pitch_error_px_equivalent**2)


def template_jitter_px(
    pitch_to_image_sequence: Sequence[np.ndarray],
    pitch_grid_m: np.ndarray,
) -> ErrorSummary:
    if len(pitch_to_image_sequence) < 2:
        return ErrorSummary.from_values([])
    previous = transform_points(pitch_grid_m, pitch_to_image_sequence[0])
    jumps: list[float] = []
    for homography in pitch_to_image_sequence[1:]:
        current = transform_points(pitch_grid_m, homography)
        valid = np.all(np.isfinite(previous), axis=1) & np.all(np.isfinite(current), axis=1)
        jumps.extend(np.linalg.norm(current[valid] - previous[valid], axis=1).tolist())
        previous = current
    return ErrorSummary.from_values(jumps)


def evaluate_registration(
    predicted_image_to_pitch: np.ndarray,
    ground_truth_image_to_pitch: np.ndarray,
    image_points: np.ndarray,
    pitch_points_m: np.ndarray,
    image_size: Tuple[int, int],
) -> RegistrationMetrics:
    image_errors, pitch_errors = correspondence_errors(
        image_points, pitch_points_m, predicted_image_to_pitch
    )
    grid_errors, valid_grid_ratio = grid_projection_errors(
        predicted_image_to_pitch,
        ground_truth_image_to_pitch,
        image_size,
    )
    symmetric = symmetric_transfer_errors(
        predicted_image_to_pitch,
        ground_truth_image_to_pitch,
        image_points,
        pitch_points_m,
    )
    return RegistrationMetrics(
        image_reprojection_px=ErrorSummary.from_values(image_errors),
        pitch_projection_m=ErrorSummary.from_values(pitch_errors),
        grid_projection_m=ErrorSummary.from_values(grid_errors),
        symmetric_transfer=ErrorSummary.from_values(symmetric),
        valid_grid_ratio=valid_grid_ratio,
    )
