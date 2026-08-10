"""Numerically guarded planar geometry helpers."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def validate_homography(homography: np.ndarray) -> np.ndarray:
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("homography must be a finite 3x3 matrix")
    scale = float(np.linalg.norm(matrix))
    if scale < 1e-12 or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("homography is singular")
    matrix = matrix / (matrix[2, 2] if abs(matrix[2, 2]) > 1e-12 else scale)
    if np.linalg.cond(matrix) > 1e14:
        raise ValueError("homography is ill-conditioned")
    return matrix


def transform_points(points_xy: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")
    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    matrix = validate_homography(homography)
    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    transformed = (matrix @ homogeneous.T).T
    denominator = transformed[:, 2]
    valid = np.abs(denominator) > 1e-12
    result = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    result[valid] = transformed[valid, :2] / denominator[valid, None]
    return result


def convex_hull_coverage(points_xy: np.ndarray, image_size: Tuple[int, int]) -> float:
    points = np.asarray(points_xy, dtype=np.float32)
    width, height = image_size
    if points.shape[0] < 3 or width <= 0 or height <= 0:
        return 0.0
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    return float(np.clip(cv2.contourArea(hull) / (width * height), 0.0, 1.0))


def normalized_homography_condition(
    homography: np.ndarray,
    source_points: np.ndarray,
    destination_points: np.ndarray,
) -> float:
    """Condition number after Hartley normalization of both point sets."""

    def normalization(points: np.ndarray) -> np.ndarray:
        centre = np.mean(points, axis=0)
        mean_distance = float(np.mean(np.linalg.norm(points - centre, axis=1)))
        scale = np.sqrt(2.0) / max(mean_distance, 1e-12)
        return np.array(
            [
                [scale, 0.0, -scale * centre[0]],
                [0.0, scale, -scale * centre[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    source = np.asarray(source_points, dtype=np.float64)
    destination = np.asarray(destination_points, dtype=np.float64)
    source_transform = normalization(source)
    destination_transform = normalization(destination)
    normalized = destination_transform @ homography @ np.linalg.inv(source_transform)
    return float(np.linalg.cond(normalized))
