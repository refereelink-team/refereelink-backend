"""Sub-pixel decoding helpers for keypoint heatmaps and semantic line masks."""

from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np


def decode_heatmap_peak(
    heatmap: np.ndarray,
    output_size: Tuple[int, int] | None = None,
    window_radius: int = 2,
) -> Tuple[Tuple[float, float], float]:
    """Decode a peak with a local probability-weighted centroid."""

    values = np.asarray(heatmap, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("heatmap must be a non-empty finite 2D array")
    peak_y, peak_x = np.unravel_index(int(np.argmax(values)), values.shape)
    y0 = max(0, peak_y - window_radius)
    y1 = min(values.shape[0], peak_y + window_radius + 1)
    x0 = max(0, peak_x - window_radius)
    x1 = min(values.shape[1], peak_x + window_radius + 1)
    patch = values[y0:y1, x0:x1]
    patch = np.exp(patch - np.max(patch))
    normalizer = float(np.sum(patch))
    grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
    x_coord = float(np.sum(grid_x * patch) / max(normalizer, 1e-12))
    y_coord = float(np.sum(grid_y * patch) / max(normalizer, 1e-12))
    confidence = float(values[peak_y, peak_x])
    if output_size is not None:
        output_width, output_height = output_size
        x_coord = (x_coord + 0.5) * output_width / values.shape[1] - 0.5
        y_coord = (y_coord + 0.5) * output_height / values.shape[0] - 0.5
    return (x_coord, y_coord), confidence


def sample_semantic_mask(
    probability_mask: np.ndarray,
    threshold: float = 0.5,
    maximum_samples: int = 256,
) -> np.ndarray:
    """Return spatially distributed line pixels from a semantic mask."""

    mask = np.asarray(probability_mask, dtype=np.float32)
    if mask.ndim != 2 or not np.all(np.isfinite(mask)):
        raise ValueError("probability mask must be a finite 2D array")
    binary = (mask >= threshold).astype(np.uint8)
    if not np.any(binary):
        return np.empty((0, 2), dtype=np.float64)
    # Skeletonization is intentionally avoided to keep opencv-contrib out of
    # the runtime.  One-pixel erosion boundary plus uniform sampling preserves
    # the spatial extent needed by the geometric optimizer.
    boundary = binary - cv2.erode(binary, np.ones((3, 3), np.uint8))
    y_coord, x_coord = np.nonzero(boundary if np.any(boundary) else binary)
    points = np.column_stack((x_coord, y_coord)).astype(np.float64)
    if points.shape[0] > maximum_samples:
        indices = np.linspace(0, points.shape[0] - 1, maximum_samples).astype(int)
        points = points[indices]
    return points


def sample_pitch_segment(
    start_xy_m: Iterable[float],
    end_xy_m: Iterable[float],
    sample_count: int = 64,
) -> np.ndarray:
    start = np.asarray(tuple(start_xy_m), dtype=np.float64)
    end = np.asarray(tuple(end_xy_m), dtype=np.float64)
    if start.shape != (2,) or end.shape != (2,) or sample_count < 2:
        raise ValueError("segment endpoints must be 2D and sample_count >= 2")
    return np.linspace(start, end, sample_count)
