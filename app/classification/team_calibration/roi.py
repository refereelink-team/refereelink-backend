from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ROIBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


class JerseyROIExtractor:
    """Extract the torso area without requiring pose or segmentation models."""

    def __init__(
        self,
        *,
        top_ratio: float = 0.15,
        bottom_ratio: float = 0.65,
        horizontal_keep_ratio: float = 0.75,
    ) -> None:
        if not 0.0 <= top_ratio < bottom_ratio <= 1.0:
            raise ValueError("ROI top_ratio and bottom_ratio must satisfy 0 <= top < bottom <= 1")
        if not 0.0 < horizontal_keep_ratio <= 1.0:
            raise ValueError("horizontal_keep_ratio must be in (0, 1]")
        self.top_ratio = float(top_ratio)
        self.bottom_ratio = float(bottom_ratio)
        self.horizontal_keep_ratio = float(horizontal_keep_ratio)

    def box(self, frame_shape: Sequence[int], bbox: Sequence[float]) -> Optional[ROIBox]:
        if len(frame_shape) < 2:
            return None
        coordinates = np.asarray(bbox, dtype=np.float64).reshape(-1)
        if coordinates.size < 4 or not np.isfinite(coordinates[:4]).all():
            return None
        frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
        x1, y1, x2, y2 = coordinates[:4]
        left = max(0, min(frame_width, int(np.floor(min(x1, x2)))))
        right = max(0, min(frame_width, int(np.ceil(max(x1, x2)))))
        top = max(0, min(frame_height, int(np.floor(min(y1, y2)))))
        bottom = max(0, min(frame_height, int(np.ceil(max(y1, y2)))))
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            return None

        torso_top = top + int(round(height * self.top_ratio))
        torso_bottom = top + int(round(height * self.bottom_ratio))
        keep_width = int(round(width * self.horizontal_keep_ratio))
        horizontal_margin = max(0, (width - keep_width) // 2)
        roi = ROIBox(
            x1=left + horizontal_margin,
            y1=torso_top,
            x2=right - (width - keep_width - horizontal_margin),
            y2=torso_bottom,
        )
        if roi.width <= 0 or roi.height <= 0:
            return None
        return roi

    def extract(self, frame: Optional[np.ndarray], bbox: Sequence[float]) -> np.ndarray:
        """Return a copied BGR crop, or an empty crop when unavailable."""

        empty = np.empty((0, 0, 3), dtype=np.uint8)
        if frame is None:
            return empty
        image = np.asarray(frame)
        if image.ndim < 2:
            return empty
        roi = self.box(image.shape, bbox)
        if roi is None:
            return empty
        crop = image[roi.y1:roi.y2, roi.x1:roi.x2]
        if crop.size == 0:
            return empty
        return crop[..., :3].copy()

