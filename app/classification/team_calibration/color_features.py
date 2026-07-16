from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


_EPS = 1e-8


class ColorFeatureExtractor:
    """HSV/Lab histogram descriptor for jersey appearance."""

    def __init__(self, *, spatial_blocks: int = 1) -> None:
        if spatial_blocks not in {1, 2}:
            raise ValueError("spatial_blocks must be 1 or 2")
        self.spatial_blocks = spatial_blocks
        # H, S, a, b histograms.  Each component is independently normalized
        # before the complete descriptor receives an L2 normalization.
        self._cell_dim = 12 + 8 + 8 + 8

    @property
    def feature_dim(self) -> int:
        return self._cell_dim * self.spatial_blocks * self.spatial_blocks

    def extract(self, crop: Optional[np.ndarray]) -> np.ndarray:
        image = np.asarray(crop) if crop is not None else np.empty((0, 0, 3), dtype=np.uint8)
        if image.ndim < 3 or image.shape[2] < 3 or image.shape[0] < 2 or image.shape[1] < 2:
            return np.zeros(self.feature_dim, dtype=np.float32)
        bgr = image[..., :3].astype(np.uint8, copy=False)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        cells: list[np.ndarray] = []
        rows = self.spatial_blocks
        cols = self.spatial_blocks
        for row in range(rows):
            y1 = row * bgr.shape[0] // rows
            y2 = (row + 1) * bgr.shape[0] // rows
            for col in range(cols):
                x1 = col * bgr.shape[1] // cols
                x2 = (col + 1) * bgr.shape[1] // cols
                cells.append(
                    self._cell_descriptor(
                        hsv[y1:y2, x1:x2],
                        lab[y1:y2, x1:x2],
                    )
                )
        feature = np.concatenate(cells).astype(np.float32)
        norm = float(np.linalg.norm(feature))
        if norm > _EPS:
            feature /= norm
        return feature

    def extract_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        return np.stack([self.extract(crop) for crop in crops]).astype(np.float32)

    @staticmethod
    def _cell_descriptor(hsv: np.ndarray, lab: np.ndarray) -> np.ndarray:
        hue = hsv[..., 0].astype(np.float32) / 180.0
        saturation = hsv[..., 1].astype(np.float32) / 255.0
        channel_a = lab[..., 1].astype(np.float32) / 255.0
        channel_b = lab[..., 2].astype(np.float32) / 255.0
        # Low-saturation pixels have no useful hue information.  Retaining
        # them in S/Lab histograms still preserves white/black kit details.
        hue_weights = np.clip(saturation, 0.05, 1.0)
        histograms = [
            np.histogram(hue, bins=12, range=(0.0, 1.0), weights=hue_weights)[0],
            np.histogram(saturation, bins=8, range=(0.0, 1.0))[0],
            np.histogram(channel_a, bins=8, range=(0.0, 1.0))[0],
            np.histogram(channel_b, bins=8, range=(0.0, 1.0))[0],
        ]
        normalized = []
        for histogram in histograms:
            values = histogram.astype(np.float32)
            total = float(values.sum())
            if total > _EPS:
                values /= total
            normalized.append(values)
        return np.concatenate(normalized)

