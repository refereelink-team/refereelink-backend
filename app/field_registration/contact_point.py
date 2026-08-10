"""Ground-contact point selection with conservative bbox fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ContactPointEstimate:
    image_xy: Optional[Tuple[float, float]]
    covariance_px2: Optional[np.ndarray]
    source: str
    confidence: float
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class ContactPointConfig:
    minimum_box_width_px: float = 8.0
    minimum_box_height_px: float = 16.0
    edge_margin_px: float = 3.0
    bbox_sigma_x_ratio: float = 0.08
    bbox_sigma_y_ratio: float = 0.05
    minimum_head_confidence: float = 0.55
    minimum_ankle_confidence: float = 0.40


class GroundContactPointSelector:
    def __init__(self, config: ContactPointConfig | None = None) -> None:
        self.config = config or ContactPointConfig()

    def select(
        self,
        bbox_xyxy: Sequence[float],
        frame_size: Tuple[int, int],
        contact_head_xy: Optional[Tuple[float, float]] = None,
        contact_head_confidence: float = 0.0,
        ankles_xyc: Optional[np.ndarray] = None,
    ) -> ContactPointEstimate:
        bbox = np.asarray(bbox_xyxy, dtype=np.float64)
        if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
            return ContactPointEstimate(None, None, "none", 0.0, "invalid_bbox")
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        frame_width, frame_height = frame_size
        if width < self.config.minimum_box_width_px or height < self.config.minimum_box_height_px:
            return ContactPointEstimate(None, None, "none", 0.0, "small_bbox")
        margin = self.config.edge_margin_px
        if x_min <= margin or x_max >= frame_width - margin or y_max >= frame_height - margin:
            return ContactPointEstimate(None, None, "none", 0.0, "truncated_bbox")

        if (
            contact_head_xy is not None
            and contact_head_confidence >= self.config.minimum_head_confidence
            and np.all(np.isfinite(contact_head_xy))
        ):
            sigma = max(1.0, height * 0.025)
            return ContactPointEstimate(
                tuple(float(value) for value in contact_head_xy),
                np.diag([sigma**2, sigma**2]),
                "contact_head",
                float(contact_head_confidence),
            )

        if ankles_xyc is not None:
            ankles = np.asarray(ankles_xyc, dtype=np.float64)
            if ankles.shape == (2, 3):
                visible = ankles[:, 2] >= self.config.minimum_ankle_confidence
                if np.any(visible):
                    point = np.mean(ankles[visible, :2], axis=0)
                    confidence = float(np.mean(ankles[visible, 2]))
                    sigma = max(1.0, height * (0.035 if np.all(visible) else 0.06))
                    return ContactPointEstimate(
                        (float(point[0]), float(point[1])),
                        np.diag([sigma**2, sigma**2]),
                        "ankles",
                        confidence,
                    )

        point = ((x_min + x_max) / 2.0, y_max)
        sigma_x = max(1.0, width * self.config.bbox_sigma_x_ratio)
        sigma_y = max(1.0, height * self.config.bbox_sigma_y_ratio)
        return ContactPointEstimate(
            image_xy=(float(point[0]), float(point[1])),
            covariance_px2=np.diag([sigma_x**2, sigma_y**2]),
            source="bbox_bottom",
            confidence=0.25,
        )
