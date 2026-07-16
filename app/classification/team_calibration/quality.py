from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityAssessment:
    accepted: bool
    score: float
    reasons: tuple[str, ...]
    blur_score: float
    brightness: float
    green_edge_ratio: float
    detection_confidence: float


class CropQualityAssessor:
    """Score a jersey crop and expose machine-readable rejection reasons."""

    def __init__(
        self,
        *,
        min_width: int = 12,
        min_height: int = 20,
        min_detection_confidence: float = 0.5,
        min_blur_score: float = 20.0,
        min_quality_score: float = 0.35,
        max_green_edge_ratio: float = 0.45,
    ) -> None:
        self.min_width = int(min_width)
        self.min_height = int(min_height)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_blur_score = float(min_blur_score)
        self.min_quality_score = float(min_quality_score)
        self.max_green_edge_ratio = float(max_green_edge_ratio)

    def assess(
        self,
        crop: Optional[np.ndarray],
        *,
        detection_confidence: float = 1.0,
    ) -> QualityAssessment:
        image = np.asarray(crop) if crop is not None else np.empty((0, 0, 3), dtype=np.uint8)
        reasons: list[str] = []
        confidence = float(np.clip(detection_confidence, 0.0, 1.0))
        if image.ndim < 3 or image.shape[2] < 3 or image.shape[0] < self.min_height:
            reasons.append("roi_too_small")
        if image.ndim < 3 or image.shape[2] < 3 or image.shape[1] < self.min_width:
            if "roi_too_small" not in reasons:
                reasons.append("roi_too_small")
        if confidence < self.min_detection_confidence:
            reasons.append("detection_confidence_low")
        # A clipped or otherwise unavailable detection can produce a zero-area
        # array with shape ``(0, 0, 3)``.  Treat it as a rejected observation
        # before calling OpenCV, which does not accept empty images.
        if image.size == 0 or image.ndim < 3 or image.shape[2] < 3:
            return QualityAssessment(False, 0.0, tuple(reasons), 0.0, 0.0, 1.0, confidence)

        bgr = image[..., :3].astype(np.uint8, copy=False)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        brightness = float(hsv[..., 2].mean() / 255.0)
        edge_height = max(1, int(round(bgr.shape[0] * 0.2)))
        edge_width = max(1, int(round(bgr.shape[1] * 0.2)))
        border = np.zeros(bgr.shape[:2], dtype=bool)
        border[:edge_height] = True
        border[-edge_height:] = True
        border[:, :edge_width] = True
        border[:, -edge_width:] = True
        green = (
            (hsv[..., 0] >= 35)
            & (hsv[..., 0] <= 90)
            & (hsv[..., 1] >= 45)
            & (hsv[..., 2] >= 35)
        )
        green_edge_ratio = float(green[border].mean()) if border.any() else 0.0

        if blur_score < self.min_blur_score:
            reasons.append("blur_low")
        if green_edge_ratio > self.max_green_edge_ratio:
            reasons.append("green_edge_ratio_high")

        blur_quality = float(np.clip(blur_score / max(self.min_blur_score * 3.0, 1.0), 0.0, 1.0))
        confidence_quality = confidence
        brightness_quality = 1.0
        if brightness < 0.08 or brightness > 0.96:
            brightness_quality = 0.25
            reasons.append("brightness_extreme")
        elif brightness < 0.16 or brightness > 0.90:
            brightness_quality = 0.65
        green_quality = float(np.clip(1.0 - green_edge_ratio, 0.0, 1.0))
        score = float(
            np.clip(
                0.30 * confidence_quality
                + 0.30 * blur_quality
                + 0.20 * brightness_quality
                + 0.20 * green_quality,
                0.0,
                1.0,
            )
        )
        accepted = not reasons and score >= self.min_quality_score
        if score < self.min_quality_score and "quality_score_low" not in reasons:
            reasons.append("quality_score_low")
        return QualityAssessment(
            accepted=accepted,
            score=score,
            reasons=tuple(reasons),
            blur_score=blur_score,
            brightness=brightness,
            green_edge_ratio=green_edge_ratio,
            detection_confidence=confidence,
        )
