"""Geometry-synchronised broadcast augmentation for pitch perception."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PitchAugmentationConfig:
    minimum_crop_ratio: float = 0.72
    maximum_vertical_shift_ratio: float = 0.08
    brightness_range: tuple[float, float] = (0.75, 1.25)
    gamma_range: tuple[float, float] = (0.75, 1.35)
    shadow_probability: float = 0.35
    blur_probability: float = 0.25
    jpeg_probability: float = 0.35
    occlusion_probability: float = 0.35
    maximum_occlusions: int = 4


@dataclass(frozen=True)
class AugmentedPitchImage:
    image: np.ndarray
    source_to_augmented: np.ndarray
    line_morphology: int


class PitchTrainingAugmenter:
    """Simulate Pan/Zoom and broadcast degradation with one shared geometry.

    The operation is affine crop-and-resize only. It intentionally excludes
    unsynchronised perspective warps, which would make pitch labels incorrect.
    """

    def __init__(
        self,
        config: PitchAugmentationConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.config = config or PitchAugmentationConfig()
        if not 0.5 <= self.config.minimum_crop_ratio <= 1.0:
            raise ValueError("minimum_crop_ratio must be between 0.5 and 1")
        self._rng = np.random.default_rng(seed)

    def _crop_and_resize(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        ratio = float(self._rng.uniform(self.config.minimum_crop_ratio, 1.0))
        crop_width = max(int(round(width * ratio)), 2)
        crop_height = max(int(round(height * ratio)), 2)
        left = int(self._rng.integers(0, max(width - crop_width + 1, 1)))
        maximum_vertical = min(
            max(height - crop_height, 0),
            int(round(height * self.config.maximum_vertical_shift_ratio)),
        )
        top = int(self._rng.integers(0, max(maximum_vertical + 1, 1)))
        cropped = image[top : top + crop_height, left : left + crop_width]
        resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
        transform = np.array(
            [
                [width / crop_width, 0.0, -left * width / crop_width],
                [0.0, height / crop_height, -top * height / crop_height],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return resized, transform

    def _photometric(self, image: np.ndarray) -> np.ndarray:
        values = image.astype(np.float32) / 255.0
        brightness = float(self._rng.uniform(*self.config.brightness_range))
        gamma = float(self._rng.uniform(*self.config.gamma_range))
        values = np.clip(values * brightness, 0.0, 1.0) ** gamma
        if self._rng.random() < self.config.shadow_probability:
            height, width = values.shape[:2]
            x0 = int(self._rng.integers(-width // 2, width))
            x1 = int(self._rng.integers(max(x0 + 1, 1), width + width // 2))
            shadow = np.ones((height, width), dtype=np.float32)
            polygon = np.asarray(
                [[x0, 0], [x1, 0], [x1 - width // 4, height], [x0 - width // 4, height]],
                dtype=np.int32,
            )
            cv2.fillPoly(shadow, [polygon], float(self._rng.uniform(0.45, 0.8)))
            values *= shadow[..., None]
        output = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
        if self._rng.random() < self.config.blur_probability:
            kernel = int(self._rng.choice((3, 5)))
            output = cv2.GaussianBlur(output, (kernel, kernel), 0)
        if self._rng.random() < self.config.jpeg_probability:
            quality = int(self._rng.integers(45, 91))
            success, encoded = cv2.imencode(
                ".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            if success:
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if decoded is not None:
                    output = decoded
        return output

    def _occlude(self, image: np.ndarray) -> np.ndarray:
        if self._rng.random() >= self.config.occlusion_probability:
            return image
        output = image.copy()
        height, width = output.shape[:2]
        count = int(self._rng.integers(1, self.config.maximum_occlusions + 1))
        for _ in range(count):
            box_width = int(self._rng.uniform(0.02, 0.08) * width)
            box_height = int(self._rng.uniform(0.08, 0.28) * height)
            left = int(self._rng.integers(0, max(width - box_width, 1)))
            top = int(self._rng.integers(int(height * 0.15), max(height - box_height, 1)))
            colour = tuple(int(value) for value in self._rng.integers(20, 220, size=3))
            cv2.rectangle(
                output,
                (left, top),
                (left + box_width, top + box_height),
                colour,
                -1,
            )
        return output

    def __call__(self, image: np.ndarray) -> AugmentedPitchImage:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("augmentation expects a BGR image")
        augmented, transform = self._crop_and_resize(image)
        augmented = self._photometric(augmented)
        augmented = self._occlude(augmented)
        morphology = int(self._rng.choice((-1, 0, 0, 0, 1)))
        return AugmentedPitchImage(augmented, transform, morphology)


def morph_semantic_classes(target: np.ndarray, operation: int) -> np.ndarray:
    """Erode/dilate each foreground class without merging class identities."""

    if operation == 0:
        return target
    kernel = np.ones((3, 3), dtype=np.uint8)
    result = np.zeros_like(target)
    for class_index in range(1, int(target.max()) + 1):
        mask = (target == class_index).astype(np.uint8)
        transformed = (
            cv2.dilate(mask, kernel, iterations=1)
            if operation > 0
            else cv2.erode(mask, kernel, iterations=1)
        )
        result[transformed.astype(bool)] = class_index
    return result
