"""Backend-neutral field perception interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from collections.abc import Callable
from typing import Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from app.field_registration.types import LineObservation, PointObservation


@dataclass(frozen=True)
class PitchPerceptionOutput:
    points: Tuple[PointObservation, ...] = field(default_factory=tuple)
    lines: Tuple[LineObservation, ...] = field(default_factory=tuple)
    inference_time_ms: float = 0.0
    model_name: str = "unknown"


@runtime_checkable
class PitchPerceptionBackend(Protocol):
    def predict(self, frame: np.ndarray) -> PitchPerceptionOutput:
        """Return semantic metric pitch observations for one rectified frame."""


class StaticPerceptionBackend:
    """Deterministic backend for integration tests and recorded observations."""

    def __init__(self, outputs: Sequence[PitchPerceptionOutput]) -> None:
        self.outputs = tuple(outputs)
        self.index = 0

    def predict(self, frame: np.ndarray) -> PitchPerceptionOutput:
        del frame
        if not self.outputs:
            return PitchPerceptionOutput(model_name="static-empty")
        output = self.outputs[min(self.index, len(self.outputs) - 1)]
        self.index += 1
        return output


class LegacyKeypointPerceptionBackend:
    """Adapt the existing 32-keypoint model to metric CPT observations."""

    def __init__(
        self,
        predictor: Callable[[np.ndarray], object],
        references: Sequence[object],
        minimum_confidence: float = 0.35,
        edge_margin_px: float = 8.0,
        world_unit_scale_to_m: float = 0.01,
    ) -> None:
        self.predictor = predictor
        self.references = tuple(references)
        self.minimum_confidence = float(minimum_confidence)
        self.edge_margin_px = float(edge_margin_px)
        self.world_unit_scale_to_m = float(world_unit_scale_to_m)

    def predict(self, frame: np.ndarray) -> PitchPerceptionOutput:
        started = time.perf_counter()
        keypoints = self.predictor(frame)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        xy_batch = np.asarray(getattr(keypoints, "xy", np.empty((0, 0, 2))))
        if xy_batch.size == 0:
            return PitchPerceptionOutput(
                inference_time_ms=elapsed_ms,
                model_name="legacy-32-keypoint",
            )
        xy = xy_batch[0]
        confidence_value = getattr(keypoints, "keypoint_confidence", None)
        if confidence_value is None:
            confidence_value = getattr(keypoints, "confidence", None)
        confidence = (
            np.ones(xy.shape[0], dtype=np.float64)
            if confidence_value is None
            else np.asarray(confidence_value)[0]
        )
        height, width = frame.shape[:2]
        observations: list[PointObservation] = []
        for index, image_point in enumerate(xy):
            if index >= len(self.references) or index >= confidence.shape[0]:
                break
            score = float(confidence[index])
            x_coord, y_coord = float(image_point[0]), float(image_point[1])
            if score < self.minimum_confidence:
                continue
            if not (
                self.edge_margin_px <= x_coord < width - self.edge_margin_px
                and self.edge_margin_px <= y_coord < height - self.edge_margin_px
            ):
                continue
            reference = self.references[index]
            world_xy = getattr(reference, "world_xy")
            observations.append(
                PointObservation(
                    label=str(getattr(reference, "label", index)),
                    image_xy=(x_coord, y_coord),
                    pitch_xy_m=(
                        float(world_xy[0]) * self.world_unit_scale_to_m,
                        float(world_xy[1]) * self.world_unit_scale_to_m,
                    ),
                    confidence=score,
                    sigma_px=max(0.75, 3.0 * (1.0 - score)),
                    source="legacy_keypoint_model",
                )
            )
        return PitchPerceptionOutput(
            points=tuple(observations),
            inference_time_ms=elapsed_ms,
            model_name="legacy-32-keypoint",
        )
