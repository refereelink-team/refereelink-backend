"""Ball detection and bounded kinematic tracking.

The player/pitch path lives in :mod:`app.vision.core`; the ball keeps a small
separate adapter because its detector and its missing-detection behavior are
different.  The adapter is model-agnostic and can be driven by a fake detector
in tests.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import supervision as sv

from app.tracking.ball import BallTrackEstimate, KinematicBallTracker


DetectionFn = Callable[[np.ndarray], sv.Detections]


def _empty_detections() -> sv.Detections:
    return sv.Detections(xyxy=np.empty((0, 4), dtype=np.float32))


class BallProcessor:
    """Run a ball detector at a fixed interval and predict between detections."""

    def __init__(
        self,
        model_path: str = "assets/weights/football-ball-detection.pt",
        device: str = "cpu",
        imgsz: int = 640,
        detection_interval: int = 2,
        max_prediction_frames: int = 8,
        inference_backend: str = "auto",
        model: object = None,
        detector: Optional[DetectionFn] = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.imgsz = max(int(imgsz), 32)
        self.detection_interval = max(int(detection_interval), 1)
        self.inference_backend = inference_backend
        self._model = model
        self._detector = detector
        self._tracker = KinematicBallTracker(max_prediction_frames=max_prediction_frames)
        self.frames_processed = 0
        self.detection_count = 0
        self.predicted_frames = 0
        self.available_frames = 0
        self.inference_time_ms = 0.0

    @property
    def tracker(self) -> KinematicBallTracker:
        return self._tracker

    def load_model(self) -> None:
        if self._detector is not None or self._model is not None:
            return
        from app.vision.backends import UltralyticsBackend

        backend = None if self.inference_backend == "auto" else self.inference_backend
        self._model = UltralyticsBackend(
            self.model_path,
            backend=backend,
            device=self.device,
        )

    def _predict(self, frame: np.ndarray) -> sv.Detections:
        if self._detector is not None:
            detections = self._detector(frame)
            return detections if detections is not None else _empty_detections()
        if self._model is None:
            return _empty_detections()
        use_fp16 = self.device.startswith("cuda")
        start = time.perf_counter()
        if hasattr(self._model, "predict"):
            result = self._model.predict(frame, imgsz=self.imgsz, half=use_fp16)[0]
        else:
            try:
                result = self._model(
                    frame,
                    imgsz=self.imgsz,
                    verbose=False,
                    half=use_fp16,
                )[0]
            except TypeError:
                try:
                    result = self._model(frame, imgsz=self.imgsz, verbose=False)[0]
                except TypeError:
                    result = self._model(frame)[0]
        self.inference_time_ms += (time.perf_counter() - start) * 1000.0
        return sv.Detections.from_ultralytics(result)

    @staticmethod
    def _best_detection(detections: sv.Detections) -> tuple[Optional[np.ndarray], float]:
        if len(detections) == 0:
            return None, 0.0
        centers = detections.get_anchors_coordinates(sv.Position.CENTER).astype(np.float32)
        if detections.confidence is None:
            index = 0
            confidence = 0.0
        else:
            confidence_values = np.asarray(detections.confidence, dtype=np.float32)
            finite = np.isfinite(confidence_values)
            if not finite.any():
                index = 0
                confidence = 0.0
            else:
                safe_values = np.where(finite, confidence_values, -np.inf)
                index = int(np.argmax(safe_values))
                confidence = float(confidence_values[index])
        return centers[index], confidence

    def process(self, frame: np.ndarray, frame_index: int, timestamp_s: Optional[float] = None) -> BallTrackEstimate:
        should_detect = (
            self.frames_processed == 0
            or frame_index % self.detection_interval == 0
        )
        position: Optional[np.ndarray] = None
        confidence = 0.0
        if should_detect:
            detections = self._predict(frame)
            self.detection_count += 1
            position, confidence = self._best_detection(detections)

        estimate = self._tracker.update(
            position=position,
            confidence=confidence,
            timestamp_s=timestamp_s,
        )
        self.frames_processed += 1
        if estimate.status == "predicted":
            self.predicted_frames += 1
        if estimate.status in {"fresh", "predicted"}:
            self.available_frames += 1
        return estimate
