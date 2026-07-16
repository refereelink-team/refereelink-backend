"""Optional Ultralytics adapter for track-level player role prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


class UltralyticsRoleClassifier:
    """Convert a role-aware YOLO checkpoint into a crop classifier interface.

    A missing checkpoint is a normal deployment state: ``predict`` returns
    UNKNOWN labels and zero confidence, allowing the rest of the pipeline to
    remain usable with the official person-only model.
    """

    def __init__(self, model_path: str, device: str = "cpu", imgsz: int = 640, model: object = None):
        self.model_path = model_path
        self.device = device
        self.imgsz = max(int(imgsz), 32)
        self._model = model

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        if self._model is not None:
            return True
        if not Path(self.model_path).exists():
            return False
        from ultralytics import YOLO

        self._model = YOLO(self.model_path).to(device=self.device)
        return True

    def predict_with_confidence(
        self, crops: Sequence[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.full(len(crops), "unknown", dtype=object)
        confidences = np.zeros(len(crops), dtype=np.float32)
        if self._model is None:
            return values, confidences

        for index, crop in enumerate(crops):
            if np.asarray(crop).size == 0:
                continue
            try:
                result = self._model(crop, imgsz=self.imgsz, verbose=False)[0]
                boxes = getattr(result, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    continue
                scores = np.asarray(boxes.conf.detach().cpu().numpy()).reshape(-1)
                classes = np.asarray(boxes.cls.detach().cpu().numpy()).reshape(-1)
                best = int(np.argmax(scores))
                class_id = int(classes[best])
                names = getattr(result, "names", {})
                values[index] = names.get(class_id, class_id) if isinstance(names, dict) else class_id
                confidences[index] = float(np.clip(scores[best], 0.0, 1.0))
            except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
                continue
        return values, confidences

    def predict(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        values, _ = self.predict_with_confidence(crops)
        return values
