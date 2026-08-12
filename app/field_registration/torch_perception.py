"""PyTorch inference adapter for dual-head pitch perception models."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.nn import functional as functional

from app.field_registration.decode import decode_heatmap_peak, sample_semantic_mask
from app.field_registration.perception import (
    PitchPerceptionOutput,
    PitchPerceptionVocabulary,
)
from app.field_registration.pitch_model import PitchModel
from app.field_registration.types import LineObservation, PointObservation


def load_pitch_perception_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a versioned dual-head checkpoint without accepting partial weights."""

    payload = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True
    )
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported pitch-perception checkpoint format")
    required = {
        "architecture",
        "input_size",
        "semantic_labels",
        "landmark_labels",
        "state_dict",
        "validation",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(
            f"checkpoint is incomplete: missing {', '.join(missing)}"
        )
    input_size = payload["input_size"]
    if not isinstance(input_size, (list, tuple)) or len(input_size) != 2:
        raise ValueError("checkpoint input_size must be [width, height]")
    if min(int(value) for value in input_size) <= 0:
        raise ValueError("checkpoint input_size must be positive")

    from app.field_registration.models import build_pitch_perception_model

    model = build_pitch_perception_model(
        str(payload["architecture"]),
        len(payload["semantic_labels"]) + 1,
        len(payload["landmark_labels"]),
        pretrained=False,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval().to(device), payload


class TorchPitchPerceptionBackend:
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        pitch_model: PitchModel,
        *,
        device: str = "cpu",
        use_fp16: bool = True,
        point_threshold: float = 0.35,
        line_threshold: float = 0.45,
    ) -> "TorchPitchPerceptionBackend":
        torch_device = torch.device(device)
        model, payload = load_pitch_perception_checkpoint(
            checkpoint_path, torch_device
        )
        expected = PitchPerceptionVocabulary.from_pitch_model(pitch_model)
        if tuple(payload["semantic_labels"]) != expected.semantic_labels:
            raise ValueError(
                "checkpoint semantic vocabulary does not match the pitch model"
            )
        if tuple(payload["landmark_labels"]) != expected.landmark_labels:
            raise ValueError(
                "checkpoint landmark vocabulary does not match the pitch model"
            )
        stored_dimensions = payload.get("pitch_dimensions_m")
        expected_dimensions = asdict(pitch_model.dimensions)
        if stored_dimensions is not None:
            try:
                dimensions_match = all(
                    np.isclose(
                        float(stored_dimensions[name]),
                        float(expected_value),
                        atol=1e-3,
                    )
                    for name, expected_value in expected_dimensions.items()
                )
            except (KeyError, TypeError, ValueError):
                dimensions_match = False
            if not dimensions_match:
                raise ValueError(
                    "checkpoint pitch dimensions do not match the runtime pitch model"
                )
        input_size = tuple(int(value) for value in payload["input_size"])
        return cls(
            model,
            pitch_model,
            device=device,
            input_size=input_size,
            use_fp16=use_fp16,
            point_threshold=point_threshold,
            line_threshold=line_threshold,
            model_name=str(payload["architecture"]),
        )

    def __init__(
        self,
        model: torch.nn.Module,
        pitch_model: PitchModel,
        *,
        device: str = "cpu",
        input_size: Tuple[int, int] = (512, 288),
        use_fp16: bool = True,
        point_threshold: float = 0.35,
        line_threshold: float = 0.45,
        model_name: str = "dual-head",
    ) -> None:
        width, height = input_size
        if width <= 0 or height <= 0:
            raise ValueError("input_size must be positive")
        self.model = model.eval().to(device)
        self.pitch_model = pitch_model
        self.vocabulary = PitchPerceptionVocabulary.from_pitch_model(pitch_model)
        self.device = torch.device(device)
        self.input_size = (int(width), int(height))
        self.use_fp16 = bool(use_fp16 and self.device.type == "cuda")
        self.point_threshold = float(point_threshold)
        self.line_threshold = float(line_threshold)
        self.model_name = model_name
        self._mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self._std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image")
        resized = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = (tensor - self._mean) / self._std
        return tensor.to(self.device, non_blocking=True)

    @torch.inference_mode()
    def predict(self, frame: np.ndarray) -> PitchPerceptionOutput:
        started = time.perf_counter()
        tensor = self._preprocess(frame)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_fp16,
        ):
            semantic_logits, landmark_logits, offset_logits = self.model(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        semantic = functional.softmax(semantic_logits.float(), dim=1)[0].cpu().numpy()
        landmarks = torch.sigmoid(landmark_logits.float())[0].cpu().numpy()
        offsets = (torch.tanh(offset_logits.float()) * 0.5)[0].cpu().numpy()
        frame_height, frame_width = frame.shape[:2]
        input_width, input_height = self.input_size

        line_observations: list[LineObservation] = []
        pitch_elements = self.pitch_model.semantic_elements()
        for class_index, label in enumerate(self.vocabulary.semantic_labels, start=1):
            image_points = sample_semantic_mask(
                semantic[class_index], threshold=self.line_threshold
            )
            if image_points.shape[0] < 2:
                continue
            image_points[:, 0] *= frame_width / input_width
            image_points[:, 1] *= frame_height / input_height
            mask_values = semantic[class_index]
            confidence = float(np.mean(mask_values[mask_values >= self.line_threshold]))
            line_observations.append(
                LineObservation(
                    label=label,
                    image_points=image_points,
                    pitch_points_m=pitch_elements[label],
                    confidence=confidence,
                    source=self.model_name,
                )
            )

        point_observations: list[PointObservation] = []
        pitch_landmarks = self.pitch_model.landmarks | self.pitch_model.point_landmarks
        for landmark_index, (heatmap, label) in enumerate(
            zip(landmarks, self.vocabulary.landmark_labels)
        ):
            heatmap_xy, confidence = decode_heatmap_peak(heatmap)
            if confidence < self.point_threshold:
                continue
            peak_x = int(np.clip(round(heatmap_xy[0]), 0, heatmap.shape[1] - 1))
            peak_y = int(np.clip(round(heatmap_xy[1]), 0, heatmap.shape[0] - 1))
            offset = offsets[
                landmark_index * 2 : landmark_index * 2 + 2,
                peak_y,
                peak_x,
            ]
            refined_xy = np.asarray(heatmap_xy) + offset
            image_xy = (
                float((refined_xy[0] + 0.5) * frame_width / heatmap.shape[1] - 0.5),
                float((refined_xy[1] + 0.5) * frame_height / heatmap.shape[0] - 0.5),
            )
            point_observations.append(
                PointObservation(
                    label=label,
                    image_xy=image_xy,
                    pitch_xy_m=pitch_landmarks[label],
                    confidence=confidence,
                    sigma_px=max(0.75, 4.0 * (1.0 - confidence)),
                    source=self.model_name,
                )
            )
        return PitchPerceptionOutput(
            points=tuple(point_observations),
            lines=tuple(line_observations),
            inference_time_ms=elapsed_ms,
            model_name=self.model_name,
        )
