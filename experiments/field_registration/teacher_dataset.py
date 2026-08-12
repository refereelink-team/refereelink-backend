"""Image + accepted teacher-cache dataset for distillation/domain adaptation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from app.field_registration.perception import PitchPerceptionVocabulary
from app.field_registration.pitch_model import PitchModel
from experiments.field_registration.augmentation import PitchTrainingAugmenter
from experiments.field_registration.soccernet import rasterize_teacher_targets
from experiments.field_registration.teacher_cache import load_teacher_frame


class TeacherDistillationDataset(Dataset[dict[str, torch.Tensor]]):
    """Load only pre-vetted teacher frames; no teacher runs inside workers."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        expected_split: str,
        input_size: tuple[int, int] = (512, 288),
        output_stride: int = 4,
        augmenter: PitchTrainingAugmenter | None = None,
    ) -> None:
        source = Path(index_path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if int(payload.get("format_version", 0)) != 1:
            raise ValueError("unsupported teacher distillation index")
        if str(payload.get("split")) != expected_split:
            raise ValueError(
                f"teacher index belongs to {payload.get('split')!r}, not {expected_split!r}"
            )
        root = Path(str(payload.get("root", ".")))
        self.root = root if root.is_absolute() else (source.parent / root).resolve()
        self.samples = tuple(payload.get("frames", ()))
        if not self.samples:
            raise ValueError("teacher distillation index has no frames")
        identities = [
            (str(item["source_id"]), int(item["frame_index"]))
            for item in self.samples
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("teacher distillation index contains duplicate frames")
        self.input_size = tuple(int(value) for value in input_size)
        self.output_stride = int(output_stride)
        if any(value % 32 for value in self.input_size):
            raise ValueError("input width and height must be divisible by 32")
        self.augmenter = augmenter
        self.pitch_model = PitchModel()
        self.vocabulary = PitchPerceptionVocabulary.from_pitch_model(self.pitch_model)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item: dict[str, Any] = self.samples[index]
        image_path = self.root / str(item["image_path"])
        teacher_path = self.root / str(item["teacher_cache_path"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        teacher = load_teacher_frame(teacher_path)
        if teacher.source_id != str(item["source_id"]):
            raise ValueError("teacher cache source_id does not match index")
        if teacher.frame_index != int(item["frame_index"]):
            raise ValueError("teacher cache frame_index does not match index")
        height, width = image.shape[:2]
        if teacher.image_size != (width, height):
            raise ValueError("teacher cache image size does not match image")

        source_to_augmented = np.eye(3, dtype=np.float64)
        if self.augmenter is not None:
            augmented = self.augmenter(image)
            image = augmented.image
            source_to_augmented = augmented.source_to_augmented
        input_width, input_height = self.input_size
        source_to_input = np.array(
            [
                [input_width / width, 0.0, 0.0],
                [0.0, input_height / height, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ) @ source_to_augmented
        targets = rasterize_teacher_targets(
            teacher,
            self.vocabulary,
            self.pitch_model,
            source_to_input,
            self.input_size,
            self.output_stride,
        )
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        landmark_count = len(self.vocabulary.landmark_labels)
        output_width = input_width // self.output_stride
        output_height = input_height // self.output_stride
        sample = {
            "image": (image_tensor - mean) / std,
            "semantic_target": torch.zeros(input_height, input_width, dtype=torch.long),
            "landmark_heatmaps": torch.zeros(
                landmark_count, output_height, output_width
            ),
            "landmark_offsets": torch.zeros(
                landmark_count * 2, output_height, output_width
            ),
            "offset_mask": torch.zeros(
                landmark_count, output_height, output_width
            ),
            "landmark_visibility": torch.zeros(landmark_count),
            "frame_index": torch.tensor(int(item["frame_index"]), dtype=torch.int64),
        }
        sample.update(targets)
        return sample
