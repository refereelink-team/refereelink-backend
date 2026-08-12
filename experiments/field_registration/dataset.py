"""Manifest-backed dual-head training targets without geometric augmentation drift."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from app.field_registration.annotations import (
    correspondence_arrays,
    fit_annotation_homography,
    pitch_dimensions_from_payload,
    validate_annotation_payload,
)
from app.field_registration.geometry import transform_points
from app.field_registration.perception import PitchPerceptionVocabulary
from app.field_registration.pitch_model import PitchModel
from experiments.field_registration.augmentation import (
    PitchTrainingAugmenter,
    morph_semantic_classes,
)


class PitchRegistrationDataset(Dataset[dict[str, torch.Tensor]]):
    """Generate dense line/landmark targets from manually aligned anchor frames."""

    def __init__(
        self,
        manifests: Sequence[str | Path],
        *,
        expected_split: str,
        input_size: tuple[int, int] = (512, 288),
        output_stride: int = 4,
        line_width_px: int = 3,
        photometric_transform: Callable[[np.ndarray], np.ndarray] | None = None,
        augmenter: PitchTrainingAugmenter | None = None,
    ) -> None:
        width, height = input_size
        if width % 32 or height % 32:
            raise ValueError("input width and height must be divisible by 32")
        if output_stride < 1 or width % output_stride or height % output_stride:
            raise ValueError("output_stride must divide the input dimensions")
        self.input_size = (int(width), int(height))
        self.output_stride = int(output_stride)
        self.line_width_px = int(line_width_px)
        self.photometric_transform = photometric_transform
        self.augmenter = augmenter
        self.samples: list[tuple[Path, dict, PitchModel, np.ndarray]] = []
        self.vocabulary: PitchPerceptionVocabulary | None = None
        self.pitch_model: PitchModel | None = None
        source_names: set[str] = set()

        for manifest_value in manifests:
            manifest_path = Path(manifest_value)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if str(payload.get("split")) != expected_split:
                raise ValueError(
                    f"{manifest_path} belongs to split {payload.get('split')!r}, "
                    f"not {expected_split!r}"
                )
            report = validate_annotation_payload(
                payload,
                root=manifest_path.parent,
                require_annotated_frames=False,
            )
            if not report.valid:
                raise ValueError(f"annotation validation failed for {manifest_path}")
            source_name = str(payload.get("source", {}).get("name", manifest_path.stem))
            if source_name in source_names:
                raise ValueError(
                    f"duplicate source sequence {source_name!r}; merge it before loading"
                )
            source_names.add(source_name)
            pitch_model = PitchModel(pitch_dimensions_from_payload(payload))
            vocabulary = PitchPerceptionVocabulary.from_pitch_model(pitch_model)
            if self.vocabulary is None:
                self.vocabulary = vocabulary
                self.pitch_model = pitch_model
            elif vocabulary != self.vocabulary:
                raise ValueError("all manifests must share one pitch vocabulary")
            elif pitch_model.dimensions != self.pitch_model.dimensions:
                raise ValueError("all manifests must share the same pitch dimensions")
            for frame in payload["frames"]:
                image_points, pitch_points = correspondence_arrays(frame, pitch_model)
                image_to_pitch, _ = fit_annotation_homography(
                    image_points, pitch_points
                )
                if image_to_pitch is None:
                    continue
                image_path = manifest_path.parent / frame["image_path"]
                self.samples.append(
                    (image_path, frame, pitch_model, np.linalg.inv(image_to_pitch))
                )
        if self.vocabulary is None or self.pitch_model is None or not self.samples:
            raise ValueError("no geometrically valid annotated frames were found")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _draw_gaussian(
        heatmap: np.ndarray,
        centre_xy: tuple[float, float],
        sigma: float = 1.5,
    ) -> None:
        x_coord, y_coord = centre_xy
        radius = max(2, int(round(3.0 * sigma)))
        x0 = max(0, int(np.floor(x_coord)) - radius)
        x1 = min(heatmap.shape[1], int(np.floor(x_coord)) + radius + 1)
        y0 = max(0, int(np.floor(y_coord)) - radius)
        y1 = min(heatmap.shape[0], int(np.floor(y_coord)) + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return
        grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
        gaussian = np.exp(
            -((grid_x - x_coord) ** 2 + (grid_y - y_coord) ** 2)
            / (2.0 * sigma**2)
        )
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], gaussian)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path, frame, pitch_model, pitch_to_image = self.samples[index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        source_height, source_width = image.shape[:2]
        input_width, input_height = self.input_size
        if self.photometric_transform is not None:
            image = self.photometric_transform(image)
        morphology = 0
        if self.augmenter is not None:
            augmented = self.augmenter(image)
            image = augmented.image
            pitch_to_image = augmented.source_to_augmented @ pitch_to_image
            morphology = augmented.line_morphology
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        scale = np.array(
            [
                [input_width / source_width, 0.0, 0.0],
                [0.0, input_height / source_height, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        resized_pitch_to_image = scale @ pitch_to_image

        semantic_target = np.zeros((input_height, input_width), dtype=np.int32)
        for class_index, label in enumerate(
            self.vocabulary.semantic_labels, start=1
        ):
            projected = transform_points(
                pitch_model.semantic_elements()[label], resized_pitch_to_image
            )
            finite = np.all(np.isfinite(projected), axis=1)
            points = np.rint(projected[finite]).astype(np.int32)
            if points.shape[0] >= 2:
                cv2.polylines(
                    semantic_target,
                    [points.reshape(-1, 1, 2)],
                    isClosed=label == "centre_circle",
                    color=class_index,
                    thickness=max(1, self.line_width_px),
                    lineType=cv2.LINE_8,
                )
        semantic_target = morph_semantic_classes(semantic_target, morphology)

        output_width = input_width // self.output_stride
        output_height = input_height // self.output_stride
        landmark_count = len(self.vocabulary.landmark_labels)
        heatmaps = np.zeros((landmark_count, output_height, output_width), dtype=np.float32)
        offsets = np.zeros((landmark_count, 2, output_height, output_width), dtype=np.float32)
        offset_mask = np.zeros((landmark_count, output_height, output_width), dtype=np.float32)
        landmark_map = pitch_model.landmarks | pitch_model.point_landmarks
        for landmark_index, label in enumerate(self.vocabulary.landmark_labels):
            image_xy = transform_points(
                np.asarray([landmark_map[label]], dtype=np.float64),
                resized_pitch_to_image,
            )[0]
            output_xy = image_xy / self.output_stride
            if not (
                0.0 <= output_xy[0] < output_width
                and 0.0 <= output_xy[1] < output_height
            ):
                continue
            self._draw_gaussian(heatmaps[landmark_index], tuple(output_xy))
            integer_xy = np.rint(output_xy).astype(int)
            integer_xy[0] = np.clip(integer_xy[0], 0, output_width - 1)
            integer_xy[1] = np.clip(integer_xy[1], 0, output_height - 1)
            heatmaps[landmark_index, integer_xy[1], integer_xy[0]] = 1.0
            offsets[landmark_index, :, integer_xy[1], integer_xy[0]] = (
                output_xy - integer_xy
            )
            offset_mask[landmark_index, integer_xy[1], integer_xy[0]] = 1.0

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        return {
            "image": (image_tensor - mean) / std,
            "semantic_target": torch.from_numpy(semantic_target).long(),
            "landmark_heatmaps": torch.from_numpy(heatmaps),
            "landmark_offsets": torch.from_numpy(offsets).reshape(
                landmark_count * 2, output_height, output_width
            ),
            "offset_mask": torch.from_numpy(offset_mask),
            "landmark_visibility": torch.from_numpy(
                (offset_mask.reshape(landmark_count, -1).max(axis=1) > 0).astype(
                    np.float32
                )
            ),
            "frame_index": torch.tensor(int(frame["frame_index"]), dtype=torch.int64),
        }
