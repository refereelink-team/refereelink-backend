from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from app.field_registration.models import (
    MobileNetV3PitchPerception,
    build_pitch_perception_model,
)
from app.field_registration.perception import PitchPerceptionVocabulary
from app.field_registration.pitch_model import PitchModel
from app.field_registration.torch_perception import TorchPitchPerceptionBackend
from experiments.field_registration.dataset import PitchRegistrationDataset
from experiments.field_registration.augmentation import (
    PitchAugmentationConfig,
    PitchTrainingAugmenter,
    morph_semantic_classes,
)
from experiments.field_registration.losses import dual_head_loss, heatmap_focal_loss
from experiments.field_registration.soccernet import rasterize_teacher_targets
from experiments.field_registration.teacher_cache import TeacherFrame
from experiments.field_registration.teacher_dataset import TeacherDistillationDataset
from tools.create_pitch_training_smoke_fixture import create_fixture
from app.field_registration.types import LineObservation, PointObservation


def test_pitch_perception_vocabulary_is_stable_and_semantic() -> None:
    vocabulary = PitchPerceptionVocabulary.from_pitch_model(PitchModel())

    assert vocabulary.semantic_class_count == 21
    assert len(vocabulary.landmark_labels) == 33
    assert vocabulary.semantic_labels == tuple(sorted(vocabulary.semantic_labels))
    assert "halfway_line" in vocabulary.semantic_labels
    assert "centre_spot" in vocabulary.landmark_labels


def test_mobilenet_dual_head_shapes_and_losses() -> None:
    model = MobileNetV3PitchPerception(21, 33, pretrained=False).eval()
    images = torch.randn(2, 3, 64, 128)

    with torch.inference_mode():
        outputs = model(images)

    assert outputs[0].shape == (2, 21, 64, 128)
    assert outputs[1].shape == (2, 33, 16, 32)
    assert outputs[2].shape == (2, 66, 16, 32)
    batch = {
        "semantic_target": torch.zeros(2, 64, 128, dtype=torch.long),
        "landmark_heatmaps": torch.zeros(2, 33, 16, 32),
        "landmark_offsets": torch.zeros(2, 66, 16, 32),
        "offset_mask": torch.zeros(2, 33, 16, 32),
    }
    batch["landmark_heatmaps"][:, :, 8, 16] = 1.0
    batch["offset_mask"][:, :, 8, 16] = 1.0
    loss, components = dual_head_loss(outputs, batch)

    assert torch.isfinite(loss)
    assert components["total"] > 0.0


def test_manifest_dataset_generates_line_and_landmark_targets(tmp_path) -> None:
    image_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(image_path), np.zeros((180, 320, 3), dtype=np.uint8))
    pitch_points = np.array(
        [[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]],
        dtype=np.float64,
    )
    image_points = np.array(
        [[20.0, 20.0], [300.0, 30.0], [290.0, 165.0], [30.0, 160.0]],
        dtype=np.float64,
    )
    payload = {
        "version": 1,
        "split": "train",
        "source": {"name": "synthetic-sequence"},
        "image_size": [320, 180],
        "pitch_dimensions_m": {"length": 105.0, "width": 68.0},
        "frames": [
            {
                "frame_index": 0,
                "image_path": image_path.name,
                "correspondences": [
                    {
                        "label": f"corner-{index}",
                        "image_xy": image.tolist(),
                        "pitch_xy_m": pitch.tolist(),
                    }
                    for index, (image, pitch) in enumerate(
                        zip(image_points, pitch_points)
                    )
                ],
                "contact_points": [],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    dataset = PitchRegistrationDataset(
        [manifest_path],
        expected_split="train",
        input_size=(128, 64),
    )

    sample = dataset[0]

    assert sample["image"].shape == (3, 64, 128)
    assert sample["semantic_target"].shape == (64, 128)
    assert int(sample["semantic_target"].max()) > 0
    assert sample["landmark_heatmaps"].shape == (33, 16, 32)
    assert float(sample["landmark_heatmaps"].max()) == 1.0
    assert int(sample["offset_mask"].sum()) > 0
    assert 0 < int(sample["landmark_visibility"].sum()) <= 33


def test_manifest_dataset_rejects_mixed_pitch_dimensions(tmp_path) -> None:
    paths = create_fixture(tmp_path)
    first = Path(paths["train_manifest"])
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["source"]["name"] = "different-pitch-sequence"
    payload["pitch_dimensions_m"]["length"] = 110.0
    second = tmp_path / "different-pitch.json"
    second.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="same pitch dimensions"):
        PitchRegistrationDataset(
            [first, second], expected_split="train", input_size=(128, 64)
        )


def test_invisible_landmark_channels_do_not_contribute_focal_loss() -> None:
    targets = torch.zeros(1, 2, 8, 8)
    targets[0, 0, 3, 4] = 1.0
    visibility = torch.tensor([[1.0, 0.0]])
    first = torch.zeros_like(targets)
    second = first.clone()
    second[:, 1] = 50.0

    first_loss = heatmap_focal_loss(first, targets, visibility)
    second_loss = heatmap_focal_loss(second, targets, visibility)

    assert second_loss == pytest.approx(float(first_loss), abs=1e-7)


def test_geometry_augmentation_transform_matches_visible_marker() -> None:
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    marker = np.asarray([120.0, 70.0])
    cv2.circle(image, tuple(marker.astype(int)), 5, (255, 255, 255), -1)
    augmenter = PitchTrainingAugmenter(
        PitchAugmentationConfig(
            brightness_range=(1.0, 1.0),
            gamma_range=(1.0, 1.0),
            shadow_probability=0.0,
            blur_probability=0.0,
            jpeg_probability=0.0,
            occlusion_probability=0.0,
        ),
        seed=13,
    )

    result = augmenter(image)
    transformed = cv2.perspectiveTransform(
        marker.reshape(1, 1, 2), result.source_to_augmented
    ).reshape(2)
    grayscale = cv2.cvtColor(result.image, cv2.COLOR_BGR2GRAY)
    marker_pixels = np.column_stack(np.nonzero(grayscale >= 200))[:, ::-1]
    observed_centre = np.mean(marker_pixels, axis=0)

    assert observed_centre == pytest.approx(tuple(transformed), abs=1.5)


def test_semantic_morphology_preserves_class_identity() -> None:
    target = np.zeros((20, 20), dtype=np.int32)
    target[5:15, 4] = 1
    target[5:15, 15] = 2

    dilated = morph_semantic_classes(target, 1)

    assert set(np.unique(dilated)) == {0, 1, 2}
    assert np.count_nonzero(dilated == 1) > np.count_nonzero(target == 1)


def test_teacher_targets_and_distillation_loss_are_finite() -> None:
    pitch_model = PitchModel()
    vocabulary = PitchPerceptionVocabulary.from_pitch_model(pitch_model)
    halfway_pitch = pitch_model.semantic_elements()["halfway_line"]
    halfway_image = halfway_pitch * np.asarray([2.0, 2.0]) + np.asarray([20.0, 10.0])
    teacher = TeacherFrame(
        teacher_id="synthetic",
        teacher_version="1",
        source_id="frame",
        frame_index=0,
        image_size=(256, 160),
        confidence=0.9,
        points=(
            PointObservation(
                "centre_spot",
                (125.0, 78.0),
                pitch_model.point_landmarks["centre_spot"],
                confidence=0.9,
            ),
        ),
        lines=(
            LineObservation(
                "halfway_line",
                halfway_image,
                halfway_pitch,
                confidence=0.9,
            ),
        ),
    )
    teacher_targets = rasterize_teacher_targets(
        teacher,
        vocabulary,
        pitch_model,
        np.eye(3),
        (256, 160),
        4,
    )
    outputs = (
        torch.zeros(1, 21, 160, 256),
        torch.zeros(1, 33, 40, 64),
        torch.zeros(1, 66, 40, 64),
    )
    batch = {
        "semantic_target": torch.zeros(1, 160, 256, dtype=torch.long),
        "landmark_heatmaps": torch.zeros(1, 33, 40, 64),
        "landmark_offsets": torch.zeros(1, 66, 40, 64),
        "offset_mask": torch.zeros(1, 33, 40, 64),
        "landmark_visibility": torch.zeros(1, 33),
    }
    batch.update({key: value.unsqueeze(0) for key, value in teacher_targets.items()})

    loss, components = dual_head_loss(
        outputs,
        batch,
        semantic_weight=0.0,
        semantic_dice_weight=0.0,
        heatmap_weight=0.0,
        offset_weight=0.0,
        semantic_distillation_weight=1.0,
        landmark_distillation_weight=1.0,
    )

    assert torch.isfinite(loss)
    assert components["semantic_distillation"] > 0.0
    assert components["landmark_distillation"] > 0.0


def test_three_phase_smoke_fixture_loads_all_datasets(tmp_path) -> None:
    paths = create_fixture(tmp_path)

    supervised = PitchRegistrationDataset(
        [paths["train_manifest"]], expected_split="train", input_size=(128, 64)
    )
    distillation = TeacherDistillationDataset(
        paths["distillation_index"],
        expected_split="distillation",
        input_size=(128, 64),
    )
    domain = TeacherDistillationDataset(
        paths["domain_index"], expected_split="domain", input_size=(128, 64)
    )

    assert len(supervised) == 2
    assert distillation[0]["teacher_semantic_mask"].sum() > 0
    assert domain[0]["teacher_landmark_visibility"].sum() > 0


def test_torch_backend_decodes_semantic_line_and_subpixel_offset() -> None:
    pitch_model = PitchModel()
    vocabulary = PitchPerceptionVocabulary.from_pitch_model(pitch_model)

    class FakeDualHead(torch.nn.Module):
        def forward(self, images):
            batch, _, height, width = images.shape
            semantic = torch.full((batch, 21, height, width), -8.0, device=images.device)
            semantic[:, 0] = 0.0
            line_index = vocabulary.semantic_labels.index("halfway_line") + 1
            semantic[:, line_index, :, width // 2 - 1 : width // 2 + 1] = 8.0
            heatmaps = torch.full((batch, 33, height // 4, width // 4), -8.0, device=images.device)
            landmark_index = vocabulary.landmark_labels.index("centre_spot")
            heatmaps[:, landmark_index, height // 8, width // 8] = 8.0
            offsets = torch.zeros((batch, 66, height // 4, width // 4), device=images.device)
            offsets[:, landmark_index * 2, height // 8, width // 8] = 0.4
            return semantic, heatmaps, offsets

    backend = TorchPitchPerceptionBackend(
        FakeDualHead(),
        pitch_model,
        device="cpu",
        input_size=(128, 64),
    )
    output = backend.predict(np.zeros((128, 256, 3), dtype=np.uint8))

    assert [line.label for line in output.lines] == ["halfway_line"]
    assert [point.label for point in output.points] == ["centre_spot"]
    assert output.points[0].image_xy[0] > 128.0


def _checkpoint(path: Path, pitch_model: PitchModel) -> None:
    vocabulary = PitchPerceptionVocabulary.from_pitch_model(pitch_model)
    model = build_pitch_perception_model(
        "mobilenet_v3_dual_head",
        vocabulary.semantic_class_count,
        len(vocabulary.landmark_labels),
        pretrained=False,
    )
    torch.save(
        {
            "format_version": 1,
            "architecture": "mobilenet_v3_dual_head",
            "input_size": [128, 64],
            "semantic_labels": list(vocabulary.semantic_labels),
            "landmark_labels": list(vocabulary.landmark_labels),
            "pitch_dimensions_m": asdict(pitch_model.dimensions),
            "state_dict": model.state_dict(),
            "validation": {"accuracy_valid": False},
        },
        path,
    )


def test_torch_backend_loads_versioned_checkpoint(tmp_path) -> None:
    pitch_model = PitchModel()
    checkpoint = tmp_path / "student.pt"
    _checkpoint(checkpoint, pitch_model)

    backend = TorchPitchPerceptionBackend.from_checkpoint(
        checkpoint, pitch_model, device="cpu", use_fp16=False
    )

    assert backend.input_size == (128, 64)
    assert backend.model_name == "mobilenet_v3_dual_head"


def test_torch_backend_rejects_vocabulary_mismatch(tmp_path) -> None:
    pitch_model = PitchModel()
    checkpoint = tmp_path / "student.pt"
    _checkpoint(checkpoint, pitch_model)
    payload = torch.load(checkpoint, weights_only=True)
    payload["semantic_labels"][0] = "wrong-label"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="semantic vocabulary"):
        TorchPitchPerceptionBackend.from_checkpoint(
            checkpoint, pitch_model, device="cpu"
        )


def test_torch_backend_rejects_pitch_dimension_mismatch(tmp_path) -> None:
    pitch_model = PitchModel()
    checkpoint = tmp_path / "student.pt"
    _checkpoint(checkpoint, pitch_model)
    payload = torch.load(checkpoint, weights_only=True)
    payload["pitch_dimensions_m"]["length_m"] = 120.0
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="pitch dimensions"):
        TorchPitchPerceptionBackend.from_checkpoint(
            checkpoint, pitch_model, device="cpu"
        )
