from __future__ import annotations

import json

import cv2
import numpy as np
import torch

from app.field_registration.models import MobileNetV3PitchPerception
from app.field_registration.perception import PitchPerceptionVocabulary
from app.field_registration.pitch_model import PitchModel
from app.field_registration.torch_perception import TorchPitchPerceptionBackend
from experiments.field_registration.dataset import PitchRegistrationDataset
from experiments.field_registration.losses import dual_head_loss


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
