from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.field_registration.annotations import (
    AnnotationFormatError,
    fit_annotation_homography,
    validate_annotation_payload,
)
from app.field_registration.geometry import transform_points
from app.field_registration.pitch_model import PitchModel


def _load_pack_generator():
    path = Path(__file__).parents[1] / "tools" / "create_pitch_annotation_pack.py"
    spec = importlib.util.spec_from_file_location("pitch_pack_generator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(tmp_path: Path) -> dict:
    image_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(image_path), np.zeros((480, 640, 3), dtype=np.uint8))
    pitch = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
    pitch_to_image = np.array(
        [[4.0, 0.2, 80.0], [0.1, 3.0, 100.0], [0.0002, 0.0001, 1.0]],
        dtype=np.float64,
    )
    image = transform_points(pitch, pitch_to_image)
    labels = [
        "left_top_corner",
        "right_top_corner",
        "right_bottom_corner",
        "left_bottom_corner",
    ]
    return {
        "version": 1,
        "split": "test",
        "image_size": [640, 480],
        "pitch_dimensions_m": {"length": 105.0, "width": 68.0},
        "frames": [
            {
                "frame_index": 5,
                "image_path": image_path.name,
                "correspondences": [
                    {"label": label, "image_xy": point.tolist()}
                    for label, point in zip(labels, image)
                ],
                "contact_points": [
                    {"contact_id": "player-7", "image_xy": [300.0, 400.0]}
                ],
            }
        ],
    }


def test_pitch_model_exposes_shared_annotation_geometry() -> None:
    model = PitchModel()

    assert len(model.landmarks) == 32
    assert "halfway_line" in model.semantic_elements()
    assert model.semantic_elements()["centre_circle"].shape == (96, 2)


def test_annotation_validator_fits_manual_correspondences(tmp_path) -> None:
    payload = _payload(tmp_path)

    report = validate_annotation_payload(payload, root=tmp_path)

    assert report.valid
    assert report.annotated_frame_count == 1
    assert report.contact_count == 1
    assert report.frames[0].homography_available
    assert report.frames[0].reprojection_median_px == pytest.approx(0.0, abs=1e-4)


def test_annotation_homography_falls_back_for_exact_regular_grid() -> None:
    pitch_x, pitch_y = np.meshgrid(np.linspace(0.0, 105.0, 6), np.linspace(0.0, 68.0, 5))
    pitch = np.column_stack((pitch_x.reshape(-1), pitch_y.reshape(-1)))
    transform = np.asarray(
        [[5.0, 0.2, 40.0], [0.1, 3.5, 30.0], [0.0005, 0.001, 1.0]],
        dtype=np.float64,
    )
    image = cv2.perspectiveTransform(
        pitch.reshape(-1, 1, 2), transform
    ).reshape(-1, 2)

    image_to_pitch, residual = fit_annotation_homography(image, pitch)

    assert image_to_pitch is not None
    assert residual == pytest.approx(0.0, abs=1e-4)


def test_annotation_validator_allows_geometry_only_rig_anchors(tmp_path) -> None:
    payload = _payload(tmp_path)
    payload["frames"][0]["contact_points"] = []

    rejected = validate_annotation_payload(payload, root=tmp_path)
    accepted = validate_annotation_payload(
        payload,
        root=tmp_path,
        require_contact_points=False,
    )

    assert not rejected.valid
    assert rejected.issues == ("no_contact_points",)
    assert accepted.valid
    assert accepted.contact_count == 0


def test_annotation_validator_rejects_unknown_landmark(tmp_path) -> None:
    payload = _payload(tmp_path)
    payload["frames"][0]["correspondences"][0]["label"] = "not_a_pitch_point"

    with pytest.raises(AnnotationFormatError, match="unknown pitch landmark"):
        validate_annotation_payload(payload, root=tmp_path)


def test_fit_annotation_homography_requires_four_points() -> None:
    matrix, residual = fit_annotation_homography(np.zeros((3, 2)), np.zeros((3, 2)))
    assert matrix is None
    assert residual is None


def test_annotation_pack_extracts_uniform_frames_and_ui(tmp_path) -> None:
    generator = _load_pack_generator()
    video_path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (160, 90),
    )
    assert writer.isOpened()
    for index in range(20):
        writer.write(np.full((90, 160, 3), index * 10, dtype=np.uint8))
    writer.release()

    output_path = tmp_path / "pack"
    manifest_path = generator.create_annotation_pack(
        video_path,
        output_path,
        requested_frames=5,
        split="validation",
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["split"] == "validation"
    assert payload["image_size"] == [160, 90]
    assert payload["source"]["reported_frame_count"] == 20
    assert payload["source"]["path_at_creation"] is None
    assert len(payload["frames"]) == 5
    assert [frame["frame_index"] for frame in payload["frames"]] == [0, 5, 10, 14, 19]
    assert all((output_path / frame["image_path"]).is_file() for frame in payload["frames"])
    assert (output_path / "index.html").is_file()
    empty_report = validate_annotation_payload(
        payload,
        root=output_path,
        require_annotated_frames=False,
        require_contact_points=False,
    )
    assert empty_report.valid


def test_annotation_pack_decoder_backs_off_from_unreadable_tail() -> None:
    generator = _load_pack_generator()

    class TailLimitedCapture:
        def __init__(self) -> None:
            self.index = 0

        def set(self, property_id: int, value: float) -> bool:
            assert property_id == cv2.CAP_PROP_POS_FRAMES
            self.index = int(value)
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.index >= 19:
                return False, None
            frame = np.full((4, 4, 3), self.index, dtype=np.uint8)
            self.index += 1
            return True, frame

    actual_index, frame = generator.decode_frame_with_backoff(
        TailLimitedCapture(),
        19,
    )

    assert actual_index == 18
    assert int(frame[0, 0, 0]) == 18


def test_annotation_pack_decoder_avoids_duplicate_fallback_frames() -> None:
    generator = _load_pack_generator()

    class TailLimitedCapture:
        def __init__(self) -> None:
            self.index = 0

        def set(self, property_id: int, value: float) -> bool:
            assert property_id == cv2.CAP_PROP_POS_FRAMES
            self.index = int(value)
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.index >= 19:
                return False, None
            frame = np.full((4, 4, 3), self.index, dtype=np.uint8)
            self.index += 1
            return True, frame

    actual_index, frame = generator.decode_unique_frame(
        TailLimitedCapture(),
        20,
        {18},
    )

    assert actual_index == 17
    assert int(frame[0, 0, 0]) == 17


def test_annotation_pack_uses_actual_decodable_frame_count() -> None:
    generator = _load_pack_generator()

    class TailLimitedCapture:
        def __init__(self) -> None:
            self.index = 0

        def set(self, property_id: int, value: float) -> bool:
            assert property_id == cv2.CAP_PROP_POS_FRAMES
            self.index = int(value)
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.index >= 19:
                return False, None
            frame = np.full((4, 4, 3), self.index, dtype=np.uint8)
            self.index += 1
            return True, frame

    count = generator.decodable_frame_count(TailLimitedCapture(), 21)

    assert count == 19


def test_annotation_pack_refuses_accidental_overwrite(tmp_path) -> None:
    generator = _load_pack_generator()
    video_path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (64, 48),
    )
    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    output_path = tmp_path / "pack"
    generator.create_annotation_pack(video_path, output_path, requested_frames=1)

    with pytest.raises(FileExistsError):
        generator.create_annotation_pack(video_path, output_path, requested_frames=1)
