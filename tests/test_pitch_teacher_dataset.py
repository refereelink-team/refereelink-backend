from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from app.field_registration.pitch_model import PitchModel
from app.field_registration.types import LineObservation, PointObservation
from experiments.field_registration.teacher_cache import TeacherFrame, save_teacher_frame
from experiments.field_registration.teacher_dataset import TeacherDistillationDataset
from tools.build_teacher_distillation_index import build_index


def _teacher(source_id: str = "test1/frame-0") -> TeacherFrame:
    pitch = PitchModel()
    pitch_line = pitch.semantic_elements()["halfway_line"]
    image_line = pitch_line * np.asarray([2.0, 2.0]) + np.asarray([20.0, 10.0])
    return TeacherFrame(
        teacher_id="teacher-consensus",
        teacher_version="pnl+tv",
        source_id=source_id,
        frame_index=0,
        image_size=(256, 160),
        confidence=0.9,
        points=(
            PointObservation(
                "centre_spot", (125.0, 78.0), pitch.point_landmarks["centre_spot"]
            ),
        ),
        lines=(LineObservation("halfway_line", image_line, pitch_line, 0.9),),
        metadata={"teacher_consensus": True},
    )


def test_teacher_index_accepts_only_consensus_and_dataset_loads(tmp_path) -> None:
    frame_path = tmp_path / "frames" / "test1" / "frame-00000000.jpg"
    cache_path = tmp_path / "teacher-cache" / "test1" / "frame-00000000"
    frame_path.parent.mkdir(parents=True)
    cv2.imwrite(str(frame_path), np.zeros((160, 256, 3), dtype=np.uint8))
    save_teacher_frame(_teacher(), cache_path)

    index_path = build_index(
        tmp_path,
        tmp_path / "domain.json",
        split="domain",
    )
    dataset = TeacherDistillationDataset(
        index_path,
        expected_split="domain",
        input_size=(128, 64),
    )

    sample = dataset[0]

    assert sample["image"].shape == (3, 64, 128)
    assert sample["teacher_semantic_probabilities"].shape == (21, 64, 128)
    assert sample["teacher_landmark_heatmaps"].shape == (33, 16, 32)
    assert float(sample["teacher_landmark_visibility"].sum()) > 0.0


def test_teacher_index_rejects_non_consensus_cache(tmp_path) -> None:
    frame_path = tmp_path / "frames" / "test1" / "frame-00000000.jpg"
    cache_path = tmp_path / "teacher-cache" / "test1" / "frame-00000000"
    frame_path.parent.mkdir(parents=True)
    cv2.imwrite(str(frame_path), np.zeros((160, 256, 3), dtype=np.uint8))
    teacher = _teacher()
    save_teacher_frame(
        TeacherFrame(
            teacher_id=teacher.teacher_id,
            teacher_version=teacher.teacher_version,
            source_id=teacher.source_id,
            frame_index=teacher.frame_index,
            image_size=teacher.image_size,
            confidence=teacher.confidence,
            points=teacher.points,
            lines=teacher.lines,
            metadata={"teacher_consensus": False},
        ),
        cache_path,
    )

    with pytest.raises(ValueError, match="no accepted"):
        build_index(tmp_path, tmp_path / "domain.json", split="domain")


def test_teacher_dataset_checks_cache_identity(tmp_path) -> None:
    frame_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(frame_path), np.zeros((160, 256, 3), dtype=np.uint8))
    cache_path = tmp_path / "teacher"
    save_teacher_frame(_teacher("actual"), cache_path)
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "split": "domain",
                "root": str(tmp_path),
                "frames": [
                    {
                        "source_id": "wrong",
                        "frame_index": 0,
                        "image_path": frame_path.name,
                        "teacher_cache_path": "teacher.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset = TeacherDistillationDataset(index_path, expected_split="domain")

    with pytest.raises(ValueError, match="source_id"):
        dataset[0]
