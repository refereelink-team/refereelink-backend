#!/usr/bin/env python3
"""Create a tiny deterministic three-phase pitch-training fixture.

The generated frames are synthetic and therefore validate only data loading,
losses, optimisation, checkpoint creation and deployment latency. They must
never be reported as model-accuracy evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from app.field_registration.geometry import transform_points
from app.field_registration.pitch_model import PitchModel
from app.field_registration.types import LineObservation, PointObservation
from experiments.field_registration.teacher_cache import TeacherFrame, save_teacher_frame


IMAGE_SIZE = (320, 192)
PITCH_TO_IMAGE = np.asarray(
    [[2.58, 0.24, 20.0], [0.08, 2.16, 18.0], [0.0004, 0.0015, 1.0]],
    dtype=np.float64,
)


def _render_frame(pitch: PitchModel, seed: int) -> np.ndarray:
    width, height = IMAGE_SIZE
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), (42, 118, 52), dtype=np.uint8)
    image = np.clip(
        image.astype(np.int16) + rng.normal(0.0, 3.0, image.shape), 0, 255
    ).astype(np.uint8)
    for points in pitch.semantic_elements().values():
        projected = transform_points(points, PITCH_TO_IMAGE)
        finite = np.all(np.isfinite(projected), axis=1)
        pixels = np.rint(projected[finite]).astype(np.int32)
        if len(pixels) >= 2:
            cv2.polylines(image, [pixels.reshape(-1, 1, 2)], False, (235, 235, 235), 2)
    return image


def _correspondences(pitch: PitchModel) -> list[dict[str, object]]:
    labels = (
        "left_top_corner",
        "left_bottom_corner",
        "right_top_corner",
        "right_bottom_corner",
        "centre_top",
        "centre_bottom",
    )
    metric = np.asarray([pitch.landmarks[label] for label in labels], dtype=np.float64)
    image = transform_points(metric, PITCH_TO_IMAGE)
    return [
        {
            "label": label,
            "image_xy": image_xy.tolist(),
            "pitch_xy_m": pitch_xy.tolist(),
        }
        for label, image_xy, pitch_xy in zip(labels, image, metric)
    ]


def _write_manifest(root: Path, split: str, frame_names: list[str]) -> Path:
    pitch = PitchModel()
    payload = {
        "version": 1,
        "split": split,
        "source": {"name": f"synthetic-{split}"},
        "image_size": list(IMAGE_SIZE),
        "pitch_dimensions_m": {"length": 105.0, "width": 68.0},
        "frames": [
            {
                "frame_index": index,
                "image_path": name,
                "correspondences": _correspondences(pitch),
                "contact_points": [],
            }
            for index, name in enumerate(frame_names)
        ],
        "metadata": {"synthetic_smoke_fixture": True, "accuracy_valid": False},
    }
    path = root / f"{split}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _teacher(pitch: PitchModel, source_id: str, frame_index: int) -> TeacherFrame:
    points = tuple(
        PointObservation(
            label,
            tuple(transform_points(np.asarray([metric]), PITCH_TO_IMAGE)[0]),
            metric,
            confidence=0.95,
            source="synthetic-consensus",
        )
        for label, metric in pitch.point_landmarks.items()
    )
    lines = tuple(
        LineObservation(
            label,
            transform_points(metric, PITCH_TO_IMAGE),
            metric,
            confidence=0.95,
            source="synthetic-consensus",
        )
        for label, metric in pitch.semantic_elements().items()
    )
    return TeacherFrame(
        teacher_id="synthetic-teacher-consensus",
        teacher_version="smoke-v1",
        source_id=source_id,
        frame_index=frame_index,
        image_size=IMAGE_SIZE,
        confidence=0.95,
        points=points,
        lines=lines,
        image_to_pitch=np.linalg.inv(PITCH_TO_IMAGE),
        metadata={"teacher_consensus": True, "accuracy_valid": False},
    )


def _write_teacher_index(root: Path, split: str, image_name: str) -> Path:
    pitch = PitchModel()
    source_id = f"synthetic-{split}/frame-0"
    cache_base = root / "teacher-cache" / split / "frame-00000000"
    save_teacher_frame(_teacher(pitch, source_id, 0), cache_base)
    payload = {
        "format_version": 1,
        "split": split,
        "root": str(root),
        "frames": [
            {
                "source_id": source_id,
                "frame_index": 0,
                "image_path": image_name,
                "teacher_cache_path": str(cache_base.with_suffix(".json").relative_to(root)),
            }
        ],
        "metadata": {"synthetic_smoke_fixture": True, "accuracy_valid": False},
    }
    path = root / f"{split}-teacher-index.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def create_fixture(output: Path) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    pitch = PitchModel()
    names = {
        "train-0": "train-0.jpg",
        "train-1": "train-1.jpg",
        "validation": "validation-0.jpg",
        "distillation": "distillation-0.jpg",
        "domain": "test1-domain-0.jpg",
    }
    for index, name in enumerate(names.values()):
        cv2.imwrite(str(output / name), _render_frame(pitch, 100 + index))
    paths = {
        "train_manifest": str(
            _write_manifest(output, "train", [names["train-0"], names["train-1"]])
        ),
        "validation_manifest": str(
            _write_manifest(output, "validation", [names["validation"]])
        ),
        "distillation_index": str(
            _write_teacher_index(output, "distillation", names["distillation"])
        ),
        "domain_index": str(_write_teacher_index(output, "domain", names["domain"])),
    }
    (output / "fixture.json").write_text(
        json.dumps(
            {"format_version": 1, "accuracy_valid": False, "paths": paths},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(create_fixture(arguments.output), indent=2))


if __name__ == "__main__":
    main()
