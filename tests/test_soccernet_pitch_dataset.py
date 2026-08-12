from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from experiments.field_registration.soccernet import (
    SOCCERNET_GOAL_STRUCTURE_LABELS,
    SOCCERNET_RAW_LABELS,
    SOCCERNET_TO_DEPLOYMENT,
    SoccerNetCalibrationDataset,
    SoccerNetFormatError,
    SoccerNetIndex,
    SoccerNetIndexFrame,
    assert_sequence_disjoint,
    deployment_polylines,
    derive_visible_landmarks,
    parse_soccernet_annotation,
)
from tools.build_soccernet_pitch_index import build_index


def _annotation_payload() -> dict[str, list[dict[str, float]]]:
    return {
        "Side line top": [{"x": 0.1, "y": 0.1}, {"x": 0.9, "y": 0.1}],
        "Side line bottom": [{"x": 0.1, "y": 0.9}, {"x": 0.9, "y": 0.9}],
        "Side line left": [{"x": 0.1, "y": 0.1}, {"x": 0.1, "y": 0.9}],
        "Side line right": [{"x": 0.9, "y": 0.1}, {"x": 0.9, "y": 0.9}],
        "Middle line": [{"x": 0.5, "y": 0.1}, {"x": 0.5, "y": 0.9}],
        "Goal left crossbar": [{"x": 0.05, "y": 0.4}, {"x": 0.05, "y": 0.6}],
    }


def test_soccernet_mapping_preserves_26_classes_and_excludes_goal_structure() -> None:
    assert len(SOCCERNET_RAW_LABELS) == 26
    assert len(SOCCERNET_TO_DEPLOYMENT) == 26
    assert len(SOCCERNET_GOAL_STRUCTURE_LABELS) == 6
    assert sum(value is not None for value in SOCCERNET_TO_DEPLOYMENT.values()) == 20
    assert len({value for value in SOCCERNET_TO_DEPLOYMENT.values() if value}) == 20


def test_normalized_soccernet_points_map_to_deployment_lines() -> None:
    raw = parse_soccernet_annotation(_annotation_payload(), (1000, 500))
    deployed = deployment_polylines(raw)

    assert raw["Side line top"][0] == pytest.approx((100.0, 50.0))
    assert deployed["top_touchline"][1] == pytest.approx((900.0, 50.0))
    assert "Goal left crossbar" not in deployed


def test_visible_landmarks_require_observed_line_geometry() -> None:
    raw = parse_soccernet_annotation(_annotation_payload(), (1000, 500))
    landmarks = derive_visible_landmarks(
        deployment_polylines(raw),
        (1000, 500),
    )

    assert landmarks["left_top_corner"] == pytest.approx((100.0, 50.0), abs=0.2)
    assert landmarks["right_bottom_corner"] == pytest.approx((900.0, 450.0), abs=0.2)
    assert "left_penalty_spot" not in landmarks
    assert "centre_circle_top" not in landmarks


def test_soccernet_dataset_masks_invisible_landmarks(tmp_path) -> None:
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    image[:] = (40, 110, 35)
    image_path = tmp_path / "frame-0001.jpg"
    annotation_path = tmp_path / "frame-0001.json"
    cv2.imwrite(str(image_path), image)
    annotation_path.write_text(json.dumps(_annotation_payload()), encoding="utf-8")
    index = SoccerNetIndex(
        split="train",
        root=str(tmp_path),
        frames=(
            SoccerNetIndexFrame(
                source_id="match-a/frame-1",
                sequence_id="match-a",
                frame_index=1,
                image_path=image_path.name,
                annotation_path=annotation_path.name,
                image_size=(320, 180),
                raw_labels_present=tuple(_annotation_payload()),
            ),
        ),
    )
    index_path = index.save(tmp_path / "index.json")
    dataset = SoccerNetCalibrationDataset(
        index_path,
        expected_split="train",
        input_size=(128, 64),
    )

    sample = dataset[0]

    assert sample["semantic_target"].shape == (64, 128)
    assert int(sample["semantic_target"].max()) > 0
    assert sample["landmark_heatmaps"].shape == (33, 16, 32)
    assert 4 <= int(sample["landmark_visibility"].sum()) < 33
    invisible = sample["landmark_visibility"] == 0
    assert float(sample["landmark_heatmaps"][invisible].sum()) == 0.0
    assert float(sample["offset_mask"][invisible].sum()) == 0.0


def test_index_builder_and_sequence_leakage_gate(tmp_path) -> None:
    train_root = tmp_path / "train"
    sequence = train_root / "league" / "match-a"
    sequence.mkdir(parents=True)
    cv2.imwrite(str(sequence / "0001.jpg"), np.zeros((100, 200, 3), dtype=np.uint8))
    (sequence / "0001.json").write_text(
        json.dumps(_annotation_payload()), encoding="utf-8"
    )
    train_path = build_index(
        train_root,
        train_root / "train-index.json",
        split="train",
    )
    train_index = SoccerNetIndex.load(train_path)
    validation_index = SoccerNetIndex(
        "validation",
        str(train_root),
        (
            SoccerNetIndexFrame(
                source_id="another",
                sequence_id="league/match-a",
                frame_index=2,
                image_path="league/match-a/0001.jpg",
                annotation_path="league/match-a/0001.json",
                image_size=(200, 100),
                raw_labels_present=("Side line top",),
            ),
        ),
    )

    assert len(train_index.frames) == 1
    with pytest.raises(SoccerNetFormatError, match="leaks"):
        assert_sequence_disjoint((train_index, validation_index))
