from __future__ import annotations

import json

import numpy as np
import pytest

from app.field_registration.pitch_model import PitchModel
from app.field_registration.types import LineObservation, PointObservation
from experiments.field_registration.research_assets import (
    ResearchAssetError,
    load_research_assets,
)
from experiments.field_registration.teacher_cache import (
    TeacherCacheError,
    TeacherFrame,
    compare_teacher_frames,
    load_teacher_frame,
    save_teacher_frame,
)


def _teacher(name: str, homography: np.ndarray) -> TeacherFrame:
    line = LineObservation(
        "halfway_line",
        np.array([[320.0, 80.0], [320.0, 440.0]]),
        PitchModel().semantic_elements()["halfway_line"],
        0.9,
        name,
    )
    return TeacherFrame(
        teacher_id=name,
        teacher_version="abc123",
        source_id="clip-a",
        frame_index=17,
        image_size=(640, 480),
        confidence=0.9,
        points=(PointObservation("centre", (320.0, 240.0), (52.5, 34.0)),),
        lines=(line, LineObservation(
            "top_touchline",
            np.array([[40.0, 100.0], [600.0, 100.0]]),
            PitchModel().semantic_elements()["top_touchline"],
            0.8,
            name,
        )),
        image_to_pitch=homography,
        camera_parameters={"focal_px": 900.0},
    )


def test_research_registry_rejects_duplicate_ids(tmp_path) -> None:
    asset = {
        "id": "same",
        "kind": "model",
        "source_url": "https://example.com/model",
        "version": "1",
        "license": "MIT",
        "data_use": "test",
        "training_data": "none",
        "allowed_in_runtime": False,
    }
    path = tmp_path / "assets.json"
    path.write_text(json.dumps({"format_version": 1, "assets": [asset, asset]}))
    with pytest.raises(ResearchAssetError, match="unique"):
        load_research_assets(path)


def test_teacher_cache_round_trip_and_hash_guard(tmp_path) -> None:
    frame = _teacher("pnlcalib", np.eye(3))
    json_path, npz_path = save_teacher_frame(frame, tmp_path / "frame-17")
    loaded = load_teacher_frame(json_path)

    assert loaded.teacher_id == "pnlcalib"
    assert loaded.frame_index == 17
    assert loaded.lines[0].image_points.shape == (2, 2)
    assert loaded.camera_parameters["focal_px"] == 900.0

    npz_path.write_bytes(npz_path.read_bytes() + b"tampered")
    with pytest.raises(TeacherCacheError, match="SHA-256"):
        load_teacher_frame(json_path)


def test_teacher_agreement_accepts_matching_geometry_and_rejects_conflict() -> None:
    # Map the 105 x 68 pitch into a 525 x 340 image region.
    image_to_pitch = np.array([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 1.0]])
    matching = compare_teacher_frames(
        _teacher("pnlcalib", image_to_pitch),
        _teacher("tvcalib", image_to_pitch.copy()),
        minimum_common_line_labels=2,
    )
    shifted = image_to_pitch.copy()
    shifted[0, 2] = 20.0
    conflict = compare_teacher_frames(
        _teacher("pnlcalib", image_to_pitch),
        _teacher("tvcalib", shifted),
        minimum_common_line_labels=2,
    )

    assert matching.accepted
    assert matching.median_grid_disagreement_px == pytest.approx(0.0)
    assert not conflict.accepted
    assert conflict.rejection_reason in {"geometry_disagreement", "insufficient_shared_field"}
