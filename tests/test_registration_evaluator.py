from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_evaluator():
    path = Path(__file__).parents[1] / "tools" / "evaluate_pitch_registration.py"
    spec = importlib.util.spec_from_file_location("registration_evaluator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluator_reports_accuracy_availability_and_latency() -> None:
    evaluator = _load_evaluator()
    truth = np.array(
        [[0.1, 0.0, -10.0], [0.0, 0.1, -5.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    predicted = truth.copy()
    predicted[0, 2] += 0.5
    payload = {
        "version": 1,
        "image_size": [1280, 720],
        "frames": [
            {
                "frame_index": 0,
                "ground_truth_image_to_pitch": truth.tolist(),
                "predicted_image_to_pitch": predicted.tolist(),
                "correspondences": [
                    {"image_xy": [100.0, 50.0], "pitch_xy_m": [0.0, 0.0]},
                    {"image_xy": [500.0, 300.0], "pitch_xy_m": [40.0, 25.0]},
                ],
                "status": "corrected",
                "latency_ms": 3.0,
            },
            {
                "frame_index": 1,
                "predicted_image_to_pitch": None,
                "status": "lost",
                "latency_ms": 1.0,
            },
        ],
    }

    report = evaluator.evaluate_payload(payload)

    assert report["frame_count"] == 2
    assert report["annotated_frame_count"] == 1
    assert report["availability_ratio"] == 0.5
    assert report["status_counts"] == {"corrected": 1, "lost": 1}
    assert report["pitch_projection_m"]["median"] == 0.5
    assert report["latency_ms"]["mean"] == 2.0


def test_evaluator_uses_correspondences_without_dense_ground_truth() -> None:
    evaluator = _load_evaluator()
    predicted = np.array(
        [[0.1, 0.0, -10.0], [0.0, 0.1, -5.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    payload = {
        "version": 1,
        "image_size": [1280, 720],
        "frames": [
            {
                "frame_index": 10,
                "predicted_image_to_pitch": predicted.tolist(),
                "correspondences": [
                    {"image_xy": [100.0, 50.0], "pitch_xy_m": [0.0, 0.0]},
                    {"image_xy": [500.0, 300.0], "pitch_xy_m": [40.0, 25.0]},
                ],
                "contact_points": [
                    {
                        "image_xy": [300.0, 400.0],
                        "predicted_image_xy": [303.0, 404.0],
                    }
                ],
                "status": "corrected",
            }
        ],
    }

    report = evaluator.evaluate_payload(payload)

    assert report["annotated_frame_count"] == 1
    assert report["pitch_projection_m"]["median"] == 0.0
    assert report["contact_image_error_px"]["median"] == 5.0
    assert report["grid_projection_m"]["count"] == 0
