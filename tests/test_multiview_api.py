from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.constants.paths import REPO_ROOT_DIR
from app.multiview.localization import (
    crop_to_original_percent,
    event_prior_window,
    largest_component_bbox,
    select_temporal_window,
    temporal_activity_scores,
    temporal_bins,
)
from app.multiview.repository import MultiviewCaseRepository
from app.multiview.service import MultiviewAnalysisService
from app.server.main import app


@pytest.fixture
def demo_client():
    with TestClient(app) as client:
        previous = app.state.multiview_service
        app.state.multiview_service = MultiviewAnalysisService(
            MultiviewCaseRepository(REPO_ROOT_DIR / "assets" / "multiview" / "cases.json")
        )
        try:
            yield client
        finally:
            app.state.multiview_service = previous


def test_multiview_cases_hide_local_paths_and_expose_media(demo_client) -> None:
    client = demo_client
    response = client.get("/api/multiview/cases")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] >= 3
    first = payload["cases"][0]
    assert "scripted_result" not in first
    assert first["videos"]
    assert "path" not in first["videos"][0]
    assert "preview_path" not in first["videos"][0]
    assert first["videos"][0]["media_url"].startswith("/api/multiview/")

    media = client.get(first["videos"][0]["media_url"])
    assert media.status_code == 200
    assert media.headers["content-type"].startswith("image/jpeg")
    assert media.headers["accept-ranges"] == "bytes"


def test_multiview_scripted_fallback_is_explicit(demo_client) -> None:
    response = demo_client.post(
        "/api/multiview/analyze",
        json={"case_id": "mvfoul_001", "device": "auto"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["decision"]["mode"] == "scripted"
    assert "非模型输出" in payload["message"]
    assert payload["decision"]["localization_source"] == "scripted"
    localization = payload["decision"]["localization"]
    assert localization
    expected_peaks = {"cam_main": 3.0, "cam_side1": 3.03, "cam_side2": 2.98}
    for camera_id, box in localization.items():
        expected_peak = expected_peaks[camera_id]
        assert box["active_start_s"] == pytest.approx(expected_peak - 0.5)
        assert box["active_end_s"] == pytest.approx(expected_peak + 0.5)
        assert box["peak_s"] == pytest.approx(expected_peak)
        assert box["temporal_source"] == "event_prior"


def test_multiview_missing_case_returns_clear_error(demo_client) -> None:
    detail = demo_client.get("/api/multiview/cases/not-found")
    assert detail.status_code == 404
    analyze = demo_client.post(
        "/api/multiview/analyze",
        json={"case_id": "not-found", "device": "auto"},
    )
    assert analyze.status_code == 200
    assert analyze.json()["status"] == "error"


def test_localization_geometry_helpers() -> None:
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    mask[9:17, 12:25] = 1
    assert largest_component_bbox(mask) == (12, 9, 13, 8)

    rect = crop_to_original_percent(20, 30, 80, 90, 1280, 720)
    assert len(rect) == 4
    x, y, width, height = rect
    assert 0 <= x <= 100
    assert 0 <= y <= 100
    assert 0 < width <= 100 - x
    assert 0 < height <= 100 - y


def test_temporal_cam_selects_peak_interval_without_averaging_time() -> None:
    cam = np.zeros((8, 7, 7), dtype=np.float32)
    cam[3, 2:5, 2:5] = 1.0
    selected = select_temporal_window(cam, 2.52, 3.48)

    assert selected["valid"] is True
    assert selected["peak_index"] == 3
    assert selected["active_indices"] == [2, 3, 4]
    assert selected["active_start_s"] == 2.76
    assert selected["active_end_s"] == 3.12
    assert selected["active_start_s"] <= selected["peak_s"] <= selected["active_end_s"]


def test_temporal_cam_keeps_one_contiguous_region_for_multiple_peaks() -> None:
    cam = np.zeros((8, 7, 7), dtype=np.float32)
    cam[1, 1:4, 1:4] = 1.0
    cam[6, 3:6, 3:6] = 0.9
    selected = select_temporal_window(cam, 0.0, 1.0)

    assert selected["valid"] is True
    assert 1 in selected["active_indices"]
    assert 6 not in selected["active_indices"]


def test_flat_and_zero_temporal_cam_require_event_prior() -> None:
    flat_scores, flat_valid, flat_reason = temporal_activity_scores(
        np.ones((8, 7, 7), dtype=np.float32)
    )
    zero_scores, zero_valid, zero_reason = temporal_activity_scores(
        np.zeros((8, 7, 7), dtype=np.float32)
    )

    assert flat_valid is False
    assert flat_reason == "flat_response"
    assert zero_valid is False
    assert zero_reason == "zero_response"
    assert flat_scores.shape == zero_scores.shape == (8,)

    fallback = event_prior_window(3.0, duration_s=5.0)
    assert fallback == {
        "active_start_s": 2.5,
        "active_end_s": 3.5,
        "peak_s": 3.0,
        "temporal_source": "event_prior",
    }


def test_temporal_bins_use_actual_model_window_seconds() -> None:
    bins = temporal_bins(np.linspace(0.0, 1.0, 8), 63 / 25, 87 / 25)
    assert len(bins) == 8
    assert bins[0]["start_s"] == 2.52
    assert bins[-1]["end_s"] == 3.48
    assert all(left["end_s"] == right["start_s"] for left, right in zip(bins, bins[1:]))


def test_no_offence_scripted_case_has_no_localization_overlay(demo_client) -> None:
    response = demo_client.post(
        "/api/multiview/analyze",
        json={"case_id": "mvfoul_003", "device": "auto"},
    )
    assert response.status_code == 200
    assert response.json()["decision"]["localization"] == {}
