from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.multiview.localization import (
    attention_gate,
    crop_to_original_percent,
    event_prior_window,
    event_weighted_spatial_grid,
    full_window_spatial_grid,
    infer_cam_grid,
    largest_component_bbox,
    localization_reliability,
    select_temporal_window,
    temporal_activity_scores,
    temporal_bins,
)
from app.multiview.repository import MultiviewCaseRepository
from app.multiview.review_store import MultiviewReviewStore
from app.multiview.service import MultiviewAnalysisService
from app.server.main import app

# Deterministic scripted cases used only by tests; the product cases.json
# holds the real SoccerNet clips, which require GPU model inference.
DEMO_CASES_PATH = Path(__file__).parent / "fixtures" / "multiview_demo_cases.json"


@pytest.fixture
def demo_client(tmp_path):
    with TestClient(app) as client:
        previous = app.state.multiview_service
        app.state.multiview_service = MultiviewAnalysisService(
            MultiviewCaseRepository(DEMO_CASES_PATH),
            MultiviewReviewStore(tmp_path / "reviews.sqlite3"),
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
    assert payload["decision"]["analysis_id"].startswith("analysis-mvfoul_001-")
    assert payload["decision"]["action_candidates"][0]["label"]
    assert payload["decision"]["severity_candidates"][0]["confidence"] > 0
    assert "非模型输出" in payload["message"]
    assert payload["decision"]["suggested_intensity"] == "reckless"
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
        assert box["reliable"] is True


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


def test_cam_grid_supports_last_and_higher_resolution_mvit_blocks() -> None:
    assert infer_cam_grid(1 + 8 * 7 * 7) == (8, 7, 7)
    assert infer_cam_grid(1 + 8 * 14 * 14) == (8, 14, 14)
    with pytest.raises(RuntimeError, match="Unexpected Grad-CAM token count"):
        infer_cam_grid(400)


def test_spatial_cam_uses_full_window_independently_from_temporal_gate() -> None:
    cam = np.zeros((8, 7, 7), dtype=np.float32)
    cam[:6, 1:3, 1:3] = 1.0
    cam[6:, 4:6, 4:6] = 2.0

    grid = full_window_spatial_grid(cam)

    assert grid[1, 1] == pytest.approx(0.75)
    assert grid[4, 4] == pytest.approx(0.5)
    assert np.array_equal(grid, cam.mean(axis=0))


def test_spatial_cam_softly_prioritizes_the_known_event_time() -> None:
    cam = np.zeros((8, 7, 7), dtype=np.float32)
    cam[0, 1:3, 1:3] = 2.0
    cam[3:5, 4:6, 4:6] = 1.0

    grid = event_weighted_spatial_grid(cam, 2.52, 3.48, 3.0)

    assert grid[4, 4] > grid[1, 1]
    assert grid[4, 4] > 0.0


def test_spatial_cam_falls_back_when_event_is_outside_model_window() -> None:
    cam = np.arange(8 * 7 * 7, dtype=np.float32).reshape(8, 7, 7)

    grid = event_weighted_spatial_grid(cam, 2.52, 3.48, 4.5)

    assert np.array_equal(grid, cam.mean(axis=0))


def test_localization_reliability_rejects_low_attention_and_crop_edges() -> None:
    valid, score, reasons = attention_gate(1, [0.46, 0.09, 0.45])
    assert valid is False
    assert 0 < score < 1
    assert reasons == ["low_view_attention"]

    grid = np.zeros((7, 7), dtype=np.float32)
    grid[2:5, 2:5] = 1.0
    reliable = localization_reliability(
        view_index=0,
        view_attention=[0.46, 0.09, 0.45],
        temporal_valid=True,
        temporal_reason=None,
        spatial_grid=grid,
        bbox=(50, 50, 80, 80),
    )
    assert reliable["reliable"] is True
    assert reliable["display_tier"] == "normal"
    assert reliable["reliability_reasons"] == []

    caution = localization_reliability(
        view_index=0,
        view_attention=[0.46, 0.09, 0.45],
        temporal_valid=False,
        temporal_reason="flat_response",
        spatial_grid=grid,
        bbox=(0, 50, 80, 80),
    )
    assert caution["reliable"] is False
    assert caution["display_tier"] == "caution"
    assert caution["reliability_score"] == 0.45
    assert caution["reliability_reasons"] == [
        "flat_response",
        "crop_boundary_contact",
    ]

    compound = localization_reliability(
        view_index=1,
        view_attention=[0.46, 0.09, 0.45],
        temporal_valid=False,
        temporal_reason="flat_response",
        spatial_grid=grid,
        bbox=(0, 50, 80, 80),
    )
    assert compound["reliable"] is False
    assert compound["display_tier"] == "hidden"
    assert 0.0 < compound["reliability_score"] <= 0.45
    assert compound["reliability_reasons"] == [
        "low_view_attention",
        "flat_response",
        "crop_boundary_contact",
    ]

    low_attention_only = localization_reliability(
        view_index=1,
        view_attention=[0.46, 0.09, 0.45],
        temporal_valid=True,
        temporal_reason=None,
        spatial_grid=grid,
        bbox=(50, 50, 80, 80),
    )
    assert low_attention_only["display_tier"] == "caution"
    assert low_attention_only["reliability_reasons"] == ["low_view_attention"]

    empty = localization_reliability(
        view_index=0,
        view_attention=[0.46, 0.09, 0.45],
        temporal_valid=True,
        temporal_reason=None,
        spatial_grid=np.zeros((7, 7), dtype=np.float32),
        bbox=(50, 50, 80, 80),
    )
    assert empty["display_tier"] == "hidden"
    assert empty["reliability_score"] == 0.0
    assert empty["reliability_reasons"] == ["zero_response"]


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


def test_temporal_cam_caps_broad_response_around_peak() -> None:
    cam = np.zeros((8, 7, 7), dtype=np.float32)
    for index, strength in enumerate((1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.0)):
        cam[index, 2:5, 2:5] = strength

    selected = select_temporal_window(cam, 2.52, 3.48)

    assert selected["valid"] is True
    assert selected["peak_index"] in selected["active_indices"]
    assert 2 <= len(selected["active_indices"]) <= 4
    assert selected["active_end_s"] - selected["active_start_s"] <= 0.4801


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


def _complete_review_payload(analysis_id: str, expected_revision: int = 0) -> dict:
    def human(value) -> dict:
        return {"value": value, "source": "human", "confirmed": True}

    return {
        "expected_revision": expected_revision,
        "analysis_id": analysis_id,
        "review_state": "reviewed",
        "facts": {
            "offence_confirmed": human(True),
            "action": human("Tackle"),
            "offender_team": human("home"),
            "victim_team": human("away"),
            "ball_in_play": human(True),
            "contact": human(True),
            "contact_region": human("lower_body"),
            "intensity": human("reckless"),
            "attempt_to_play_ball": human(True),
            "tactical_impact": human("none"),
            "location": {"x_m": 30.0, "y_m": 34.0, "source": "human", "confirmed": True},
            "home_defends_side": human("left"),
        },
    }


def test_review_api_persists_server_computed_assessment_and_history(demo_client) -> None:
    analyzed = demo_client.post(
        "/api/multiview/analyze",
        json={"case_id": "mvfoul_001", "device": "auto"},
    ).json()["decision"]
    payload = _complete_review_payload(analyzed["analysis_id"])

    saved = demo_client.put("/api/multiview/cases/mvfoul_001/review", json=payload)
    assert saved.status_code == 200
    review = saved.json()["review"]
    assert review["revision"] == 1
    assert review["assessment"]["restart"] == "direct_free_kick"
    assert review["assessment"]["sanction"] == "yellow_card"
    assert review["assessment"]["status"] == "complete"

    restored = demo_client.get("/api/multiview/cases/mvfoul_001/review").json()
    assert restored["review"] == review
    assert restored["analysis"]["analysis_id"] == analyzed["analysis_id"]
    history = demo_client.get("/api/multiview/cases/mvfoul_001/review/history").json()
    assert history["count"] == 1
    assert history["history"][0]["revision"] == 1


def test_review_api_rejects_stale_revision(demo_client) -> None:
    analyzed = demo_client.post(
        "/api/multiview/analyze",
        json={"case_id": "mvfoul_001", "device": "auto"},
    ).json()["decision"]
    payload = _complete_review_payload(analyzed["analysis_id"])
    assert demo_client.put("/api/multiview/cases/mvfoul_001/review", json=payload).status_code == 200

    conflict = demo_client.put("/api/multiview/cases/mvfoul_001/review", json=payload)
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "REVIEW_REVISION_CONFLICT"


def test_explanation_api_has_deterministic_template_fallback(demo_client) -> None:
    analyzed = demo_client.post(
        "/api/multiview/analyze",
        json={"case_id": "mvfoul_001", "device": "auto"},
    ).json()["decision"]
    payload = _complete_review_payload(analyzed["analysis_id"])
    saved = demo_client.put("/api/multiview/cases/mvfoul_001/review", json=payload).json()["review"]

    response = demo_client.post(
        "/api/multiview/cases/mvfoul_001/explanation",
        json={"revision": saved["revision"], "use_llm": False},
    )
    assert response.status_code == 200
    explanation = response.json()
    assert explanation["source"] == "template"
    assert explanation["restart"] == "direct_free_kick"
    assert explanation["sanction"] == "yellow_card"
    assert explanation["summary"] == saved["assessment"]["explanation_template"]
