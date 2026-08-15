from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from app.multiview.localization import crop_to_original_percent, largest_component_bbox
from app.server.main import app


def test_multiview_cases_hide_local_paths_and_expose_media() -> None:
    with TestClient(app) as client:
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


def test_multiview_scripted_fallback_is_explicit() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/multiview/analyze",
            json={"case_id": "mvfoul_001", "device": "auto"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["decision"]["mode"] in {"model", "scripted"}
        if payload["decision"]["mode"] == "scripted":
            assert "非模型输出" in payload["message"]
            assert payload["decision"]["localization_source"] == "scripted"


def test_multiview_missing_case_returns_clear_error() -> None:
    with TestClient(app) as client:
        detail = client.get("/api/multiview/cases/not-found")
        assert detail.status_code == 404
        analyze = client.post(
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
