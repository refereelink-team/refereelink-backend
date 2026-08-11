from __future__ import annotations

from tools.check_pitch_registration_soak import check_soak_report


def valid_report() -> dict[str, object]:
    return {
        "status": "ok",
        "elapsed_sec": 1800.2,
        "end_to_end_fps": 38.0,
        "rss_mb": {"slope_per_minute": 0.2},
        "peak_gpu_memory_mb": 240.0,
        "frame_latency_ms": {"p95": 28.0},
        "physical_pan_constraint_active": True,
    }


def test_soak_check_accepts_report_within_explicit_limits() -> None:
    issues = check_soak_report(
        valid_report(),
        minimum_duration_sec=1800.0,
        maximum_rss_slope_mb_per_minute=5.0,
        maximum_peak_gpu_memory_mb=500.0,
        baseline_fps=41.0,
        maximum_fps_drop_percent=10.0,
        maximum_p95_latency_ms=35.0,
        require_physical_pan=True,
    )

    assert issues == []


def test_soak_check_reports_each_failed_gate() -> None:
    report = valid_report()
    report.update(
        {
            "elapsed_sec": 120.0,
            "end_to_end_fps": 30.0,
            "rss_mb": {"slope_per_minute": 8.0},
            "peak_gpu_memory_mb": 600.0,
            "frame_latency_ms": {"p95": 45.0},
            "physical_pan_constraint_active": False,
        }
    )

    issues = check_soak_report(
        report,
        minimum_duration_sec=1800.0,
        maximum_rss_slope_mb_per_minute=5.0,
        maximum_peak_gpu_memory_mb=500.0,
        baseline_fps=40.0,
        maximum_fps_drop_percent=10.0,
        maximum_p95_latency_ms=35.0,
        require_physical_pan=True,
    )

    assert len(issues) == 6
    assert any("duration" in issue for issue in issues)
    assert any("FPS drop" in issue for issue in issues)
    assert any("RSS slope" in issue for issue in issues)
    assert any("peak GPU" in issue for issue in issues)
    assert any("P95 latency" in issue for issue in issues)
    assert any("physical pan" in issue for issue in issues)


def test_soak_check_rejects_missing_stability_measurements() -> None:
    report = valid_report()
    report["rss_mb"] = {"slope_per_minute": None}
    report["peak_gpu_memory_mb"] = None

    issues = check_soak_report(
        report,
        minimum_duration_sec=1800.0,
        maximum_rss_slope_mb_per_minute=5.0,
        maximum_peak_gpu_memory_mb=500.0,
    )

    assert "RSS slope is unavailable" in issues
    assert "peak GPU memory is unavailable" in issues
