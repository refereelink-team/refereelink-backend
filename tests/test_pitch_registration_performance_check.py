from __future__ import annotations

from tools.check_pitch_registration_performance import check_performance_report


def _report() -> dict[str, object]:
    return {
        "accuracy_valid": False,
        "model_latency_ms": {"p95": 7.5},
        "end_to_end_fps": 35.0,
        "baseline_fps": 36.0,
        "model_gpu_memory_mb": 150.0,
        "elapsed_sec": 1800.1,
        "rss_mb": {"slope_per_minute": 0.9},
    }


def test_performance_check_accepts_complete_cuda_evidence() -> None:
    assert check_performance_report(_report()) == []


def test_performance_check_reports_each_missing_or_failed_gate() -> None:
    report = _report()
    report.update(
        {
            "accuracy_valid": True,
            "model_latency_ms": {"p95": 9.0},
            "end_to_end_fps": 20.0,
            "baseline_fps": 40.0,
            "model_gpu_memory_mb": 350.0,
            "elapsed_sec": 100.0,
            "rss_mb": {"slope_per_minute": 7.0},
        }
    )

    issues = check_performance_report(report)

    assert len(issues) == 7
    assert any("accuracy_valid" in issue for issue in issues)
    assert any("model P95" in issue for issue in issues)
    assert any("end-to-end FPS" in issue for issue in issues)
    assert any("FPS drop" in issue for issue in issues)
    assert any("GPU memory" in issue for issue in issues)
    assert any("duration" in issue for issue in issues)
    assert any("RSS slope" in issue for issue in issues)
