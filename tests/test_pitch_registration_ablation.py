from __future__ import annotations

import json

from experiments.field_registration.ablation import EXPERIMENTS, experiment_by_id
from tools.summarize_pitch_registration_ablation import summarize


def _accuracy(valid: bool = True) -> dict[str, object]:
    return {
        "accuracy_valid": valid,
        "projectable_frame_count": 50,
        "safe_coverage": 0.8,
        "safe_plus_preview_coverage": 0.97,
        "false_safe_count": 0,
        "safe_landmark_reprojection_720p_px": {"median": 3.0, "p95": 7.0},
        "safe_grid_projection_m": {"median": 0.5, "p95": 1.2},
        "hard_cut_count": 0,
        "hard_cut_recovery_frames": {"p95": None},
    }


def _performance(*, fps: float = 35.0) -> dict[str, object]:
    return {
        "accuracy_valid": False,
        "model_latency_ms": {"p95": 7.0},
        "end_to_end_fps": fps,
        "baseline_fps": 36.0,
        "model_gpu_memory_mb": 120.0,
        "elapsed_sec": 1800.0,
        "rss_mb": {"slope_per_minute": 0.8},
    }


def test_ablation_matrix_matches_staged_plan() -> None:
    assert [item.experiment_id for item in EXPERIMENTS] == [
        "E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"
    ]
    assert experiment_by_id("E6").temporal == "offline_bidirectional_rts"
    assert experiment_by_id("E7").conditional is True


def test_ablation_summary_marks_missing_artifacts_blocked(tmp_path) -> None:
    report = summarize(tmp_path)

    assert all(row["status"] == "blocked" for row in report["experiments"])
    assert report["selection"]["selected_experiment"] is None
    assert report["selection"]["pidnet_s_unlocked"] is False


def test_ablation_summary_selects_passing_mobilenet_and_keeps_pidnet_locked(tmp_path) -> None:
    (tmp_path / "E5-accuracy.json").write_text(json.dumps(_accuracy()))
    (tmp_path / "E5-performance.json").write_text(
        json.dumps(_performance())
    )

    report = summarize(tmp_path, teacher_jac_at_5=0.82, student_jac_at_5=0.72)

    assert report["selection"]["selected_experiment"] == "E5"
    assert report["selection"]["mobilenet_gate_passed"] is True
    assert report["selection"]["pidnet_s_unlocked"] is False


def test_pidnet_unlock_requires_failed_mobilenet_and_large_jac_gap(tmp_path) -> None:
    (tmp_path / "E5-accuracy.json").write_text(json.dumps(_accuracy(valid=False)))
    (tmp_path / "E5-performance.json").write_text(json.dumps(_performance()))
    report = summarize(tmp_path, teacher_jac_at_5=0.82, student_jac_at_5=0.74)

    assert report["selection"]["pidnet_s_unlocked"] is True

    smaller = summarize(tmp_path, teacher_jac_at_5=0.82, student_jac_at_5=0.76)
    assert smaller["selection"]["pidnet_s_unlocked"] is False


def test_missing_mobilenet_results_do_not_unlock_pidnet(tmp_path) -> None:
    report = summarize(tmp_path, teacher_jac_at_5=0.82, student_jac_at_5=0.70)

    assert report["selection"]["mobilenet_evaluated"] is False
    assert report["selection"]["pidnet_s_unlocked"] is False


def test_performance_failure_blocks_selection(tmp_path) -> None:
    (tmp_path / "E5-accuracy.json").write_text(json.dumps(_accuracy()))
    (tmp_path / "E5-performance.json").write_text(
        json.dumps(_performance(fps=20.0))
    )

    report = summarize(tmp_path)
    row = next(row for row in report["experiments"] if row["experiment_id"] == "E5")

    assert row["accuracy_gate_passed"] is True
    assert row["performance_gate_passed"] is False
    assert row["all_gates_passed"] is False
    assert report["selection"]["selected_experiment"] is None
