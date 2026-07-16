from __future__ import annotations

from experiments.team_classification.benchmark import run_benchmark


def test_team_benchmark_uses_disjoint_calibration_and_test_tracks() -> None:
    result = run_benchmark(seed=11)
    assert result["evaluation_split"]["leakage_free"] is True
    assert result["evaluation_split"]["calibration_tracks"] == 20
    assert result["evaluation_split"]["test_tracks"] == 20
    assert {entry["mode"] for entry in result["mode_results"]} == {"color", "deep", "fusion"}
    assert {entry["calibration_tracks_per_team"] for entry in result["mode_results"]} == {
        1, 3, 5, 8, 10
    }
    assert {entry["deep_feature_update_interval_frames"] for entry in result["update_interval_results"]} == {
        1, 5, 10, 20
    }
