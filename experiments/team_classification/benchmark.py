from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from app.classification.team_calibration.predictor import SupervisedPrototypeClassifier
from app.classification.team_calibration.prototypes import build_prototypes
from app.classification.team_calibration.types import PlayerRole, TeamLabel, TeamPrediction, TrackFeature


def run_benchmark(*, seed: int = 7) -> dict:
    """Run a leakage-free synthetic Track-level ablation benchmark.

    Calibration and test Tracks are generated from separate random streams.
    This validates classifier plumbing and rejection behavior; it is not a
    substitute for a labelled football-match accuracy report.
    """

    rng = np.random.default_rng(seed)
    all_tracks = _make_tracks(rng, tracks_per_team=20)
    calibration_tracks = all_tracks[:20]
    test_tracks = all_tracks[20:]
    sample_counts = [1, 3, 5, 8, 10]
    mode_results = []
    for mode in ("color", "deep", "fusion"):
        for sample_count in sample_counts:
            selected = _balanced_sample(calibration_tracks, sample_count)
            prototypes = build_prototypes(selected, include_goalkeepers=False)
            classifier = SupervisedPrototypeClassifier(
                prototypes,
                color_weight=1.0 if mode == "color" else 0.0 if mode == "deep" else 0.6,
                deep_weight=1.0 if mode == "deep" else 0.0 if mode == "color" else 0.4,
                min_observations=1,
                max_distance=10.0,
                min_margin=0.01,
            )
            predictions, elapsed = _predict_test(classifier, test_tracks, mode)
            known = [prediction for prediction in predictions if prediction.team != TeamLabel.UNKNOWN]
            correct = sum(
                prediction.team == expected.team
                for prediction, expected in zip(predictions, test_tracks)
                if prediction.team != TeamLabel.UNKNOWN
            )
            mode_results.append(
                {
                    "mode": mode,
                    "calibration_tracks_per_team": sample_count,
                    "test_tracks": len(test_tracks),
                    "track_accuracy_excluding_unknown": round(correct / max(len(known), 1), 4),
                    "coverage": round(len(known) / max(len(predictions), 1), 4),
                    "unknown_rate": round(1.0 - len(known) / max(len(predictions), 1), 4),
                    "mean_prediction_latency_ms": round(elapsed * 1000 / max(len(predictions), 1), 4),
                }
            )

    update_results = []
    prototypes = build_prototypes(_balanced_sample(calibration_tracks, 5), include_goalkeepers=False)
    classifier = SupervisedPrototypeClassifier(
        prototypes,
        min_observations=1,
        max_distance=10.0,
        min_margin=0.01,
    )
    for interval in (1, 5, 10, 20):
        start = time.perf_counter()
        calls = 0
        for index, track in enumerate(test_tracks * 4):
            if index % interval == 0:
                classifier.predict(
                    color_feature=track.color_feature,
                    deep_feature=track.deep_feature,
                    role=track.role,
                    observation_count=1,
                )
                calls += 1
        elapsed = time.perf_counter() - start
        update_results.append(
            {
                "deep_feature_update_interval_frames": interval,
                "classifier_calls": calls,
                "mean_classifier_call_latency_ms": round(elapsed * 1000 / max(calls, 1), 4),
            }
        )

    return {
        "benchmark": "supervised_team_calibration_synthetic",
        "seed": seed,
        "evaluation_split": {
            "calibration_tracks": len(calibration_tracks),
            "test_tracks": len(test_tracks),
            "leakage_free": True,
        },
        "warning": "Synthetic features validate plumbing and rejection logic, not real-match accuracy.",
        "mode_results": mode_results,
        "update_interval_results": update_results,
    }


def _make_tracks(rng: np.random.Generator, *, tracks_per_team: int) -> list[TrackFeature]:
    tracks: list[TrackFeature] = []
    for team_index, team in enumerate((TeamLabel.HOME, TeamLabel.AWAY)):
        for offset in range(tracks_per_team):
            color_center = np.zeros(36, dtype=np.float32)
            color_center[2 + team_index * 16] = 1.0
            color = color_center + rng.normal(0.0, 0.035, 36).astype(np.float32)
            color = np.clip(color, 0.0, None)
            color /= max(float(np.linalg.norm(color)), 1e-8)
            deep_center = np.zeros(8, dtype=np.float32)
            deep_center[team_index] = 1.0
            deep = deep_center + rng.normal(0.0, 0.08, 8).astype(np.float32)
            deep /= max(float(np.linalg.norm(deep)), 1e-8)
            tracks.append(
                TrackFeature(
                    track_id=team_index * tracks_per_team + offset,
                    team=team,
                    role=PlayerRole.OUTFIELD,
                    color_feature=color,
                    deep_feature=deep,
                    observation_count=5,
                    quality_sum=4.0,
                    last_update_frame=offset,
                )
            )
    rng.shuffle(tracks)
    return tracks


def _balanced_sample(tracks: Iterable[TrackFeature], count: int) -> list[TrackFeature]:
    selected: list[TrackFeature] = []
    for team in (TeamLabel.HOME, TeamLabel.AWAY):
        selected.extend([track for track in tracks if track.team == team][:count])
    return selected


def _predict_test(
    classifier: SupervisedPrototypeClassifier,
    test_tracks: list[TrackFeature],
    mode: str,
) -> tuple[list[TeamPrediction], float]:
    predictions: list[TeamPrediction] = []
    start = time.perf_counter()
    for track in test_tracks:
        predictions.append(
            classifier.predict(
                color_feature=track.color_feature if mode != "deep" else None,
                deep_feature=track.deep_feature if mode != "color" else None,
                role=track.role,
                observation_count=track.observation_count,
            )
        )
    return predictions, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_benchmark(seed=args.seed)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

