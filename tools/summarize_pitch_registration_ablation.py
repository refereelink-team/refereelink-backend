#!/usr/bin/env python3
"""Summarize E0-E7 reports and enforce the staged model-selection policy."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools._bootstrap import ensure_repository_root
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from _bootstrap import ensure_repository_root

ensure_repository_root(__file__)

from experiments.field_registration.ablation import EXPERIMENTS
from tools.check_pitch_registration_accuracy import check_accuracy_report
from tools.check_pitch_registration_performance import check_performance_report


def _read(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def summarize(
    report_root: str | Path,
    *,
    teacher_jac_at_5: float | None = None,
    student_jac_at_5: float | None = None,
) -> dict[str, Any]:
    root = Path(report_root)
    rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        accuracy_path = root / f"{experiment.experiment_id}-accuracy.json"
        performance_path = root / f"{experiment.experiment_id}-performance.json"
        accuracy = _read(accuracy_path)
        performance = _read(performance_path)
        blockers: list[str] = []
        accuracy_issues: list[str] = []
        performance_issues: list[str] = []
        if accuracy is None:
            blockers.append(f"missing {accuracy_path.name}")
        else:
            accuracy_issues = check_accuracy_report(accuracy)
        if performance is None:
            blockers.append(f"missing {performance_path.name}")
        else:
            performance_issues = check_performance_report(performance)
        accuracy_passed = accuracy is not None and not accuracy_issues
        performance_passed = performance is not None and not performance_issues
        row = {
            "experiment_id": experiment.experiment_id,
            "perception": experiment.perception,
            "temporal": experiment.temporal,
            "production_candidate": experiment.production_candidate,
            "conditional": experiment.conditional,
            "status": "blocked" if blockers else "evaluated",
            "blockers": blockers,
            "accuracy_gate_passed": accuracy_passed,
            "accuracy_issues": accuracy_issues,
            "performance_gate_passed": performance_passed,
            "performance_issues": performance_issues,
            "all_gates_passed": not blockers and accuracy_passed and performance_passed,
            "accuracy": accuracy,
            "performance": performance,
        }
        rows.append(row)

    mobilenet_rows = [
        row for row in rows if row["experiment_id"] in {"E3", "E4", "E5", "E6"}
    ]
    mobilenet_passes = [
        row
        for row in mobilenet_rows
        if row["all_gates_passed"]
    ]
    jac_gap = (
        teacher_jac_at_5 - student_jac_at_5
        if teacher_jac_at_5 is not None and student_jac_at_5 is not None
        else None
    )
    mobilenet_evaluated = any(row["status"] == "evaluated" for row in mobilenet_rows)
    pidnet_unlocked = (
        mobilenet_evaluated
        and not mobilenet_passes
        and jac_gap is not None
        and jac_gap > 0.07
    )
    online_passes = [
        row for row in mobilenet_passes if row["experiment_id"] in {"E3", "E4", "E5"}
    ]
    offline_passes = [
        row for row in mobilenet_passes if row["experiment_id"] == "E6"
    ]
    selected_online = online_passes[-1]["experiment_id"] if online_passes else None
    selected_offline = offline_passes[-1]["experiment_id"] if offline_passes else None
    return {
        "format_version": 1,
        "experiments": rows,
        "selection": {
            "selected_experiment": selected_offline or selected_online,
            "selected_online_experiment": selected_online,
            "selected_offline_experiment": selected_offline,
            "mobilenet_gate_passed": bool(mobilenet_passes),
            "mobilenet_evaluated": mobilenet_evaluated,
            "teacher_student_jac_at_5_gap": jac_gap,
            "pidnet_s_unlocked": pidnet_unlocked,
            "policy": (
                "Select the most complete online MobileNet candidate through E5 and "
                "report E6 separately for offline output. Every selection requires "
                "accuracy, safety, CUDA performance and 30-minute stability gates. "
                "Train PIDNet-S only after an evaluated MobileNet candidate fails "
                "and teacher-student JaC@5 gap > 0.07."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-jac-at-5", type=float)
    parser.add_argument("--student-jac-at-5", type=float)
    arguments = parser.parse_args()
    report = summarize(
        arguments.report_root,
        teacher_jac_at_5=arguments.teacher_jac_at_5,
        student_jac_at_5=arguments.student_jac_at_5,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
