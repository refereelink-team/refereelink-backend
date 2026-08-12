#!/usr/bin/env python3
"""Apply final accuracy and safety gates to held-out registration reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def check_accuracy_report(
    report: dict[str, Any],
    *,
    minimum_annotated_frames: int = 40,
    minimum_safe_coverage: float = 0.75,
    minimum_display_coverage: float = 0.95,
    maximum_landmark_median_720p_px: float = 4.0,
    maximum_landmark_p95_720p_px: float = 8.0,
    maximum_grid_median_m: float = 0.75,
    maximum_grid_p95_m: float = 1.5,
    maximum_hard_cut_recovery_frames: int = 15,
) -> list[str]:
    issues: list[str] = []
    if report.get("accuracy_valid") is not True:
        issues.append("accuracy_valid is false; held-out validation/test labels are required")
    count = int(report.get("projectable_frame_count", 0))
    if count < minimum_annotated_frames:
        issues.append(
            f"projectable annotated frames {count} are below {minimum_annotated_frames}"
        )
    safe = float(report.get("safe_coverage", 0.0))
    if safe < minimum_safe_coverage:
        issues.append(
            f"SAFE coverage {safe:.3f} is below {minimum_safe_coverage:.3f}"
        )
    display = float(report.get("safe_plus_preview_coverage", 0.0))
    if display < minimum_display_coverage:
        issues.append(
            f"SAFE+PREVIEW coverage {display:.3f} is below {minimum_display_coverage:.3f}"
        )
    false_safe = int(report.get("false_safe_count", 0))
    if false_safe:
        issues.append(f"{false_safe} erroneous matrices entered SAFE")

    def maximum(summary_name: str, key: str, limit: float, label: str) -> None:
        summary = report.get(summary_name, {})
        value = summary.get(key) if isinstance(summary, dict) else None
        if value is None:
            issues.append(f"{label} is unavailable")
        elif float(value) > limit:
            issues.append(f"{label} {float(value):.3f} exceeds {limit:.3f}")

    maximum(
        "safe_landmark_reprojection_720p_px",
        "median",
        maximum_landmark_median_720p_px,
        "SAFE landmark median at 720p",
    )
    maximum(
        "safe_landmark_reprojection_720p_px",
        "p95",
        maximum_landmark_p95_720p_px,
        "SAFE landmark P95 at 720p",
    )
    maximum(
        "safe_grid_projection_m",
        "median",
        maximum_grid_median_m,
        "SAFE grid median",
    )
    maximum(
        "safe_grid_projection_m",
        "p95",
        maximum_grid_p95_m,
        "SAFE grid P95",
    )
    if int(report.get("hard_cut_count", 0)) > 0:
        recovery = report.get("hard_cut_recovery_frames", {})
        p95 = recovery.get("p95") if isinstance(recovery, dict) else None
        if p95 is None:
            issues.append("hard-cut recovery is unavailable")
        elif float(p95) > maximum_hard_cut_recovery_frames:
            issues.append(
                f"hard-cut P95 recovery {float(p95):.1f} frames exceeds "
                f"{maximum_hard_cut_recovery_frames}"
            )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--minimum-annotated-frames", type=int, default=40)
    arguments = parser.parse_args()
    report = json.loads(arguments.report.read_text(encoding="utf-8"))
    issues = check_accuracy_report(
        report, minimum_annotated_frames=arguments.minimum_annotated_frames
    )
    result = {"valid": not issues, "report": str(arguments.report), "issues": issues}
    print(json.dumps(result, indent=2, sort_keys=True))
    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
