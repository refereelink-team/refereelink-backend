#!/usr/bin/env python3
"""Apply explicit stability/performance gates to a pitch-registration soak report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def check_soak_report(
    report: dict[str, Any],
    *,
    minimum_duration_sec: float,
    maximum_rss_slope_mb_per_minute: float,
    maximum_peak_gpu_memory_mb: float,
    baseline_fps: float | None = None,
    maximum_fps_drop_percent: float = 10.0,
    maximum_p95_latency_ms: float | None = None,
    require_physical_pan: bool = False,
) -> list[str]:
    issues: list[str] = []
    if report.get("status") != "ok":
        issues.append("benchmark status is not ok")
    elapsed_sec = float(report.get("elapsed_sec", 0.0))
    if elapsed_sec < minimum_duration_sec:
        issues.append(
            f"duration {elapsed_sec:.1f}s is below {minimum_duration_sec:.1f}s"
        )
    fps = float(report.get("end_to_end_fps", 0.0))
    if fps <= 0.0:
        issues.append("end-to-end FPS is missing or non-positive")
    if baseline_fps is not None and baseline_fps > 0.0 and fps > 0.0:
        drop_percent = max(0.0, (baseline_fps - fps) * 100.0 / baseline_fps)
        if drop_percent > maximum_fps_drop_percent:
            issues.append(
                f"FPS drop {drop_percent:.2f}% exceeds "
                f"{maximum_fps_drop_percent:.2f}%"
            )

    rss = report.get("rss_mb")
    slope = rss.get("slope_per_minute") if isinstance(rss, dict) else None
    if slope is None:
        issues.append("RSS slope is unavailable")
    elif float(slope) > maximum_rss_slope_mb_per_minute:
        issues.append(
            f"RSS slope {float(slope):.3f} MB/min exceeds "
            f"{maximum_rss_slope_mb_per_minute:.3f} MB/min"
        )

    peak_gpu = report.get("peak_gpu_memory_mb")
    if peak_gpu is None:
        issues.append("peak GPU memory is unavailable")
    elif float(peak_gpu) > maximum_peak_gpu_memory_mb:
        issues.append(
            f"peak GPU memory {float(peak_gpu):.1f} MB exceeds "
            f"{maximum_peak_gpu_memory_mb:.1f} MB"
        )

    latency = report.get("frame_latency_ms")
    p95_latency = latency.get("p95") if isinstance(latency, dict) else None
    if maximum_p95_latency_ms is not None:
        if p95_latency is None:
            issues.append("P95 frame latency is unavailable")
        elif float(p95_latency) > maximum_p95_latency_ms:
            issues.append(
                f"P95 latency {float(p95_latency):.2f} ms exceeds "
                f"{maximum_p95_latency_ms:.2f} ms"
            )

    if require_physical_pan and not report.get("physical_pan_constraint_active"):
        issues.append("physical pan constraint was required but not active")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--minimum-duration-sec", type=float, default=1800.0)
    parser.add_argument("--maximum-rss-slope-mb-per-minute", type=float, default=5.0)
    parser.add_argument("--maximum-peak-gpu-memory-mb", type=float, default=500.0)
    parser.add_argument("--baseline-fps", type=float)
    parser.add_argument("--maximum-fps-drop-percent", type=float, default=10.0)
    parser.add_argument("--maximum-p95-latency-ms", type=float)
    parser.add_argument("--require-physical-pan", action="store_true")
    arguments = parser.parse_args()
    report = json.loads(arguments.report.read_text(encoding="utf-8"))
    issues = check_soak_report(
        report,
        minimum_duration_sec=arguments.minimum_duration_sec,
        maximum_rss_slope_mb_per_minute=(
            arguments.maximum_rss_slope_mb_per_minute
        ),
        maximum_peak_gpu_memory_mb=arguments.maximum_peak_gpu_memory_mb,
        baseline_fps=arguments.baseline_fps,
        maximum_fps_drop_percent=arguments.maximum_fps_drop_percent,
        maximum_p95_latency_ms=arguments.maximum_p95_latency_ms,
        require_physical_pan=arguments.require_physical_pan,
    )
    result = {
        "valid": not issues,
        "report": str(arguments.report),
        "issues": issues,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
