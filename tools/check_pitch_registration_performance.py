#!/usr/bin/env python3
"""Apply deployment and long-run gates to a registration performance report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _summary_p95(report: dict[str, Any], *names: str) -> float | None:
    for name in names:
        summary = report.get(name)
        if isinstance(summary, dict) and summary.get("p95") is not None:
            return float(summary["p95"])
    return None


def check_performance_report(
    report: dict[str, Any],
    *,
    maximum_model_p95_ms: float = 8.0,
    minimum_end_to_end_fps: float = 30.0,
    maximum_fps_drop_percent: float = 10.0,
    maximum_model_gpu_memory_mb: float = 300.0,
    minimum_duration_sec: float = 1800.0,
    maximum_rss_slope_mb_per_minute: float = 5.0,
) -> list[str]:
    """Return every failed performance gate instead of stopping at the first."""

    issues: list[str] = []
    if report.get("accuracy_valid") is not False:
        issues.append("performance report must explicitly set accuracy_valid=false")

    model_p95 = _summary_p95(report, "model_latency_ms", "latency_ms")
    if model_p95 is None:
        issues.append("model P95 latency is unavailable")
    elif model_p95 > maximum_model_p95_ms:
        issues.append(
            f"model P95 latency {model_p95:.2f} ms exceeds "
            f"{maximum_model_p95_ms:.2f} ms"
        )

    fps = report.get("end_to_end_fps")
    if fps is None or float(fps) < minimum_end_to_end_fps:
        rendered = "unavailable" if fps is None else f"{float(fps):.2f}"
        issues.append(
            f"end-to-end FPS {rendered} is below {minimum_end_to_end_fps:.2f}"
        )

    baseline_fps = report.get("baseline_fps")
    if baseline_fps is None or float(baseline_fps) <= 0.0:
        issues.append("baseline FPS is unavailable")
    elif fps is not None and float(fps) > 0.0:
        drop = max(
            0.0,
            (float(baseline_fps) - float(fps)) * 100.0 / float(baseline_fps),
        )
        if drop > maximum_fps_drop_percent:
            issues.append(
                f"FPS drop {drop:.2f}% exceeds {maximum_fps_drop_percent:.2f}%"
            )

    gpu_memory = report.get("model_gpu_memory_mb")
    if gpu_memory is None:
        # A model-only benchmark's peak allocation is an acceptable conservative
        # substitute, but complete-pipeline peak memory is not incremental memory.
        gpu_memory = report.get("model_peak_gpu_memory_mb")
    if gpu_memory is None:
        issues.append("model GPU memory is unavailable")
    elif float(gpu_memory) > maximum_model_gpu_memory_mb:
        issues.append(
            f"model GPU memory {float(gpu_memory):.1f} MB exceeds "
            f"{maximum_model_gpu_memory_mb:.1f} MB"
        )

    elapsed = report.get("elapsed_sec")
    if elapsed is None or float(elapsed) < minimum_duration_sec:
        rendered = "unavailable" if elapsed is None else f"{float(elapsed):.1f}s"
        issues.append(
            f"duration {rendered} is below {minimum_duration_sec:.1f}s"
        )

    rss = report.get("rss_mb")
    rss_slope = rss.get("slope_per_minute") if isinstance(rss, dict) else None
    if rss_slope is None:
        issues.append("RSS slope is unavailable")
    elif float(rss_slope) > maximum_rss_slope_mb_per_minute:
        issues.append(
            f"RSS slope {float(rss_slope):.3f} MB/min exceeds "
            f"{maximum_rss_slope_mb_per_minute:.3f} MB/min"
        )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    arguments = parser.parse_args()
    report = json.loads(arguments.report.read_text(encoding="utf-8"))
    issues = check_performance_report(report)
    result = {"valid": not issues, "report": str(arguments.report), "issues": issues}
    print(json.dumps(result, indent=2, sort_keys=True))
    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
