from __future__ import annotations

import numpy as np
import pytest

from tools.validate_pitch_perception_deployment import (
    compare_output_sets,
    require_active_provider,
    summarize_latencies,
)


def test_summarize_latencies_reports_distribution() -> None:
    summary = summarize_latencies([1.0, 2.0, 3.0, 4.0])

    assert summary["mean_ms"] == 2.5
    assert summary["median_ms"] == 2.5
    assert summary["max_ms"] == 4.0
    assert summary["p95_ms"] == pytest.approx(3.85)


def test_compare_output_sets_reports_numerical_parity() -> None:
    reference = [np.array([[1.0, 2.0]], dtype=np.float32)]
    deployed = [np.array([[1.001, 1.999]], dtype=np.float16)]

    comparison = compare_output_sets(reference, deployed, atol=0.01, rtol=0.01)

    assert comparison[0]["shape"] == [1, 2]
    assert comparison[0]["allclose"] is True
    assert 0.0 < comparison[0]["max_abs_error"] < 0.01


def test_compare_output_sets_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="different shapes"):
        compare_output_sets(
            [np.zeros((1, 2), dtype=np.float32)],
            [np.zeros((2, 1), dtype=np.float32)],
            atol=0.01,
            rtol=0.01,
        )


def test_require_active_provider_rejects_silent_cpu_fallback() -> None:
    with pytest.raises(RuntimeError, match="not active"):
        require_active_provider(
            "CUDAExecutionProvider",
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CPUExecutionProvider"],
        )
