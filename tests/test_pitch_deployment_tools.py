from __future__ import annotations

from pathlib import Path

import pytest
import torch

from app.field_registration.models import build_pitch_perception_model
from tools.benchmark_pitch_registration_long_run import (
    _counter_ratio,
    _memory_slope_mb_per_minute,
)
from tools.export_pitch_perception import build_trtexec_command, load_trained_model


def test_load_trained_pitch_model_requires_complete_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "invalid.pt"
    torch.save({"format_version": 1}, checkpoint)

    with pytest.raises(ValueError, match="checkpoint is incomplete"):
        load_trained_model(checkpoint, torch.device("cpu"))


def test_load_trained_pitch_model_round_trip(tmp_path: Path) -> None:
    model = build_pitch_perception_model(
        "mobilenet_v3_dual_head",
        semantic_class_count=3,
        landmark_count=2,
        pretrained=False,
    )
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "format_version": 1,
            "architecture": "mobilenet_v3_dual_head",
            "input_size": [128, 64],
            "semantic_labels": ["touchline", "goal_line"],
            "landmark_labels": ["left", "right"],
            "state_dict": model.state_dict(),
            "validation": {"loss": 0.2},
        },
        checkpoint,
    )

    loaded, payload = load_trained_model(checkpoint, torch.device("cpu"))

    assert not loaded.training
    assert payload["input_size"] == [128, 64]


def test_trtexec_command_is_explicit_and_supports_fp16(tmp_path: Path) -> None:
    command = build_trtexec_command(
        "/opt/tensorrt/bin/trtexec",
        tmp_path / "model.onnx",
        tmp_path / "model.engine",
        use_fp16=True,
        workspace_mib=1024,
    )

    assert command[0] == "/opt/tensorrt/bin/trtexec"
    assert "--fp16" in command
    assert "--skipInference" in command
    assert "--memPoolSize=workspace:1024" in command


def test_memory_slope_uses_elapsed_minutes() -> None:
    samples = [(0.0, 100.0), (300.0, 110.0), (600.0, 120.0)]

    assert _memory_slope_mb_per_minute(samples) == pytest.approx(2.0)


def test_memory_slope_rejects_short_smoke_run() -> None:
    assert _memory_slope_mb_per_minute([(0.0, 100.0), (30.0, 105.0)]) is None


def test_counter_ratio_excludes_warmup_counts() -> None:
    assert _counter_ratio(current=330, baseline=30, measured_frames=300) == 1.0
