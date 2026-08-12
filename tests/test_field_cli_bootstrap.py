from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "script",
    (
        "benchmark_pitch_perception.py",
        "benchmark_pitch_registration_long_run.py",
        "build_soccernet_pitch_index.py",
        "build_teacher_distillation_index.py",
        "calibrate_camera_rig.py",
        "check_soccernet_splits.py",
        "create_pitch_annotation_pack.py",
        "create_pitch_training_smoke_fixture.py",
        "evaluate_offline_pitch_registration.py",
        "evaluate_pitch_registration.py",
        "export_pitch_perception.py",
        "run_offline_pitch_registration.py",
        "summarize_pitch_registration_ablation.py",
        "validate_field_research_assets.py",
        "validate_pitch_annotations.py",
        "validate_pitch_perception_deployment.py",
    ),
)
def test_field_tools_run_directly_from_checkout(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
