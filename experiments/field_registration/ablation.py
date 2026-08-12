"""Machine-readable E0-E7 pitch-registration ablation definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldAblationExperiment:
    experiment_id: str
    perception: str
    temporal: str
    production_candidate: bool
    requires_student_checkpoint: bool = False
    conditional: bool = False


EXPERIMENTS: tuple[FieldAblationExperiment, ...] = (
    FieldAblationExperiment("E0", "legacy_32_keypoint", "none", False),
    FieldAblationExperiment("E1", "pnlcalib_teacher", "single_frame", False),
    FieldAblationExperiment("E2", "tvcalib_teacher", "single_frame", False),
    FieldAblationExperiment(
        "E3", "mobilenet_v3_dual_head", "single_frame", True, True
    ),
    FieldAblationExperiment(
        "E4", "mobilenet_v3_dual_head", "flow_ekf", True, True
    ),
    FieldAblationExperiment(
        "E5", "mobilenet_v3_dual_head", "flow_ekf_point_line", True, True
    ),
    FieldAblationExperiment(
        "E6", "mobilenet_v3_dual_head", "offline_bidirectional_rts", True, True
    ),
    FieldAblationExperiment(
        "E7", "pidnet_s_dual_head", "flow_ekf_point_line", True, True, True
    ),
)


def experiment_by_id(experiment_id: str) -> FieldAblationExperiment:
    for experiment in EXPERIMENTS:
        if experiment.experiment_id == experiment_id:
            return experiment
    raise KeyError(experiment_id)
