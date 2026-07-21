from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.classification.team_calibration.types import (
    PlayerRole,
    TeamLabel,
    TeamPrototype,
    ValidationReport,
)


BUNDLE_VERSION = "team-calibration-v1"


@dataclass
class CalibrationBundle:
    match_id: str
    camera_id: str
    prototypes: dict[tuple[TeamLabel, PlayerRole], TeamPrototype]
    validation_report: ValidationReport
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    feature_version: str = "hsv-lab+mobilenet-v3-small-native"
    bundle_version: str = BUNDLE_VERSION

    @property
    def team_ready(self) -> bool:
        return all(
            (team, PlayerRole.OUTFIELD) in self.prototypes
            for team in (TeamLabel.HOME, TeamLabel.AWAY)
        ) and self.validation_report.passed

    @property
    def goalkeeper_mapping_ready(self) -> bool:
        return self.validation_report.goalkeeper_mapping_ready

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        metadata: dict[str, Any] = {
            "bundle_version": self.bundle_version,
            "match_id": self.match_id,
            "camera_id": self.camera_id,
            "created_at": self.created_at,
            "feature_version": self.feature_version,
            "validation_report": self.validation_report.__dict__,
            "prototypes": [],
        }
        for index, ((team, role), prototype) in enumerate(sorted(self.prototypes.items(), key=str)):
            color_key = f"color_{index}"
            deep_key = f"deep_{index}"
            if prototype.color_feature is not None:
                arrays[color_key] = np.asarray(prototype.color_feature, dtype=np.float32)
            if prototype.deep_feature is not None:
                arrays[deep_key] = np.asarray(prototype.deep_feature, dtype=np.float32)
            metadata["prototypes"].append(
                {
                    "team": team.value,
                    "role": role.value,
                    "color_key": color_key if color_key in arrays else None,
                    "deep_key": deep_key if deep_key in arrays else None,
                    "track_count": prototype.track_count,
                    "sample_count": prototype.sample_count,
                    "intra_class_dispersion": prototype.intra_class_dispersion,
                    "color_intra_scale": prototype.color_intra_scale,
                    "color_inter_scale": prototype.color_inter_scale,
                    "deep_intra_scale": prototype.deep_intra_scale,
                    "deep_inter_scale": prototype.deep_inter_scale,
                }
            )
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
        with target.open("wb") as handle:
            np.savez(handle, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationBundle":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata_value = data["metadata_json"]
            metadata = json.loads(str(metadata_value.item() if metadata_value.ndim == 0 else metadata_value))
            if metadata.get("bundle_version") != BUNDLE_VERSION:
                raise ValueError("unsupported team calibration bundle version")
            report = ValidationReport(**metadata["validation_report"])
            prototypes: dict[tuple[TeamLabel, PlayerRole], TeamPrototype] = {}
            for item in metadata.get("prototypes", []):
                color = data[item["color_key"]].astype(np.float32) if item.get("color_key") else None
                deep = data[item["deep_key"]].astype(np.float32) if item.get("deep_key") else None
                team = TeamLabel(item["team"])
                role = PlayerRole(item["role"])
                prototypes[(team, role)] = TeamPrototype(
                    team=team,
                    role=role,
                    color_feature=color,
                    deep_feature=deep,
                    track_count=int(item["track_count"]),
                    sample_count=int(item["sample_count"]),
                    intra_class_dispersion=float(item["intra_class_dispersion"]),
                    color_intra_scale=float(item["color_intra_scale"]),
                    color_inter_scale=float(item["color_inter_scale"]),
                    deep_intra_scale=float(item["deep_intra_scale"]),
                    deep_inter_scale=float(item["deep_inter_scale"]),
                )
        return cls(
            match_id=str(metadata["match_id"]),
            camera_id=str(metadata["camera_id"]),
            prototypes=prototypes,
            validation_report=report,
            created_at=str(metadata["created_at"]),
            feature_version=str(metadata["feature_version"]),
            bundle_version=BUNDLE_VERSION,
        )

