"""Safe JSON/NPZ exchange format for isolated pitch-calibration teachers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from app.field_registration.geometry import transform_points
from app.field_registration.pitch_model import PitchModel
from app.field_registration.types import LineObservation, PointObservation
from experiments.field_registration.research_assets import sha256_file


class TeacherCacheError(ValueError):
    """Raised when teacher output is malformed, incomplete, or tampered with."""


@dataclass(frozen=True)
class TeacherFrame:
    teacher_id: str
    teacher_version: str
    source_id: str
    frame_index: int
    image_size: tuple[int, int]
    confidence: float
    points: tuple[PointObservation, ...] = ()
    lines: tuple[LineObservation, ...] = ()
    image_to_pitch: Optional[np.ndarray] = None
    camera_parameters: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, str | float | int | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise TeacherCacheError("teacher image_size must be positive")
        if self.frame_index < 0:
            raise TeacherCacheError("teacher frame_index must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise TeacherCacheError("teacher confidence must be between zero and one")
        if self.image_to_pitch is not None:
            matrix = np.asarray(self.image_to_pitch, dtype=np.float64)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise TeacherCacheError("image_to_pitch must be a finite 3x3 matrix")
            if abs(float(matrix[2, 2])) < 1e-12:
                raise TeacherCacheError("image_to_pitch has an invalid scale")
            object.__setattr__(self, "image_to_pitch", matrix / matrix[2, 2])


def _cache_paths(path: str | Path) -> tuple[Path, Path]:
    value = Path(path)
    base = value.with_suffix("") if value.suffix in {".json", ".npz"} else value
    return base.with_suffix(".json"), base.with_suffix(".npz")


def save_teacher_frame(frame: TeacherFrame, path: str | Path) -> tuple[Path, Path]:
    json_path, npz_path = _cache_paths(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    point_payload: list[dict[str, Any]] = []
    for observation in frame.points:
        point_payload.append(
            {
                "label": observation.label,
                "image_xy": list(observation.image_xy),
                "pitch_xy_m": list(observation.pitch_xy_m),
                "confidence": observation.confidence,
                "sigma_px": observation.sigma_px,
                "source": observation.source,
            }
        )
    line_payload: list[dict[str, Any]] = []
    for index, observation in enumerate(frame.lines):
        image_key = f"line_{index}_image"
        pitch_key = f"line_{index}_pitch"
        arrays[image_key] = observation.image_points.astype(np.float32)
        arrays[pitch_key] = observation.pitch_points_m.astype(np.float32)
        line_payload.append(
            {
                "label": observation.label,
                "image_key": image_key,
                "pitch_key": pitch_key,
                "confidence": observation.confidence,
                "source": observation.source,
            }
        )
    if frame.image_to_pitch is not None:
        arrays["image_to_pitch"] = frame.image_to_pitch.astype(np.float64)
    np.savez_compressed(npz_path, **arrays)
    payload = {
        "format_version": 1,
        "teacher_id": frame.teacher_id,
        "teacher_version": frame.teacher_version,
        "source_id": frame.source_id,
        "frame_index": frame.frame_index,
        "image_size": list(frame.image_size),
        "confidence": frame.confidence,
        "points": point_payload,
        "lines": line_payload,
        "camera_parameters": dict(frame.camera_parameters),
        "metadata": dict(frame.metadata),
        "arrays_file": npz_path.name,
        "arrays_sha256": sha256_file(npz_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json_path, npz_path


def load_teacher_frame(path: str | Path) -> TeacherFrame:
    json_path, default_npz_path = _cache_paths(path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if int(payload.get("format_version", 0)) != 1:
        raise TeacherCacheError("unsupported teacher cache format")
    npz_path = json_path.parent / str(payload.get("arrays_file", default_npz_path.name))
    if not npz_path.is_file():
        raise TeacherCacheError(f"teacher array file is missing: {npz_path}")
    if sha256_file(npz_path) != str(payload.get("arrays_sha256", "")):
        raise TeacherCacheError("teacher array SHA-256 mismatch")
    with np.load(npz_path, allow_pickle=False) as arrays:
        points = tuple(
            PointObservation(
                label=str(item["label"]),
                image_xy=tuple(float(value) for value in item["image_xy"]),
                pitch_xy_m=tuple(float(value) for value in item["pitch_xy_m"]),
                confidence=float(item.get("confidence", 1.0)),
                sigma_px=float(item.get("sigma_px", 1.0)),
                source=str(item.get("source", payload["teacher_id"])),
            )
            for item in payload.get("points", [])
        )
        lines = tuple(
            LineObservation(
                label=str(item["label"]),
                image_points=np.asarray(arrays[str(item["image_key"])]),
                pitch_points_m=np.asarray(arrays[str(item["pitch_key"])]),
                confidence=float(item.get("confidence", 1.0)),
                source=str(item.get("source", payload["teacher_id"])),
            )
            for item in payload.get("lines", [])
        )
        image_to_pitch = (
            np.asarray(arrays["image_to_pitch"], dtype=np.float64)
            if "image_to_pitch" in arrays.files
            else None
        )
    return TeacherFrame(
        teacher_id=str(payload["teacher_id"]),
        teacher_version=str(payload["teacher_version"]),
        source_id=str(payload["source_id"]),
        frame_index=int(payload["frame_index"]),
        image_size=tuple(int(value) for value in payload["image_size"]),
        confidence=float(payload["confidence"]),
        points=points,
        lines=lines,
        image_to_pitch=image_to_pitch,
        camera_parameters={
            str(key): float(value)
            for key, value in payload.get("camera_parameters", {}).items()
        },
        metadata=payload.get("metadata", {}),
    )


@dataclass(frozen=True)
class TeacherAgreement:
    accepted: bool
    median_grid_disagreement_px: Optional[float]
    p95_grid_disagreement_px: Optional[float]
    common_line_labels: int
    rejection_reason: Optional[str] = None


def compare_teacher_frames(
    primary: TeacherFrame,
    secondary: TeacherFrame,
    *,
    pitch_model: PitchModel | None = None,
    max_median_grid_disagreement_px: float = 12.0,
    max_p95_grid_disagreement_px: float = 24.0,
    minimum_common_line_labels: int = 2,
) -> TeacherAgreement:
    if primary.source_id != secondary.source_id or primary.frame_index != secondary.frame_index:
        return TeacherAgreement(False, None, None, 0, "frame_identity_mismatch")
    if primary.image_size != secondary.image_size:
        return TeacherAgreement(False, None, None, 0, "image_size_mismatch")
    common_labels = len({line.label for line in primary.lines} & {line.label for line in secondary.lines})
    if primary.image_to_pitch is None or secondary.image_to_pitch is None:
        return TeacherAgreement(False, None, None, common_labels, "missing_homography")
    model = pitch_model or PitchModel()
    grid = model.grid(22, 15)
    try:
        primary_image = transform_points(grid, np.linalg.inv(primary.image_to_pitch))
        secondary_image = transform_points(grid, np.linalg.inv(secondary.image_to_pitch))
    except (np.linalg.LinAlgError, ValueError):
        return TeacherAgreement(False, None, None, common_labels, "invalid_homography")
    width, height = primary.image_size
    visible = (
        np.all(np.isfinite(primary_image), axis=1)
        & np.all(np.isfinite(secondary_image), axis=1)
        & (primary_image[:, 0] >= 0.0)
        & (primary_image[:, 0] < width)
        & (primary_image[:, 1] >= 0.0)
        & (primary_image[:, 1] < height)
        & (secondary_image[:, 0] >= 0.0)
        & (secondary_image[:, 0] < width)
        & (secondary_image[:, 1] >= 0.0)
        & (secondary_image[:, 1] < height)
    )
    if int(np.count_nonzero(visible)) < 8:
        return TeacherAgreement(False, None, None, common_labels, "insufficient_shared_field")
    errors = np.linalg.norm(primary_image[visible] - secondary_image[visible], axis=1)
    median = float(np.median(errors))
    p95 = float(np.percentile(errors, 95))
    if common_labels < minimum_common_line_labels:
        return TeacherAgreement(False, median, p95, common_labels, "insufficient_common_lines")
    if median > max_median_grid_disagreement_px or p95 > max_p95_grid_disagreement_px:
        return TeacherAgreement(False, median, p95, common_labels, "geometry_disagreement")
    return TeacherAgreement(True, median, p95, common_labels)
