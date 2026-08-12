"""SoccerNet-Calibration V3 contracts and deployment-target conversion.

The adapter intentionally preserves the official 26-class ontology while the
realtime student only learns the 20 painted ground-plane elements. Goal posts
and crossbars remain available in metadata but are never flattened onto the
pitch plane.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from app.field_registration.geometry import transform_points
from app.field_registration.perception import PitchPerceptionVocabulary
from app.field_registration.pitch_model import PitchModel
from experiments.field_registration.dataset import PitchRegistrationDataset
from experiments.field_registration.teacher_cache import load_teacher_frame


SOCCERNET_RAW_LABELS: tuple[str, ...] = (
    "Big rect. left bottom",
    "Big rect. left main",
    "Big rect. left top",
    "Big rect. right bottom",
    "Big rect. right main",
    "Big rect. right top",
    "Circle central",
    "Circle left",
    "Circle right",
    "Goal left crossbar",
    "Goal left post left",
    "Goal left post right",
    "Goal right crossbar",
    "Goal right post left",
    "Goal right post right",
    "Middle line",
    "Side line bottom",
    "Side line left",
    "Side line right",
    "Side line top",
    "Small rect. left bottom",
    "Small rect. left main",
    "Small rect. left top",
    "Small rect. right bottom",
    "Small rect. right main",
    "Small rect. right top",
)

SOCCERNET_TO_DEPLOYMENT: Mapping[str, Optional[str]] = {
    "Big rect. left bottom": "left_penalty_area_bottom",
    "Big rect. left main": "left_penalty_area_front",
    "Big rect. left top": "left_penalty_area_top",
    "Big rect. right bottom": "right_penalty_area_bottom",
    "Big rect. right main": "right_penalty_area_front",
    "Big rect. right top": "right_penalty_area_top",
    "Circle central": "centre_circle",
    "Circle left": "left_penalty_arc",
    "Circle right": "right_penalty_arc",
    "Goal left crossbar": None,
    "Goal left post left": None,
    "Goal left post right": None,
    "Goal right crossbar": None,
    "Goal right post left": None,
    "Goal right post right": None,
    "Middle line": "halfway_line",
    "Side line bottom": "bottom_touchline",
    "Side line left": "left_goal_line",
    "Side line right": "right_goal_line",
    "Side line top": "top_touchline",
    "Small rect. left bottom": "left_goal_area_bottom",
    "Small rect. left main": "left_goal_area_front",
    "Small rect. left top": "left_goal_area_top",
    "Small rect. right bottom": "right_goal_area_bottom",
    "Small rect. right main": "right_goal_area_front",
    "Small rect. right top": "right_goal_area_top",
}

SOCCERNET_CURVE_LABELS = frozenset(
    {"Circle central", "Circle left", "Circle right"}
)
SOCCERNET_GOAL_STRUCTURE_LABELS = frozenset(
    label for label, deployment in SOCCERNET_TO_DEPLOYMENT.items() if deployment is None
)


class SoccerNetFormatError(ValueError):
    pass


@dataclass(frozen=True)
class SoccerNetIndexFrame:
    source_id: str
    sequence_id: str
    frame_index: int
    image_path: str
    annotation_path: str
    image_size: tuple[int, int]
    raw_labels_present: tuple[str, ...]


@dataclass(frozen=True)
class SoccerNetIndex:
    split: str
    root: str
    frames: tuple[SoccerNetIndexFrame, ...]
    format_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "dataset": "SoccerNet-Calibration-V3",
            "split": self.split,
            "root": self.root,
            "raw_label_vocabulary": list(SOCCERNET_RAW_LABELS),
            "deployment_mapping": dict(SOCCERNET_TO_DEPLOYMENT),
            "frames": [
                asdict(frame) | {"image_size": list(frame.image_size)}
                for frame in self.frames
            ],
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "SoccerNetIndex":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if int(payload.get("format_version", 0)) != 1:
            raise SoccerNetFormatError("unsupported SoccerNet index version")
        if tuple(payload.get("raw_label_vocabulary", ())) != SOCCERNET_RAW_LABELS:
            raise SoccerNetFormatError("SoccerNet raw label vocabulary changed")
        if payload.get("deployment_mapping") != dict(SOCCERNET_TO_DEPLOYMENT):
            raise SoccerNetFormatError("SoccerNet deployment mapping changed")
        frames = tuple(
            SoccerNetIndexFrame(
                source_id=str(item["source_id"]),
                sequence_id=str(item["sequence_id"]),
                frame_index=int(item["frame_index"]),
                image_path=str(item["image_path"]),
                annotation_path=str(item["annotation_path"]),
                image_size=tuple(int(value) for value in item["image_size"]),
                raw_labels_present=tuple(str(value) for value in item["raw_labels_present"]),
            )
            for item in payload.get("frames", [])
        )
        if not frames:
            raise SoccerNetFormatError("SoccerNet index contains no frames")
        root = str(payload.get("root", "."))
        if not Path(root).is_absolute():
            root = str((source.parent / root).resolve())
        return cls(str(payload["split"]), root, frames)


def _annotation_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("lines", "pitch_lines", "annotations"):
        nested = payload.get(key)
        if isinstance(nested, Mapping) and any(
            label in nested for label in SOCCERNET_RAW_LABELS
        ):
            return nested
    return payload


def _point_xy(value: object, image_size: tuple[int, int]) -> tuple[float, float]:
    if isinstance(value, Mapping):
        raw = (value.get("x"), value.get("y"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = tuple(value[:2])
    else:
        raise SoccerNetFormatError("line point must be an {x,y} object or pair")
    try:
        x_coord, y_coord = float(raw[0]), float(raw[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise SoccerNetFormatError("line point coordinates must be numeric") from exc
    if not np.all(np.isfinite((x_coord, y_coord))):
        raise SoccerNetFormatError("line point coordinates must be finite")
    width, height = image_size
    # SoccerNet distributions exist in both pixel and normalized variants.
    if 0.0 <= x_coord <= 1.0 and 0.0 <= y_coord <= 1.0:
        x_coord *= width
        y_coord *= height
    return x_coord, y_coord


def parse_soccernet_annotation(
    payload: Mapping[str, Any],
    image_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    mapping = _annotation_mapping(payload)
    parsed: dict[str, np.ndarray] = {}
    unknown = set(mapping) - set(SOCCERNET_RAW_LABELS)
    metadata_keys = {
        "image_path", "image", "frame_index", "sequence_id", "match_id",
        "game_id", "width", "height", "camera", "metadata",
    }
    unknown -= metadata_keys
    if unknown:
        raise SoccerNetFormatError(
            "unknown SoccerNet labels: " + ", ".join(sorted(unknown))
        )
    for label in SOCCERNET_RAW_LABELS:
        values = mapping.get(label, [])
        if values is None:
            values = []
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise SoccerNetFormatError(f"{label!r} must contain a point array")
        points = np.asarray(
            [_point_xy(value, image_size) for value in values],
            dtype=np.float64,
        ).reshape(-1, 2)
        if points.shape[0] == 1:
            raise SoccerNetFormatError(f"{label!r} needs zero or at least two points")
        parsed[label] = points
    return parsed


def deployment_polylines(
    raw: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    lines: dict[str, np.ndarray] = {}
    for raw_label, deployment_label in SOCCERNET_TO_DEPLOYMENT.items():
        if deployment_label is None:
            continue
        points = np.asarray(raw.get(raw_label, np.empty((0, 2))), dtype=np.float64)
        if points.shape[0] < 2:
            continue
        if raw_label in SOCCERNET_CURVE_LABELS and points.shape[0] >= 3:
            centre = np.mean(points, axis=0)
            angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
            points = points[np.argsort(angles)]
        lines[deployment_label] = points
    return lines


def _fit_line(points: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if points.shape[0] < 2:
        return None
    values = cv2.fitLine(
        points.astype(np.float32), cv2.DIST_L2, 0.0, 0.01, 0.01
    ).reshape(-1)
    direction = np.asarray(values[:2], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return None
    return np.asarray(values[2:4], dtype=np.float64), direction / norm


def _line_intersection(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> Optional[np.ndarray]:
    first_origin, first_direction = first
    second_origin, second_direction = second
    system = np.column_stack((first_direction, -second_direction))
    if abs(float(np.linalg.det(system))) < 1e-5:
        return None
    parameters = np.linalg.solve(system, second_origin - first_origin)
    return first_origin + parameters[0] * first_direction


def _landmark_memberships(pitch_model: PitchModel) -> dict[str, tuple[str, ...]]:
    elements = pitch_model.semantic_elements(curve_samples=192)
    memberships: dict[str, tuple[str, ...]] = {}
    for label, point in (pitch_model.landmarks | pitch_model.point_landmarks).items():
        value = np.asarray(point, dtype=np.float64)
        touching = tuple(
            element_label
            for element_label, polyline in elements.items()
            if float(np.min(np.linalg.norm(polyline - value, axis=1))) <= 0.08
        )
        memberships[label] = touching
    return memberships


def derive_visible_landmarks(
    polylines: Mapping[str, np.ndarray],
    image_size: tuple[int, int],
    pitch_model: PitchModel | None = None,
) -> dict[str, np.ndarray]:
    """Derive only unambiguous line/line and halfway/circle intersections."""

    model = pitch_model or PitchModel()
    memberships = _landmark_memberships(model)
    curve_labels = {"centre_circle", "left_penalty_arc", "right_penalty_arc"}
    fitted = {
        label: line
        for label, points in polylines.items()
        if label not in curve_labels
        for line in [_fit_line(np.asarray(points))]
        if line is not None
    }
    width, height = image_size

    def visible(point: np.ndarray) -> bool:
        return bool(
            np.all(np.isfinite(point))
            and -4.0 <= point[0] <= width + 4.0
            and -4.0 <= point[1] <= height + 4.0
        )

    result: dict[str, np.ndarray] = {}
    for landmark, touching in memberships.items():
        straight = [label for label in touching if label in fitted]
        if len(straight) < 2:
            continue
        point = _line_intersection(fitted[straight[0]], fitted[straight[1]])
        if point is not None and visible(point):
            result[landmark] = point

    circle_points = polylines.get("centre_circle")
    halfway = fitted.get("halfway_line")
    centre_top = result.get("centre_top")
    centre_bottom = result.get("centre_bottom")
    if (
        circle_points is not None
        and len(circle_points) >= 5
        and halfway is not None
        and centre_top is not None
        and centre_bottom is not None
    ):
        ellipse = cv2.fitEllipse(np.asarray(circle_points, dtype=np.float32))
        centre = np.asarray(ellipse[0], dtype=np.float64)
        axes = np.asarray(ellipse[1], dtype=np.float64) / 2.0
        angle = np.deg2rad(float(ellipse[2]))
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float64,
        )
        origin, direction = halfway
        local_origin = rotation.T @ (origin - centre)
        local_direction = rotation.T @ direction
        coefficients = (
            float(np.sum((local_direction / axes) ** 2)),
            float(2.0 * np.sum(local_origin * local_direction / axes**2)),
            float(np.sum((local_origin / axes) ** 2) - 1.0),
        )
        roots = np.roots(coefficients)
        candidates = [origin + float(root.real) * direction for root in roots if abs(root.imag) < 1e-6]
        if len(candidates) == 2 and all(visible(point) for point in candidates):
            axis = centre_bottom - centre_top
            denominator = max(float(axis @ axis), 1e-8)
            ordered = sorted(candidates, key=lambda point: float((point - centre_top) @ axis / denominator))
            result["centre_circle_top"] = ordered[0]
            result["centre_circle_bottom"] = ordered[1]
    return result


def teacher_landmarks(
    teacher_cache_path: str | Path,
    pitch_model: PitchModel,
    image_size: tuple[int, int],
    *,
    minimum_confidence: float = 0.65,
) -> dict[str, np.ndarray]:
    teacher = load_teacher_frame(teacher_cache_path)
    if teacher.confidence < minimum_confidence or teacher.image_to_pitch is None:
        return {}
    if teacher.image_size != image_size:
        raise SoccerNetFormatError("teacher cache image size does not match dataset frame")
    pitch_to_image = np.linalg.inv(teacher.image_to_pitch)
    labels = pitch_model.landmarks | pitch_model.point_landmarks
    projected = transform_points(np.asarray(list(labels.values())), pitch_to_image)
    width, height = image_size
    return {
        label: point
        for label, point in zip(labels, projected)
        if np.all(np.isfinite(point))
        and 0.0 <= point[0] < width
        and 0.0 <= point[1] < height
    }


def assert_sequence_disjoint(indices: Sequence[SoccerNetIndex]) -> None:
    owners: dict[str, str] = {}
    for index in indices:
        for sequence_id in {frame.sequence_id for frame in index.frames}:
            existing = owners.setdefault(sequence_id, index.split)
            if existing != index.split:
                raise SoccerNetFormatError(
                    f"sequence {sequence_id!r} leaks across {existing!r} and {index.split!r}"
                )


class SoccerNetCalibrationDataset(Dataset[dict[str, torch.Tensor]]):
    """Draw student targets from visible SoccerNet line annotations."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        expected_split: str,
        input_size: tuple[int, int] = (512, 288),
        output_stride: int = 4,
        line_width_px: int = 3,
        teacher_cache_root: str | Path | None = None,
    ) -> None:
        self.index = SoccerNetIndex.load(index_path)
        if self.index.split != expected_split:
            raise ValueError(
                f"index belongs to {self.index.split!r}, not {expected_split!r}"
            )
        width, height = input_size
        if width % 32 or height % 32:
            raise ValueError("input width and height must be divisible by 32")
        self.input_size = int(width), int(height)
        self.output_stride = int(output_stride)
        self.line_width_px = max(int(line_width_px), 1)
        self.root = Path(self.index.root)
        self.teacher_cache_root = (
            Path(teacher_cache_root) if teacher_cache_root is not None else None
        )
        self.pitch_model = PitchModel()
        self.vocabulary = PitchPerceptionVocabulary.from_pitch_model(self.pitch_model)

    def __len__(self) -> int:
        return len(self.index.frames)

    def _teacher_path(self, frame: SoccerNetIndexFrame) -> Optional[Path]:
        if self.teacher_cache_root is None:
            return None
        candidate = self.teacher_cache_root / frame.source_id / f"{frame.frame_index:08d}.json"
        return candidate if candidate.is_file() else None

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        frame = self.index.frames[index]
        image_path = self.root / frame.image_path
        annotation_path = self.root / frame.annotation_path
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        source_height, source_width = image.shape[:2]
        if (source_width, source_height) != frame.image_size:
            raise SoccerNetFormatError("indexed image dimensions changed")
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        raw = parse_soccernet_annotation(payload, frame.image_size)
        polylines = deployment_polylines(raw)
        landmarks = derive_visible_landmarks(polylines, frame.image_size, self.pitch_model)
        teacher_path = self._teacher_path(frame)
        if teacher_path is not None:
            landmarks.update(
                teacher_landmarks(teacher_path, self.pitch_model, frame.image_size)
            )

        input_width, input_height = self.input_size
        scale_xy = np.asarray(
            [input_width / source_width, input_height / source_height], dtype=np.float64
        )
        semantic_target = np.zeros((input_height, input_width), dtype=np.int32)
        for class_index, label in enumerate(self.vocabulary.semantic_labels, start=1):
            points = polylines.get(label)
            if points is None or points.shape[0] < 2:
                continue
            scaled = np.rint(points * scale_xy).astype(np.int32)
            cv2.polylines(
                semantic_target,
                [scaled.reshape(-1, 1, 2)],
                isClosed=label == "centre_circle",
                color=class_index,
                thickness=self.line_width_px,
                lineType=cv2.LINE_8,
            )

        output_width = input_width // self.output_stride
        output_height = input_height // self.output_stride
        landmark_count = len(self.vocabulary.landmark_labels)
        heatmaps = np.zeros((landmark_count, output_height, output_width), dtype=np.float32)
        offsets = np.zeros((landmark_count, 2, output_height, output_width), dtype=np.float32)
        offset_mask = np.zeros((landmark_count, output_height, output_width), dtype=np.float32)
        visibility = np.zeros(landmark_count, dtype=np.float32)
        for landmark_index, label in enumerate(self.vocabulary.landmark_labels):
            image_xy = landmarks.get(label)
            if image_xy is None:
                continue
            output_xy = image_xy * scale_xy / self.output_stride
            if not (
                0.0 <= output_xy[0] < output_width
                and 0.0 <= output_xy[1] < output_height
            ):
                continue
            PitchRegistrationDataset._draw_gaussian(
                heatmaps[landmark_index], tuple(output_xy)
            )
            integer_xy = np.rint(output_xy).astype(int)
            integer_xy[0] = np.clip(integer_xy[0], 0, output_width - 1)
            integer_xy[1] = np.clip(integer_xy[1], 0, output_height - 1)
            heatmaps[landmark_index, integer_xy[1], integer_xy[0]] = 1.0
            offsets[landmark_index, :, integer_xy[1], integer_xy[0]] = output_xy - integer_xy
            offset_mask[landmark_index, integer_xy[1], integer_xy[0]] = 1.0
            visibility[landmark_index] = 1.0

        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        return {
            "image": (image_tensor - mean) / std,
            "semantic_target": torch.from_numpy(semantic_target).long(),
            "landmark_heatmaps": torch.from_numpy(heatmaps),
            "landmark_offsets": torch.from_numpy(offsets).reshape(
                landmark_count * 2, output_height, output_width
            ),
            "offset_mask": torch.from_numpy(offset_mask),
            "landmark_visibility": torch.from_numpy(visibility),
            "frame_index": torch.tensor(frame.frame_index, dtype=torch.int64),
        }
