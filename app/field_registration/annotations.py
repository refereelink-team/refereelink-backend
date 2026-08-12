"""Safe JSON contracts and validation for pitch-registration ground truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from app.field_registration.geometry import convex_hull_coverage
from app.field_registration.pitch_model import PitchDimensions, PitchModel


ANNOTATION_FORMAT_VERSION = 1
VALID_SPLITS = frozenset({"calibration", "train", "validation", "test"})


class AnnotationFormatError(ValueError):
    """Raised when an annotation file is malformed or geometrically unsafe."""


@dataclass(frozen=True)
class FrameAnnotationQuality:
    frame_index: int
    correspondence_count: int
    contact_count: int
    image_coverage: float
    pitch_coverage: float
    homography_available: bool
    reprojection_median_px: float | None
    issues: tuple[str, ...]


@dataclass(frozen=True)
class AnnotationValidationReport:
    valid: bool
    frame_count: int
    annotated_frame_count: int
    contact_count: int
    split: str
    issues: tuple[str, ...]
    frames: tuple[FrameAnnotationQuality, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "frame_count": self.frame_count,
            "annotated_frame_count": self.annotated_frame_count,
            "contact_count": self.contact_count,
            "split": self.split,
            "issues": list(self.issues),
            "frames": [asdict(frame) for frame in self.frames],
        }


def pitch_dimensions_from_payload(payload: Mapping[str, Any]) -> PitchDimensions:
    dimensions = payload.get("pitch_dimensions_m", {})
    if not isinstance(dimensions, Mapping):
        raise AnnotationFormatError("pitch_dimensions_m must be an object")
    return PitchDimensions(
        length_m=float(dimensions.get("length", 105.0)),
        width_m=float(dimensions.get("width", 68.0)),
        penalty_area_depth_m=float(dimensions.get("penalty_area_depth", 16.5)),
        penalty_area_width_m=float(dimensions.get("penalty_area_width", 40.32)),
        goal_area_depth_m=float(dimensions.get("goal_area_depth", 5.5)),
        goal_area_width_m=float(dimensions.get("goal_area_width", 18.32)),
        centre_circle_radius_m=float(dimensions.get("centre_circle_radius", 9.15)),
        penalty_spot_distance_m=float(dimensions.get("penalty_spot_distance", 11.0)),
    )


def correspondence_arrays(
    frame: Mapping[str, Any],
    pitch_model: PitchModel,
) -> tuple[np.ndarray, np.ndarray]:
    image_points: list[tuple[float, float]] = []
    pitch_points: list[tuple[float, float]] = []
    landmarks = pitch_model.landmarks | pitch_model.point_landmarks
    for item in frame.get("correspondences", []):
        if not isinstance(item, Mapping):
            raise AnnotationFormatError("each correspondence must be an object")
        image_xy = np.asarray(item.get("image_xy"), dtype=np.float64)
        if image_xy.shape != (2,) or not np.all(np.isfinite(image_xy)):
            raise AnnotationFormatError("correspondence image_xy must be a finite pair")
        label = str(item.get("label", ""))
        pitch_raw = item.get("pitch_xy_m")
        if pitch_raw is None:
            if label not in landmarks:
                raise AnnotationFormatError(f"unknown pitch landmark: {label!r}")
            pitch_xy = np.asarray(landmarks[label], dtype=np.float64)
        else:
            pitch_xy = np.asarray(pitch_raw, dtype=np.float64)
            if pitch_xy.shape != (2,) or not np.all(np.isfinite(pitch_xy)):
                raise AnnotationFormatError("correspondence pitch_xy_m must be a finite pair")
        image_points.append((float(image_xy[0]), float(image_xy[1])))
        pitch_points.append((float(pitch_xy[0]), float(pitch_xy[1])))
    return np.asarray(image_points, dtype=np.float64), np.asarray(
        pitch_points, dtype=np.float64
    )


def fit_annotation_homography(
    image_points: np.ndarray,
    pitch_points_m: np.ndarray,
) -> tuple[np.ndarray | None, float | None]:
    """Fit a diagnostic image-to-pitch transform from manual landmarks.

    This transform is useful for annotation QA and metric evaluation, but it
    does not turn sparse human clicks into error-free ground truth.
    """

    if image_points.shape != pitch_points_m.shape or image_points.shape[0] < 4:
        return None, None
    def fit(method: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        try:
            return cv2.findHomography(
                pitch_points_m.astype(np.float64),
                image_points.astype(np.float64),
                method=method,
                ransacReprojThreshold=3.0,
                maxIters=10_000,
                confidence=0.999,
            )
        except cv2.error:
            return None, None

    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    matrix, mask = fit(method)
    # OpenCV USAC can reject exact/regular planar grids in some builds. Use
    # the same guarded RANSAC fallback as the runtime initializer; subsequent
    # residual and coverage checks still decide whether annotation QA passes.
    if matrix is None and method != cv2.RANSAC:
        matrix, mask = fit(cv2.RANSAC)
    if matrix is None or not np.all(np.isfinite(matrix)):
        return None, None
    projected = cv2.perspectiveTransform(
        pitch_points_m.reshape(-1, 1, 2).astype(np.float64), matrix
    ).reshape(-1, 2)
    residual = np.linalg.norm(projected - image_points, axis=1)
    if mask is not None:
        inliers = mask.reshape(-1).astype(bool)
        if np.count_nonzero(inliers) >= 4:
            residual = residual[inliers]
    try:
        image_to_pitch = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None, None
    return image_to_pitch / image_to_pitch[2, 2], float(np.median(residual))


def validate_annotation_payload(
    payload: Mapping[str, Any],
    root: str | Path | None = None,
    require_annotated_frames: bool = True,
    require_contact_points: bool | None = None,
) -> AnnotationValidationReport:
    if require_contact_points is None:
        require_contact_points = require_annotated_frames
    if int(payload.get("version", 0)) != ANNOTATION_FORMAT_VERSION:
        raise AnnotationFormatError("unsupported annotation format version")
    image_size_raw = payload.get("image_size")
    if not isinstance(image_size_raw, Sequence) or len(image_size_raw) != 2:
        raise AnnotationFormatError("image_size must contain width and height")
    image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
    if min(image_size) <= 0:
        raise AnnotationFormatError("image_size values must be positive")
    split = str(payload.get("split", ""))
    if split not in VALID_SPLITS:
        raise AnnotationFormatError(f"split must be one of {sorted(VALID_SPLITS)}")
    frames_raw = payload.get("frames")
    if not isinstance(frames_raw, Sequence) or isinstance(frames_raw, (str, bytes)):
        raise AnnotationFormatError("frames must be an array")

    pitch_model = PitchModel(pitch_dimensions_from_payload(payload))
    root_path = Path(root) if root is not None else None
    frame_reports: list[FrameAnnotationQuality] = []
    global_issues: list[str] = []
    seen_indices: set[int] = set()
    annotated_frames = 0
    total_contacts = 0

    for frame_raw in frames_raw:
        if not isinstance(frame_raw, Mapping):
            raise AnnotationFormatError("each frame must be an object")
        frame_index = int(frame_raw.get("frame_index", -1))
        if frame_index < 0:
            raise AnnotationFormatError("frame_index must be non-negative")
        if frame_index in seen_indices:
            raise AnnotationFormatError(f"duplicate frame_index: {frame_index}")
        seen_indices.add(frame_index)
        issues: list[str] = []
        if root_path is not None:
            image_path = root_path / str(frame_raw.get("image_path", ""))
            if not image_path.is_file():
                issues.append("missing_frame_image")

        image_points, pitch_points = correspondence_arrays(frame_raw, pitch_model)
        correspondence_count = int(image_points.shape[0])
        contacts_raw = frame_raw.get("contact_points", [])
        if not isinstance(contacts_raw, Sequence) or isinstance(contacts_raw, (str, bytes)):
            raise AnnotationFormatError("contact_points must be an array")
        contact_count = len(contacts_raw)
        total_contacts += contact_count
        for contact in contacts_raw:
            if not isinstance(contact, Mapping):
                raise AnnotationFormatError("each contact point must be an object")
            image_xy = np.asarray(contact.get("image_xy"), dtype=np.float64)
            if image_xy.shape != (2,) or not np.all(np.isfinite(image_xy)):
                raise AnnotationFormatError("contact image_xy must be a finite pair")

        image_coverage = convex_hull_coverage(image_points, image_size)
        pitch_coverage = convex_hull_coverage(
            pitch_points,
            (int(round(pitch_model.dimensions.length_m)), int(round(pitch_model.dimensions.width_m))),
        )
        image_to_pitch, residual = fit_annotation_homography(image_points, pitch_points)
        if correspondence_count:
            annotated_frames += 1
        if 0 < correspondence_count < 4:
            issues.append("fewer_than_four_correspondences")
        elif correspondence_count >= 4 and image_to_pitch is None:
            issues.append("degenerate_correspondences")
        if correspondence_count >= 4 and image_coverage < 0.01:
            issues.append("insufficient_image_coverage")
        if residual is not None and residual > 5.0:
            issues.append("high_landmark_residual")
        frame_reports.append(
            FrameAnnotationQuality(
                frame_index=frame_index,
                correspondence_count=correspondence_count,
                contact_count=contact_count,
                image_coverage=image_coverage,
                pitch_coverage=pitch_coverage,
                homography_available=image_to_pitch is not None,
                reprojection_median_px=residual,
                issues=tuple(issues),
            )
        )
    if require_annotated_frames and annotated_frames == 0:
        global_issues.append("no_annotated_frames")
    if require_contact_points and total_contacts == 0:
        global_issues.append("no_contact_points")
    if any(frame.issues for frame in frame_reports):
        global_issues.append("frame_validation_failed")
    return AnnotationValidationReport(
        valid=not global_issues,
        frame_count=len(frame_reports),
        annotated_frame_count=annotated_frames,
        contact_count=total_contacts,
        split=split,
        issues=tuple(global_issues),
        frames=tuple(frame_reports),
    )
