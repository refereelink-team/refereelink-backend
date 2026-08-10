"""Multi-anchor initialization for a fixed-position horizontal-pan camera rig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np

from app.field_registration.camera_model import CameraRigProfile
from app.field_registration.lens import LensCalibration, LensModel
from app.field_registration.types import PointObservation


@dataclass(frozen=True)
class PanAnchor:
    anchor_id: str
    observations: Tuple[PointObservation, ...]
    encoder_pan_rad: float | None = None


@dataclass(frozen=True)
class RigCalibrationResult:
    profile: CameraRigProfile
    anchor_pan_rad: dict[str, float]
    anchor_reprojection_error_px: dict[str, float]
    camera_centre_spread_m: float


def _wrap_angle(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def _average_rotations(rotations: Sequence[np.ndarray]) -> np.ndarray:
    mean = np.mean(np.stack(rotations), axis=0)
    left, _, right = np.linalg.svd(mean)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    return rotation


def _rotation_about_z(angle_rad: float) -> np.ndarray:
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def calibrate_fixed_pan_rig(
    camera_id: str,
    lens: LensCalibration,
    anchors: Sequence[PanAnchor],
    pitch_size_m: Tuple[float, float],
    pan_limits_rad: Tuple[float, float] = (-np.pi, np.pi),
) -> RigCalibrationResult:
    """Estimate a practical rig initializer from at least two pan anchors.

    Each planar anchor is solved independently with IPPE, then the camera
    centres and rotations are jointly factored into one fixed centre, one base
    orientation, and per-anchor horizontal pan.  This initializer is intended
    to seed the later point/line refinement; large centre spread is explicitly
    reported instead of hidden.
    """

    if len(anchors) < 2:
        raise ValueError("at least two pan anchors are required")
    if lens.lens_model is LensModel.FISHEYE:
        raise ValueError("fisheye anchor points must be rectified before rig calibration")
    rotations: list[np.ndarray] = []
    centres: list[np.ndarray] = []
    headings: list[float] = []
    errors: dict[str, float] = {}
    for anchor in anchors:
        if len(anchor.observations) < 4:
            raise ValueError(f"anchor {anchor.anchor_id} has fewer than four points")
        object_points = np.asarray(
            [(*observation.pitch_xy_m, 0.0) for observation in anchor.observations],
            dtype=np.float64,
        )
        image_points = np.asarray(
            [observation.image_xy for observation in anchor.observations], dtype=np.float64
        )
        success, rotation_vector, translation = cv2.solvePnP(
            object_points,
            image_points,
            lens.camera_matrix,
            lens.distortion_coefficients,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not success or not np.all(np.isfinite(rotation_vector)) or not np.all(
            np.isfinite(translation)
        ):
            raise ValueError(f"pose estimation failed for anchor {anchor.anchor_id}")
        rotation, _ = cv2.Rodrigues(rotation_vector)
        centre = -rotation.T @ translation.reshape(3)
        if centre[2] < 0:
            raise ValueError(f"anchor {anchor.anchor_id} produced a camera below the pitch")
        projected, _ = cv2.projectPoints(
            object_points,
            rotation_vector,
            translation,
            lens.camera_matrix,
            lens.distortion_coefficients,
        )
        error = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
        errors[anchor.anchor_id] = float(np.mean(error))
        forward_world = rotation.T[:, 2]
        headings.append(float(np.arctan2(forward_world[1], forward_world[0])))
        rotations.append(rotation)
        centres.append(centre)

    encoder_values = [anchor.encoder_pan_rad for anchor in anchors]
    if all(value is not None for value in encoder_values):
        pans = [float(value) for value in encoder_values if value is not None]
        reference_pan = pans[0]
        pans = [_wrap_angle(value - reference_pan) for value in pans]
    else:
        reference_heading = headings[0]
        pans = [_wrap_angle(heading - reference_heading) for heading in headings]
    base_candidates = [
        rotation @ _rotation_about_z(pan)
        for rotation, pan in zip(rotations, pans)
    ]
    base_rotation = _average_rotations(base_candidates)
    centre_array = np.stack(centres)
    centre = np.median(centre_array, axis=0)
    spread = float(np.max(np.linalg.norm(centre_array - centre, axis=1)))
    forward = base_rotation.T[:, 2]
    fixed_tilt = float(np.arctan2(forward[2], np.hypot(forward[0], forward[1])))
    camera_up = -base_rotation.T[:, 1]
    fixed_roll = float(np.arctan2(camera_up[1], max(abs(camera_up[2]), 1e-12)))
    profile = CameraRigProfile(
        camera_id=camera_id,
        image_size=lens.image_size,
        lens_model=lens.lens_model,
        camera_matrix=lens.camera_matrix,
        distortion_coefficients=lens.distortion_coefficients,
        pitch_length_m=float(pitch_size_m[0]),
        pitch_width_m=float(pitch_size_m[1]),
        pan_axis_origin_xyz_m=centre,
        pan_axis_direction=np.array([0.0, 0.0, 1.0]),
        optical_center_offset_m=np.zeros(3),
        base_rotation_world_to_camera=base_rotation,
        fixed_tilt_rad=fixed_tilt,
        fixed_roll_rad=fixed_roll,
        fixed_focal_px=float((lens.camera_matrix[0, 0] + lens.camera_matrix[1, 1]) / 2.0),
        pan_zero_rad=0.0,
        pan_limits_rad=pan_limits_rad,
    )
    return RigCalibrationResult(
        profile=profile,
        anchor_pan_rad={anchor.anchor_id: pan for anchor, pan in zip(anchors, pans)},
        anchor_reprojection_error_px=errors,
        camera_centre_spread_m=spread,
    )
