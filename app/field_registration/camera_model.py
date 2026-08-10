"""Physical fixed-rig camera model with horizontal pan as runtime state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from app.field_registration.lens import LensModel


def _array(value: object, shape: Tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return result


@dataclass(frozen=True)
class CameraRigProfile:
    camera_id: str
    image_size: Tuple[int, int]
    lens_model: LensModel
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    pitch_length_m: float
    pitch_width_m: float
    pan_axis_origin_xyz_m: np.ndarray
    pan_axis_direction: np.ndarray
    optical_center_offset_m: np.ndarray
    base_rotation_world_to_camera: np.ndarray
    fixed_tilt_rad: float = 0.0
    fixed_roll_rad: float = 0.0
    fixed_focal_px: float | None = None
    pan_zero_rad: float = 0.0
    pan_limits_rad: Tuple[float, float] = (-np.pi, np.pi)
    version: int = 1

    def __post_init__(self) -> None:
        matrix = _array(self.camera_matrix, (3, 3), "camera_matrix")
        rotation = _array(
            self.base_rotation_world_to_camera,
            (3, 3),
            "base_rotation_world_to_camera",
        )
        origin = _array(self.pan_axis_origin_xyz_m, (3,), "pan_axis_origin_xyz_m")
        axis = _array(self.pan_axis_direction, (3,), "pan_axis_direction")
        offset = _array(self.optical_center_offset_m, (3,), "optical_center_offset_m")
        distortion = np.asarray(self.distortion_coefficients, dtype=np.float64).reshape(-1)
        if self.lens_model is LensModel.FISHEYE and distortion.size != 4:
            raise ValueError("fisheye rig requires exactly 4 distortion coefficients")
        if self.lens_model is LensModel.PINHOLE and distortion.size < 4:
            raise ValueError("pinhole rig requires at least 4 distortion coefficients")
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            raise ValueError("pan axis direction cannot be zero")
        if min(self.image_size) <= 0:
            raise ValueError("image_size must be positive")
        if self.pitch_length_m <= 0 or self.pitch_width_m <= 0:
            raise ValueError("pitch dimensions must be positive")
        if self.pan_limits_rad[0] >= self.pan_limits_rad[1]:
            raise ValueError("pan limits must be increasing")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
            raise ValueError("base rotation must be orthonormal")
        if np.linalg.det(rotation) < 0.999 or np.linalg.det(rotation) > 1.001:
            raise ValueError("base rotation must have determinant +1")
        if self.fixed_focal_px is not None and self.fixed_focal_px <= 0:
            raise ValueError("fixed focal length must be positive")
        object.__setattr__(self, "camera_matrix", matrix)
        object.__setattr__(self, "distortion_coefficients", distortion)
        object.__setattr__(self, "pan_axis_origin_xyz_m", origin)
        object.__setattr__(self, "pan_axis_direction", axis / axis_norm)
        object.__setattr__(self, "optical_center_offset_m", offset)
        object.__setattr__(self, "base_rotation_world_to_camera", rotation)

    @staticmethod
    def _axis_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
        matrix, _ = cv2.Rodrigues(axis.reshape(3, 1) * float(angle_rad))
        return matrix

    def effective_pan(self, pan_rad: float) -> float:
        return float(pan_rad + self.pan_zero_rad)

    def camera_center_world(self, pan_rad: float) -> np.ndarray:
        rotation = self._axis_rotation(self.pan_axis_direction, self.effective_pan(pan_rad))
        return self.pan_axis_origin_xyz_m + rotation @ self.optical_center_offset_m

    def world_to_camera_rotation(self, pan_rad: float) -> np.ndarray:
        rotation_world = self._axis_rotation(
            self.pan_axis_direction, -self.effective_pan(pan_rad)
        )
        return self.base_rotation_world_to_camera @ rotation_world

    def intrinsic_matrix(self, image_size: Tuple[int, int] | None = None) -> np.ndarray:
        target_size = image_size or self.image_size
        source_width, source_height = self.image_size
        target_width, target_height = target_size
        if abs(source_width / source_height - target_width / target_height) > 0.01:
            raise ValueError("target image aspect ratio does not match rig profile")
        matrix = self.camera_matrix.copy()
        matrix[0, :] *= target_width / source_width
        matrix[1, :] *= target_height / source_height
        if self.fixed_focal_px is not None:
            scale = target_width / source_width
            matrix[0, 0] = self.fixed_focal_px * scale
            matrix[1, 1] = self.fixed_focal_px * scale
        return matrix

    def extrinsics(self, pan_rad: float) -> Tuple[np.ndarray, np.ndarray]:
        rotation = self.world_to_camera_rotation(pan_rad)
        centre = self.camera_center_world(pan_rad)
        translation = -rotation @ centre
        return rotation, translation

    def pitch_to_image_homography(
        self, pan_rad: float, image_size: Tuple[int, int] | None = None
    ) -> np.ndarray:
        rotation, translation = self.extrinsics(pan_rad)
        plane_transform = np.column_stack((rotation[:, 0], rotation[:, 1], translation))
        homography = self.intrinsic_matrix(image_size) @ plane_transform
        if abs(homography[2, 2]) > 1e-12:
            homography = homography / homography[2, 2]
        return homography

    def image_to_pitch_homography(
        self, pan_rad: float, image_size: Tuple[int, int] | None = None
    ) -> np.ndarray:
        homography = self.pitch_to_image_homography(pan_rad, image_size)
        if np.linalg.cond(homography) > 1e14:
            raise ValueError("camera pose produces a singular pitch homography")
        inverse = np.linalg.inv(homography)
        return inverse / inverse[2, 2]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "camera_id": self.camera_id,
            "image_size": list(self.image_size),
            "lens_model": self.lens_model.value,
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.distortion_coefficients.tolist(),
            "pitch_length_m": self.pitch_length_m,
            "pitch_width_m": self.pitch_width_m,
            "pan_axis_origin_xyz_m": self.pan_axis_origin_xyz_m.tolist(),
            "pan_axis_direction": self.pan_axis_direction.tolist(),
            "optical_center_offset_m": self.optical_center_offset_m.tolist(),
            "base_rotation_world_to_camera": self.base_rotation_world_to_camera.tolist(),
            "fixed_tilt_rad": self.fixed_tilt_rad,
            "fixed_roll_rad": self.fixed_roll_rad,
            "fixed_focal_px": self.fixed_focal_px,
            "pan_zero_rad": self.pan_zero_rad,
            "pan_limits_rad": list(self.pan_limits_rad),
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CameraRigProfile":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("version", 0)) != 1:
            raise ValueError("unsupported camera rig profile version")
        return cls(
            camera_id=str(payload["camera_id"]),
            image_size=tuple(int(value) for value in payload["image_size"]),
            lens_model=LensModel(payload["lens_model"]),
            camera_matrix=np.asarray(payload["camera_matrix"], dtype=np.float64),
            distortion_coefficients=np.asarray(
                payload["distortion_coefficients"], dtype=np.float64
            ),
            pitch_length_m=float(payload["pitch_length_m"]),
            pitch_width_m=float(payload["pitch_width_m"]),
            pan_axis_origin_xyz_m=np.asarray(
                payload["pan_axis_origin_xyz_m"], dtype=np.float64
            ),
            pan_axis_direction=np.asarray(payload["pan_axis_direction"], dtype=np.float64),
            optical_center_offset_m=np.asarray(
                payload["optical_center_offset_m"], dtype=np.float64
            ),
            base_rotation_world_to_camera=np.asarray(
                payload["base_rotation_world_to_camera"], dtype=np.float64
            ),
            fixed_tilt_rad=float(payload.get("fixed_tilt_rad", 0.0)),
            fixed_roll_rad=float(payload.get("fixed_roll_rad", 0.0)),
            fixed_focal_px=(
                None
                if payload.get("fixed_focal_px") is None
                else float(payload["fixed_focal_px"])
            ),
            pan_zero_rad=float(payload.get("pan_zero_rad", 0.0)),
            pan_limits_rad=tuple(float(value) for value in payload["pan_limits_rad"]),
            version=1,
        )
