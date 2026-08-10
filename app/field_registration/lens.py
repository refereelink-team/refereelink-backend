"""Validated pinhole/fisheye calibration and cached image rectification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class LensModel(str, Enum):
    PINHOLE = "pinhole"
    FISHEYE = "fisheye"


class LensCalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class LensCalibration:
    lens_model: LensModel
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    image_size: Tuple[int, int]
    reprojection_error_px: Optional[float] = None
    version: int = 1

    def __post_init__(self) -> None:
        matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        distortion = np.asarray(self.distortion_coefficients, dtype=np.float64).reshape(-1)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise LensCalibrationError("camera_matrix must be a finite 3x3 matrix")
        if self.lens_model is LensModel.FISHEYE and distortion.size != 4:
            raise LensCalibrationError("fisheye calibration requires exactly 4 coefficients")
        if self.lens_model is LensModel.PINHOLE and distortion.size < 4:
            raise LensCalibrationError("pinhole calibration requires at least 4 coefficients")
        if not np.all(np.isfinite(distortion)):
            raise LensCalibrationError("distortion coefficients must be finite")
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise LensCalibrationError("calibration image size must be positive")
        if self.reprojection_error_px is not None and (
            not np.isfinite(self.reprojection_error_px) or self.reprojection_error_px < 0
        ):
            raise LensCalibrationError("reprojection error must be finite and non-negative")
        object.__setattr__(self, "camera_matrix", matrix)
        object.__setattr__(self, "distortion_coefficients", distortion)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        reprojection = np.nan if self.reprojection_error_px is None else self.reprojection_error_px
        np.savez(
            target,
            version=np.array(self.version, dtype=np.int64),
            lens_model=np.array(self.lens_model.value),
            camera_matrix=self.camera_matrix,
            distortion_coefficients=self.distortion_coefficients,
            image_width=np.array(self.image_size[0], dtype=np.int64),
            image_height=np.array(self.image_size[1], dtype=np.int64),
            reprojection_error=np.array(reprojection, dtype=np.float64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LensCalibration":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Lens calibration file not found: {source}")
        with np.load(source, allow_pickle=False) as data:
            required = {
                "camera_matrix",
                "distortion_coefficients",
                "image_width",
                "image_height",
            }
            missing = required.difference(data.files)
            if missing:
                raise LensCalibrationError(
                    f"calibration file is missing fields: {', '.join(sorted(missing))}"
                )
            version = int(np.asarray(data["version"]).item()) if "version" in data else 1
            if version != 1:
                raise LensCalibrationError(f"unsupported lens calibration version: {version}")
            model_value = (
                str(np.asarray(data["lens_model"]).item())
                if "lens_model" in data
                else LensModel.PINHOLE.value
            )
            reprojection = None
            if "reprojection_error" in data:
                candidate = float(np.asarray(data["reprojection_error"]).item())
                if np.isfinite(candidate):
                    reprojection = candidate
            return cls(
                lens_model=LensModel(model_value),
                camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
                distortion_coefficients=np.asarray(
                    data["distortion_coefficients"], dtype=np.float64
                ),
                image_size=(
                    int(np.asarray(data["image_width"]).item()),
                    int(np.asarray(data["image_height"]).item()),
                ),
                reprojection_error_px=reprojection,
                version=version,
            )

    def scaled_camera_matrix(self, image_size: Tuple[int, int]) -> np.ndarray:
        width, height = image_size
        if width <= 0 or height <= 0:
            raise LensCalibrationError("target image size must be positive")
        source_width, source_height = self.image_size
        if abs(source_width / source_height - width / height) > 0.01:
            raise LensCalibrationError(
                "frame aspect ratio does not match calibration: "
                f"calibration={source_width}x{source_height}, frame={width}x{height}"
            )
        matrix = self.camera_matrix.copy()
        matrix[0, :] *= width / source_width
        matrix[1, :] *= height / source_height
        return matrix


class LensUndistorter:
    """Rectify frames and points into one downstream coordinate system."""

    def __init__(self, calibration: LensCalibration, balance: float = 0.0) -> None:
        if not 0.0 <= balance <= 1.0:
            raise LensCalibrationError("balance must be between 0 and 1")
        self.calibration = calibration
        self.balance = float(balance)
        self._cache: dict[
            Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}

    def _maps(
        self, image_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._cache.get(image_size)
        if cached is not None:
            return cached
        width, height = image_size
        matrix = self.calibration.scaled_camera_matrix(image_size)
        distortion = self.calibration.distortion_coefficients
        if self.calibration.lens_model is LensModel.FISHEYE:
            rectified_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                matrix,
                distortion.reshape(4, 1),
                image_size,
                np.eye(3),
                balance=self.balance,
                new_size=image_size,
            )
            map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
                matrix,
                distortion.reshape(4, 1),
                np.eye(3),
                rectified_matrix,
                image_size,
                cv2.CV_32FC1,
            )
        else:
            rectified_matrix, _ = cv2.getOptimalNewCameraMatrix(
                matrix,
                distortion,
                image_size,
                self.balance,
                image_size,
            )
            map_x, map_y = cv2.initUndistortRectifyMap(
                matrix,
                distortion,
                None,
                rectified_matrix,
                image_size,
                cv2.CV_32FC1,
            )
        result = (map_x, map_y, rectified_matrix)
        self._cache[image_size] = result
        return result

    def rectify(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim < 2:
            raise LensCalibrationError("frame must have at least two dimensions")
        height, width = frame.shape[:2]
        map_x, map_y, _ = self._maps((width, height))
        return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    def rectified_camera_matrix(self, image_size: Tuple[int, int]) -> np.ndarray:
        return self._maps(image_size)[2].copy()

    def rectify_points(
        self, points_xy: np.ndarray, image_size: Tuple[int, int]
    ) -> np.ndarray:
        points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
        matrix = self.calibration.scaled_camera_matrix(image_size)
        rectified_matrix = self.rectified_camera_matrix(image_size)
        if self.calibration.lens_model is LensModel.FISHEYE:
            result = cv2.fisheye.undistortPoints(
                points,
                matrix,
                self.calibration.distortion_coefficients.reshape(4, 1),
                P=rectified_matrix,
            )
        else:
            result = cv2.undistortPoints(
                points,
                matrix,
                self.calibration.distortion_coefficients,
                P=rectified_matrix,
            )
        return result.reshape(-1, 2)
