"""Camera calibration and wide-angle image rectification helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraCalibrationError(ValueError):
    """Raised when a calibration file cannot be used for a frame."""


@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    image_size: Tuple[int, int]
    reprojection_error: Optional[float] = None

    @classmethod
    def load(cls, path: str | Path) -> "CameraCalibration":
        calibration_path = Path(path)
        if not calibration_path.exists():
            raise FileNotFoundError(f"Camera calibration file not found: {calibration_path}")

        with np.load(calibration_path, allow_pickle=False) as data:
            required = {"camera_matrix", "distortion_coefficients", "image_width", "image_height"}
            missing = required.difference(data.files)
            if missing:
                raise CameraCalibrationError(
                    f"Calibration file is missing fields: {', '.join(sorted(missing))}"
                )

            camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
            distortion = np.asarray(data["distortion_coefficients"], dtype=np.float64)
            image_size = (
                int(np.asarray(data["image_width"]).item()),
                int(np.asarray(data["image_height"]).item()),
            )
            reprojection_error = None
            if "reprojection_error" in data.files:
                value = float(np.asarray(data["reprojection_error"]).item())
                if np.isfinite(value):
                    reprojection_error = value

        if camera_matrix.shape != (3, 3):
            raise CameraCalibrationError("camera_matrix must have shape (3, 3)")
        if distortion.size < 4:
            raise CameraCalibrationError("distortion_coefficients must contain at least 4 values")
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise CameraCalibrationError("Calibration image size must be positive")

        return cls(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion,
            image_size=image_size,
            reprojection_error=reprojection_error,
        )


class CameraUndistorter:
    """Rectify frames with optional calibration and resolution-aware maps.

    A missing calibration file is intentionally a non-fatal condition so the
    pipeline can be brought up before camera calibration is complete.  When a
    calibration is present, all downstream consumers receive the rectified
    frame and therefore share one coordinate system.
    """

    def __init__(
        self,
        calibration_path: Optional[str] = None,
        enabled: bool = True,
        alpha: float = 0.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.alpha = float(alpha)
        if not 0.0 <= self.alpha <= 1.0:
            raise CameraCalibrationError("calibration alpha must be between 0 and 1")
        self.calibration: Optional[CameraCalibration] = None
        self._maps: dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

        if not self.enabled:
            return
        if not calibration_path:
            logger.warning("Camera calibration path is empty; undistortion is bypassed")
            return
        try:
            self.calibration = CameraCalibration.load(calibration_path)
            logger.info(
                "Loaded camera calibration from %s (size=%sx%s, rms=%s)",
                calibration_path,
                self.calibration.image_size[0],
                self.calibration.image_size[1],
                self.calibration.reprojection_error,
            )
        except FileNotFoundError:
            logger.warning(
                "Camera calibration file %s is unavailable; undistortion is bypassed",
                calibration_path,
            )
        except CameraCalibrationError:
            raise

    @property
    def available(self) -> bool:
        return self.enabled and self.calibration is not None

    def _scaled_camera_matrix(self, width: int, height: int) -> np.ndarray:
        assert self.calibration is not None
        calibrated_width, calibrated_height = self.calibration.image_size
        calibrated_ratio = calibrated_width / calibrated_height
        frame_ratio = width / height
        if abs(calibrated_ratio - frame_ratio) > 0.01:
            raise CameraCalibrationError(
                "Frame aspect ratio does not match calibration: "
                f"calibration={calibrated_width}x{calibrated_height}, "
                f"frame={width}x{height}"
            )

        matrix = self.calibration.camera_matrix.copy()
        matrix[0, :] *= width / calibrated_width
        matrix[1, :] *= height / calibrated_height
        return matrix

    def _map_for_shape(self, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (width, height)
        cached = self._maps.get(key)
        if cached is not None:
            return cached
        if self.calibration is None:
            raise CameraCalibrationError("No camera calibration is loaded")

        camera_matrix = self._scaled_camera_matrix(width, height)
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            self.calibration.distortion_coefficients,
            (width, height),
            self.alpha,
            (width, height),
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            camera_matrix,
            self.calibration.distortion_coefficients,
            None,
            new_camera_matrix,
            (width, height),
            cv2.CV_32FC1,
        )
        self._maps[key] = (map_x, map_y)
        return map_x, map_y

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.available:
            return frame
        if frame.ndim < 2:
            raise CameraCalibrationError("Frame must have at least two dimensions")
        height, width = frame.shape[:2]
        map_x, map_y = self._map_for_shape(width, height)
        return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)


def build_undistorter(
    calibration_path: Optional[str],
    enabled: bool = True,
    alpha: float = 0.0,
) -> CameraUndistorter:
    return CameraUndistorter(
        calibration_path=calibration_path,
        enabled=enabled,
        alpha=alpha,
    )
