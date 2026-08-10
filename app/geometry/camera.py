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
    lens_model: str = "pinhole"

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
            lens_model = (
                str(np.asarray(data["lens_model"]).item())
                if "lens_model" in data.files
                else "pinhole"
            )

        if camera_matrix.shape != (3, 3):
            raise CameraCalibrationError("camera_matrix must have shape (3, 3)")
        if lens_model not in {"pinhole", "fisheye"}:
            raise CameraCalibrationError("lens_model must be 'pinhole' or 'fisheye'")
        if lens_model == "fisheye" and distortion.size != 4:
            raise CameraCalibrationError("fisheye calibration must contain exactly 4 values")
        if lens_model == "pinhole" and distortion.size < 4:
            raise CameraCalibrationError("distortion_coefficients must contain at least 4 values")
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise CameraCalibrationError("Calibration image size must be positive")

        return cls(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion,
            image_size=image_size,
            reprojection_error=reprojection_error,
            lens_model=lens_model,
        )


@dataclass(frozen=True)
class CameraMotionEstimate:
    """Global image motion measured since the last pitch refresh."""

    shift_x_px: float
    shift_y_px: float
    response: float
    threshold_px: float
    minimum_response: float

    @property
    def magnitude_px(self) -> float:
        return float(np.hypot(self.shift_x_px, self.shift_y_px))

    @property
    def requires_refresh(self) -> bool:
        return (
            self.response >= self.minimum_response
            and self.magnitude_px >= self.threshold_px
        )


class CameraMotionEstimator:
    """Detect global pan/rotation-induced image drift cheaply.

    ``phaseCorrelate`` is used against the frame saved at the last successful
    pitch refresh.  This catches cumulative horizontal camera motion during
    low-frequency pitch inference, so a stale homography is never reused after
    the view has moved materially.  Player motion is treated as noise through
    the phase-correlation response threshold.
    """

    def __init__(
        self,
        threshold_px: float = 6.0,
        minimum_response: float = 0.15,
        analysis_width: int = 320,
    ) -> None:
        if threshold_px <= 0:
            raise ValueError("camera motion threshold must be positive")
        if not 0.0 <= minimum_response <= 1.0:
            raise ValueError("camera motion minimum response must be between 0 and 1")
        self.threshold_px = float(threshold_px)
        self.minimum_response = float(minimum_response)
        self.analysis_width = max(int(analysis_width), 32)
        self._reference_gray: Optional[np.ndarray] = None
        self._window: Optional[np.ndarray] = None

    @staticmethod
    def _gray(frame: np.ndarray, analysis_width: int) -> tuple[np.ndarray, float]:
        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3 and frame.shape[2] >= 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            raise CameraCalibrationError("Frame must be a grayscale or BGR image")

        height, width = gray.shape[:2]
        if height <= 0 or width <= 0:
            raise CameraCalibrationError("Frame dimensions must be positive")
        target_width = min(width, analysis_width)
        target_height = max(1, int(round(height * target_width / width)))
        scale = target_width / width
        resized = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(resized, (3, 3), 0).astype(np.float32), scale

    def _prepare(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        gray, scale = self._gray(frame, self.analysis_width)
        if self._reference_gray is None or self._reference_gray.shape != gray.shape:
            self._window = cv2.createHanningWindow(
                (gray.shape[1], gray.shape[0]), cv2.CV_32F
            )
        return gray, scale

    def mark_reference(self, frame: np.ndarray) -> None:
        """Save the current frame after a successful pitch refresh."""

        gray, _ = self._prepare(frame)
        self._reference_gray = gray

    def reset(self) -> None:
        self._reference_gray = None
        self._window = None

    def measure(self, frame: np.ndarray) -> Optional[CameraMotionEstimate]:
        """Measure drift from the last pitch-refresh reference frame."""

        gray, scale = self._prepare(frame)
        if self._reference_gray is None or self._reference_gray.shape != gray.shape:
            self._reference_gray = gray
            return None

        if self._window is None or self._window.shape != gray.shape:
            self._window = cv2.createHanningWindow(
                (gray.shape[1], gray.shape[0]), cv2.CV_32F
            )
        try:
            shift, response = cv2.phaseCorrelate(self._reference_gray, gray, self._window)
        except cv2.error:
            return CameraMotionEstimate(
                shift_x_px=0.0,
                shift_y_px=0.0,
                response=0.0,
                threshold_px=self.threshold_px,
                minimum_response=self.minimum_response,
            )

        # The reference and current frames have the same shape here.  Use the
        # current scale to convert the low-resolution phase shift to pixels.
        return CameraMotionEstimate(
            shift_x_px=float(shift[0] / max(scale, 1e-6)),
            shift_y_px=float(shift[1] / max(scale, 1e-6)),
            response=float(response),
            threshold_px=self.threshold_px,
            minimum_response=self.minimum_response,
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
        if self.calibration.lens_model == "fisheye":
            distortion = self.calibration.distortion_coefficients.reshape(4, 1)
            new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                camera_matrix,
                distortion,
                (width, height),
                np.eye(3),
                balance=self.alpha,
                new_size=(width, height),
            )
            map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
                camera_matrix,
                distortion,
                np.eye(3),
                new_camera_matrix,
                (width, height),
                cv2.CV_32FC1,
            )
        else:
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
