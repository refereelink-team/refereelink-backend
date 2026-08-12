"""Physical broadcast-camera decomposition and shot-local state filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.field_registration.geometry import transform_points, validate_homography
from app.field_registration.pan_filter import wrap_angle
from app.field_registration.pitch_model import PitchModel


def _intrinsic_matrix(focal_px: float, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    return np.asarray(
        [[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotation_from_pan_tilt_roll(
    pan_rad: float,
    tilt_rad: float,
    roll_rad: float,
) -> np.ndarray:
    """Build world-to-camera rotation using z-up pitch coordinates."""

    cos_tilt = np.cos(tilt_rad)
    forward = np.asarray(
        [
            cos_tilt * np.cos(pan_rad),
            cos_tilt * np.sin(pan_rad),
            -np.sin(tilt_rad),
        ],
        dtype=np.float64,
    )
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    norm = float(np.linalg.norm(right))
    if norm < 1e-8:
        raise ValueError("camera optical axis cannot be parallel to world up")
    right /= norm
    down = np.cross(forward, right)
    rolled_right = np.cos(roll_rad) * right + np.sin(roll_rad) * down
    rolled_down = -np.sin(roll_rad) * right + np.cos(roll_rad) * down
    return np.vstack((rolled_right, rolled_down, forward))


def pan_tilt_roll_from_rotation(rotation: np.ndarray) -> tuple[float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    forward = matrix[2]
    pan = float(np.arctan2(forward[1], forward[0]))
    tilt = float(np.arctan2(-forward[2], np.hypot(forward[0], forward[1])))
    base = rotation_from_pan_tilt_roll(pan, tilt, 0.0)
    roll = float(
        np.arctan2(
            np.dot(matrix[0], base[1]),
            np.dot(matrix[0], base[0]),
        )
    )
    return wrap_angle(pan), tilt, wrap_angle(roll)


@dataclass(frozen=True)
class BroadcastCameraParameters:
    camera_center_xyz_m: np.ndarray
    pan_rad: float
    tilt_rad: float
    roll_rad: float
    log_focal_px: float
    image_size: tuple[int, int]
    fit_median_error_px: float = 0.0
    fit_p95_error_px: float = 0.0

    def __post_init__(self) -> None:
        center = np.asarray(self.camera_center_xyz_m, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("camera center must be a finite xyz vector")
        if center[2] <= 0.0:
            raise ValueError("broadcast camera must be above the pitch plane")
        if min(self.image_size) <= 0:
            raise ValueError("image_size must be positive")
        values = (
            self.pan_rad,
            self.tilt_rad,
            self.roll_rad,
            self.log_focal_px,
            self.fit_median_error_px,
            self.fit_p95_error_px,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("broadcast camera parameters must be finite")
        if self.focal_px <= 0.0:
            raise ValueError("focal length must be positive")
        object.__setattr__(self, "camera_center_xyz_m", center)

    @property
    def focal_px(self) -> float:
        return float(np.exp(self.log_focal_px))

    @property
    def observation_vector(self) -> np.ndarray:
        return np.asarray(
            [self.pan_rad, self.tilt_rad, self.roll_rad, self.log_focal_px],
            dtype=np.float64,
        )

    def pitch_to_image_homography(self) -> np.ndarray:
        rotation = rotation_from_pan_tilt_roll(
            self.pan_rad, self.tilt_rad, self.roll_rad
        )
        translation = -rotation @ self.camera_center_xyz_m
        plane = np.column_stack((rotation[:, 0], rotation[:, 1], translation))
        matrix = _intrinsic_matrix(self.focal_px, self.image_size) @ plane
        return validate_homography(matrix)

    def image_to_pitch_homography(self) -> np.ndarray:
        inverse = np.linalg.inv(self.pitch_to_image_homography())
        return validate_homography(inverse)


@dataclass(frozen=True)
class BroadcastCameraEstimatorConfig:
    minimum_focal_width_ratio: float = 0.35
    maximum_focal_width_ratio: float = 6.0
    focal_search_steps: int = 160
    fixed_center_iterations: int = 18
    huber_delta_px: float = 6.0
    maximum_physical_fit_p95_px: float = 12.0


class BroadcastCameraEstimator:
    """Recover a practical tripod-camera state from a plane homography."""

    def __init__(
        self,
        pitch_model: PitchModel | None = None,
        config: BroadcastCameraEstimatorConfig | None = None,
    ) -> None:
        self.pitch_model = pitch_model or PitchModel()
        self.config = config or BroadcastCameraEstimatorConfig()

    @staticmethod
    def _orthogonality_objective(
        pitch_to_image: np.ndarray,
        focal_px: float,
        image_size: tuple[int, int],
    ) -> float:
        normalized = np.linalg.inv(_intrinsic_matrix(focal_px, image_size)) @ pitch_to_image
        first = normalized[:, 0]
        second = normalized[:, 1]
        first_norm = float(np.linalg.norm(first))
        second_norm = float(np.linalg.norm(second))
        denominator = max(first_norm * second_norm, 1e-12)
        orthogonal = float(np.dot(first, second) / denominator)
        equal_norm = (first_norm - second_norm) / max(first_norm + second_norm, 1e-12)
        return orthogonal**2 + equal_norm**2

    def _estimate_focal(
        self,
        pitch_to_image: np.ndarray,
        image_size: tuple[int, int],
    ) -> float:
        width = float(image_size[0])
        lower = np.log(width * self.config.minimum_focal_width_ratio)
        upper = np.log(width * self.config.maximum_focal_width_ratio)
        candidates = np.linspace(lower, upper, self.config.focal_search_steps)
        values = np.asarray(
            [
                self._orthogonality_objective(
                    pitch_to_image, float(np.exp(value)), image_size
                )
                for value in candidates
            ]
        )
        best = int(np.argmin(values))
        left = candidates[max(best - 1, 0)]
        right = candidates[min(best + 1, len(candidates) - 1)]
        golden = (np.sqrt(5.0) - 1.0) / 2.0
        for _ in range(36):
            first = right - golden * (right - left)
            second = left + golden * (right - left)
            first_value = self._orthogonality_objective(
                pitch_to_image, float(np.exp(first)), image_size
            )
            second_value = self._orthogonality_objective(
                pitch_to_image, float(np.exp(second)), image_size
            )
            if first_value <= second_value:
                right = second
            else:
                left = first
        return float(np.exp((left + right) / 2.0))

    def decompose(
        self,
        pitch_to_image: np.ndarray,
        image_size: tuple[int, int],
    ) -> BroadcastCameraParameters:
        homography = validate_homography(pitch_to_image)
        focal = self._estimate_focal(homography, image_size)
        normalized = np.linalg.inv(_intrinsic_matrix(focal, image_size)) @ homography
        scale = 2.0 / max(
            float(np.linalg.norm(normalized[:, 0]) + np.linalg.norm(normalized[:, 1])),
            1e-12,
        )
        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        for sign in (1.0, -1.0):
            first = sign * scale * normalized[:, 0]
            second = sign * scale * normalized[:, 1]
            third = np.cross(first, second)
            approximate = np.column_stack((first, second, third))
            left, _, right = np.linalg.svd(approximate)
            rotation = left @ right
            if np.linalg.det(rotation) < 0.0:
                left[:, -1] *= -1.0
                rotation = left @ right
            translation = sign * scale * normalized[:, 2]
            center = -rotation.T @ translation
            positive_depth = float(center[2])
            candidates.append((positive_depth, rotation, center))
        positive = [item for item in candidates if item[0] > 0.0]
        if not positive:
            raise ValueError("homography does not yield an above-pitch camera")
        _, rotation, center = max(positive, key=lambda item: item[0])
        pan, tilt, roll = pan_tilt_roll_from_rotation(rotation)
        initial = BroadcastCameraParameters(
            center,
            pan,
            tilt,
            roll,
            float(np.log(focal)),
            image_size,
        )
        return self._with_fit_error(initial, homography)

    def _visible_grid(
        self,
        target_homography: np.ndarray,
        image_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        pitch = self.pitch_model.grid(18, 12)
        image = transform_points(pitch, target_homography)
        width, height = image_size
        visible = (
            np.all(np.isfinite(image), axis=1)
            & (image[:, 0] >= -0.2 * width)
            & (image[:, 0] <= 1.2 * width)
            & (image[:, 1] >= -0.2 * height)
            & (image[:, 1] <= 1.2 * height)
        )
        return pitch[visible], image[visible]

    def _with_fit_error(
        self,
        parameters: BroadcastCameraParameters,
        target_homography: np.ndarray,
    ) -> BroadcastCameraParameters:
        pitch, target = self._visible_grid(target_homography, parameters.image_size)
        if len(pitch) < 4:
            raise ValueError("homography has insufficient visible pitch support")
        predicted = transform_points(pitch, parameters.pitch_to_image_homography())
        errors = np.linalg.norm(predicted - target, axis=1)
        return BroadcastCameraParameters(
            parameters.camera_center_xyz_m,
            parameters.pan_rad,
            parameters.tilt_rad,
            parameters.roll_rad,
            parameters.log_focal_px,
            parameters.image_size,
            float(np.median(errors)),
            float(np.percentile(errors, 95)),
        )

    def with_fit_error(
        self,
        parameters: BroadcastCameraParameters,
        target_homography: np.ndarray,
    ) -> BroadcastCameraParameters:
        """Attach grid reprojection diagnostics against a projective target."""

        return self._with_fit_error(parameters, validate_homography(target_homography))

    def fit_with_fixed_center(
        self,
        pitch_to_image: np.ndarray,
        image_size: tuple[int, int],
        camera_center_xyz_m: np.ndarray,
        initial: Optional[BroadcastCameraParameters] = None,
    ) -> BroadcastCameraParameters:
        target_h = validate_homography(pitch_to_image)
        decomposed = self.decompose(target_h, image_size)
        vector = (
            decomposed.observation_vector.copy()
            if initial is None
            else initial.observation_vector.copy()
        )
        pitch, target = self._visible_grid(target_h, image_size)
        if len(pitch) < 8:
            raise ValueError("fixed-center fit needs at least eight visible grid points")

        def residual(values: np.ndarray) -> np.ndarray:
            candidate = BroadcastCameraParameters(
                camera_center_xyz_m,
                wrap_angle(float(values[0])),
                float(values[1]),
                wrap_angle(float(values[2])),
                float(values[3]),
                image_size,
            )
            projected = transform_points(pitch, candidate.pitch_to_image_homography())
            return (projected - target).reshape(-1)

        damping = 1e-3
        steps = np.asarray([1e-5, 1e-5, 1e-5, 1e-4], dtype=np.float64)
        for _ in range(self.config.fixed_center_iterations):
            current = residual(vector)
            point_error = np.linalg.norm(current.reshape(-1, 2), axis=1)
            weights = np.minimum(
                1.0,
                self.config.huber_delta_px / np.maximum(point_error, 1e-6),
            )
            expanded = np.repeat(np.sqrt(weights), 2)
            jacobian = np.empty((current.size, 4), dtype=np.float64)
            for column, step in enumerate(steps):
                shifted = vector.copy()
                shifted[column] += step
                jacobian[:, column] = (residual(shifted) - current) / step
            weighted_jacobian = jacobian * expanded[:, None]
            weighted_residual = current * expanded
            normal = weighted_jacobian.T @ weighted_jacobian + damping * np.eye(4)
            gradient = weighted_jacobian.T @ weighted_residual
            try:
                delta = -np.linalg.solve(normal, gradient)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(delta)):
                break
            trial = vector + delta
            trial[0] = wrap_angle(float(trial[0]))
            trial[2] = wrap_angle(float(trial[2]))
            if np.mean(residual(trial) ** 2) < np.mean(current**2):
                vector = trial
                damping = max(damping * 0.5, 1e-8)
                if float(np.linalg.norm(delta)) < 1e-7:
                    break
            else:
                damping = min(damping * 10.0, 1e8)
        fitted = BroadcastCameraParameters(
            camera_center_xyz_m,
            wrap_angle(float(vector[0])),
            float(vector[1]),
            wrap_angle(float(vector[2])),
            float(vector[3]),
            image_size,
        )
        return self._with_fit_error(fitted, target_h)


@dataclass(frozen=True)
class BroadcastCameraFilterConfig:
    process_angle_variance_s: float = 2.5e-5
    process_log_focal_variance_s: float = 1e-4
    process_velocity_variance_s: float = 2.5e-3
    initial_angle_variance: float = 2.5e-3
    initial_log_focal_variance: float = 1e-2
    initial_velocity_variance: float = 0.1
    innovation_gate_sigma: float = 6.0


class BroadcastCameraStateFilter:
    """Seven-state EKF for pan, tilt, roll, log-focal and velocities."""

    def __init__(self, config: BroadcastCameraFilterConfig | None = None) -> None:
        self.config = config or BroadcastCameraFilterConfig()
        self.state = np.zeros(7, dtype=np.float64)
        self.covariance = np.eye(7, dtype=np.float64)
        self.initialized = False

    def reset(self, measurement: np.ndarray) -> None:
        values = np.asarray(measurement, dtype=np.float64)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("broadcast camera measurement must be four finite values")
        self.state[:] = (values[0], values[1], values[2], values[3], 0.0, 0.0, 0.0)
        self.state[0] = wrap_angle(float(self.state[0]))
        self.state[2] = wrap_angle(float(self.state[2]))
        self.covariance = np.diag(
            [
                self.config.initial_angle_variance,
                self.config.initial_angle_variance,
                self.config.initial_angle_variance,
                self.config.initial_log_focal_variance,
                self.config.initial_velocity_variance,
                self.config.initial_velocity_variance,
                self.config.initial_velocity_variance,
            ]
        )
        self.initialized = True

    @staticmethod
    def transition(dt_seconds: float) -> np.ndarray:
        matrix = np.eye(7, dtype=np.float64)
        matrix[0, 4] = dt_seconds
        matrix[1, 5] = dt_seconds
        matrix[3, 6] = dt_seconds
        return matrix

    def predict(self, dt_seconds: float) -> None:
        if dt_seconds <= 0.0 or not np.isfinite(dt_seconds):
            raise ValueError("dt_seconds must be finite and positive")
        if not self.initialized:
            return
        transition = self.transition(dt_seconds)
        process = np.diag(
            [
                self.config.process_angle_variance_s * dt_seconds,
                self.config.process_angle_variance_s * dt_seconds,
                self.config.process_angle_variance_s * dt_seconds,
                self.config.process_log_focal_variance_s * dt_seconds,
                self.config.process_velocity_variance_s * dt_seconds,
                self.config.process_velocity_variance_s * dt_seconds,
                self.config.process_velocity_variance_s * dt_seconds,
            ]
        )
        self.state = transition @ self.state
        self.state[0] = wrap_angle(float(self.state[0]))
        self.state[2] = wrap_angle(float(self.state[2]))
        self.covariance = transition @ self.covariance @ transition.T + process

    def update(self, measurement: np.ndarray, variances: np.ndarray) -> bool:
        values = np.asarray(measurement, dtype=np.float64)
        variance = np.asarray(variances, dtype=np.float64)
        if values.shape != (4,) or variance.shape != (4,):
            raise ValueError("measurement and variances must have shape (4,)")
        if not np.all(np.isfinite(values)) or np.any(variance <= 0.0):
            raise ValueError("measurement must be finite and variances positive")
        if not self.initialized:
            self.reset(values)
            self.covariance[:4, :4] = np.diag(variance)
            return True
        observation = np.zeros((4, 7), dtype=np.float64)
        observation[0, 0] = observation[1, 1] = 1.0
        observation[2, 2] = observation[3, 3] = 1.0
        innovation = values - observation @ self.state
        innovation[0] = wrap_angle(float(innovation[0]))
        innovation[2] = wrap_angle(float(innovation[2]))
        noise = np.diag(variance)
        innovation_covariance = observation @ self.covariance @ observation.T + noise
        try:
            normalized = float(
                np.sqrt(innovation @ np.linalg.solve(innovation_covariance, innovation))
            )
        except np.linalg.LinAlgError:
            return False
        if normalized > self.config.innovation_gate_sigma:
            return False
        gain = self.covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        self.state += gain @ innovation
        self.state[0] = wrap_angle(float(self.state[0]))
        self.state[2] = wrap_angle(float(self.state[2]))
        identity = np.eye(7) - gain @ observation
        self.covariance = identity @ self.covariance @ identity.T + gain @ noise @ gain.T
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        return True

    def parameters(
        self,
        camera_center_xyz_m: np.ndarray,
        image_size: tuple[int, int],
        *,
        fit_median_error_px: float = 0.0,
        fit_p95_error_px: float = 0.0,
    ) -> BroadcastCameraParameters:
        if not self.initialized:
            raise ValueError("broadcast camera filter is not initialized")
        return BroadcastCameraParameters(
            camera_center_xyz_m,
            wrap_angle(float(self.state[0])),
            float(self.state[1]),
            wrap_angle(float(self.state[2])),
            float(self.state[3]),
            image_size,
            fit_median_error_px,
            fit_p95_error_px,
        )
