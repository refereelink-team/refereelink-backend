"""Constant-velocity one-dimensional EKF for horizontal camera pan."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_angle(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class PanFilterConfig:
    process_noise_pan_rad2_s: float = 2.5e-5
    process_noise_velocity_rad2_s3: float = 2.5e-3
    initial_pan_variance_rad2: float = 0.01
    initial_velocity_variance_rad2_s2: float = 0.1
    innovation_gate_sigma: float = 4.0


class PanExtendedKalmanFilter:
    def __init__(self, config: PanFilterConfig | None = None) -> None:
        self.config = config or PanFilterConfig()
        self.state = np.zeros(2, dtype=np.float64)
        self.covariance = np.diag(
            [
                self.config.initial_pan_variance_rad2,
                self.config.initial_velocity_variance_rad2_s2,
            ]
        )
        self.initialized = False

    @property
    def pan_rad(self) -> float:
        return float(self.state[0])

    @property
    def velocity_rad_s(self) -> float:
        return float(self.state[1])

    def reset(self, pan_rad: float = 0.0, velocity_rad_s: float = 0.0) -> None:
        self.state[:] = (wrap_angle(pan_rad), velocity_rad_s)
        self.covariance = np.diag(
            [
                self.config.initial_pan_variance_rad2,
                self.config.initial_velocity_variance_rad2_s2,
            ]
        )
        self.initialized = True

    def predict(self, dt_seconds: float) -> None:
        if dt_seconds <= 0 or not np.isfinite(dt_seconds):
            raise ValueError("dt_seconds must be finite and positive")
        if not self.initialized:
            self.reset()
        transition = np.array([[1.0, dt_seconds], [0.0, 1.0]], dtype=np.float64)
        process = np.diag(
            [
                self.config.process_noise_pan_rad2_s * dt_seconds,
                self.config.process_noise_velocity_rad2_s3 * dt_seconds,
            ]
        )
        self.state = transition @ self.state
        self.state[0] = wrap_angle(float(self.state[0]))
        self.covariance = transition @ self.covariance @ transition.T + process

    def update(self, measured_pan_rad: float, measurement_variance_rad2: float) -> bool:
        if measurement_variance_rad2 <= 0 or not np.isfinite(measurement_variance_rad2):
            raise ValueError("measurement variance must be finite and positive")
        if not self.initialized:
            self.reset(measured_pan_rad)
            self.covariance[0, 0] = measurement_variance_rad2
            return True
        innovation = wrap_angle(measured_pan_rad - self.state[0])
        innovation_variance = float(self.covariance[0, 0] + measurement_variance_rad2)
        normalized = abs(innovation) / np.sqrt(max(innovation_variance, 1e-12))
        if normalized > self.config.innovation_gate_sigma:
            return False
        gain = self.covariance[:, 0] / innovation_variance
        self.state += gain * innovation
        self.state[0] = wrap_angle(float(self.state[0]))
        identity_minus_gain = np.eye(2) - np.outer(gain, np.array([1.0, 0.0]))
        self.covariance = identity_minus_gain @ self.covariance
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        return True
