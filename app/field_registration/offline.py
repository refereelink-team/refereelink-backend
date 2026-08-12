"""Shot-safe bidirectional fusion and RTS smoothing for offline videos."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from app.field_registration.broadcast_camera import (
    BroadcastCameraEstimator,
    BroadcastCameraParameters,
    BroadcastCameraStateFilter,
)
from app.field_registration.pan_filter import wrap_angle
from app.field_registration.types import (
    CameraState,
    CameraTrackingStatus,
    MeasurementTier,
)


@dataclass(frozen=True)
class OfflineCameraObservation:
    frame_index: int
    shot_id: int
    image_size: tuple[int, int]
    state: CameraState

    def __post_init__(self) -> None:
        if self.frame_index < 0 or self.shot_id < 0:
            raise ValueError("frame_index and shot_id must be non-negative")
        if min(self.image_size) <= 0:
            raise ValueError("image_size must be positive")


@dataclass(frozen=True)
class OfflineSmoothingConfig:
    fps: float = 25.0
    process_angle_variance_s: float = 2.5e-5
    process_log_focal_variance_s: float = 1e-4
    process_velocity_variance_s: float = 2.5e-3
    minimum_observation_variance: float = 1e-8


@dataclass(frozen=True)
class OfflineSmoothingResult:
    states: tuple[CameraState, ...]
    frame_indices: tuple[int, ...]
    smoothed_frame_count: int
    preview_only_frame_count: int
    shot_count: int


class BidirectionalCameraSmoother:
    """Fuse forward/backward camera estimates without crossing hard cuts."""

    def __init__(
        self,
        estimator: BroadcastCameraEstimator | None = None,
        config: OfflineSmoothingConfig | None = None,
    ) -> None:
        self.estimator = estimator or BroadcastCameraEstimator()
        self.config = config or OfflineSmoothingConfig()
        if self.config.fps <= 0.0:
            raise ValueError("offline smoothing fps must be positive")

    @staticmethod
    def _physical(state: CameraState) -> bool:
        return (
            state.pitch_to_image is not None
            and state.camera_center_xyz_m is not None
            and state.focal_px is not None
            and np.isfinite(state.pan_rad)
            and np.isfinite(state.tilt_rad)
            and np.isfinite(state.roll_rad)
        )

    def _shot_center(
        self,
        forward: Sequence[OfflineCameraObservation],
        backward: Sequence[OfflineCameraObservation],
    ) -> Optional[np.ndarray]:
        for collection in (forward, backward):
            preferred = [
                item
                for item in collection
                if self._physical(item.state)
                and item.state.measurement_tier is MeasurementTier.SAFE
                and item.state.status is not CameraTrackingStatus.PREDICTED
            ]
            candidates = preferred or [
                item for item in collection if self._physical(item.state)
            ]
            if candidates:
                # The online broadcast core freezes this value per Shot. The
                # median protects imported caches that contain small numeric
                # differences without blending camera centers across Shots.
                return np.median(
                    np.stack(
                        [item.state.camera_center_xyz_m for item in candidates]
                    ),
                    axis=0,
                )
        return None

    def _estimate(
        self,
        item: OfflineCameraObservation,
        center: np.ndarray,
        *,
        reverse_velocity: bool,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        state = item.state
        if state.pitch_to_image is None:
            return None
        try:
            initial = (
                BroadcastCameraParameters(
                    center,
                    state.pan_rad,
                    state.tilt_rad,
                    state.roll_rad,
                    float(np.log(state.focal_px)),
                    item.image_size,
                )
                if self._physical(state)
                else None
            )
            canonical = self.estimator.fit_with_fixed_center(
                state.pitch_to_image,
                item.image_size,
                center,
                initial,
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return None
        if (
            canonical.fit_p95_error_px
            > self.estimator.config.maximum_physical_fit_p95_px
        ):
            return None
        vector = np.asarray(
            [
                canonical.pan_rad,
                canonical.tilt_rad,
                canonical.roll_rad,
                canonical.log_focal_px,
                state.pan_velocity_rad_s,
                state.tilt_velocity_rad_s,
                state.zoom_velocity_log_s,
            ],
            dtype=np.float64,
        )
        covariance = (
            state.camera_parameter_covariance.copy()
            if state.camera_parameter_covariance is not None
            else np.diag(
                [
                    2.5e-3,
                    2.5e-3,
                    2.5e-3,
                    1e-2,
                    0.1,
                    0.1,
                    0.1,
                ]
            )
        )
        covariance = (covariance + covariance.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = eigenvectors @ np.diag(
            np.maximum(eigenvalues, self.config.minimum_observation_variance)
        ) @ eigenvectors.T
        if reverse_velocity:
            direction = np.diag([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
            vector = direction @ vector
            covariance = direction @ covariance @ direction
        return vector, covariance

    @staticmethod
    def _align_angles(vector: np.ndarray, reference: np.ndarray) -> np.ndarray:
        aligned = vector.copy()
        for index in (0, 2):
            delta = wrap_angle(float(aligned[index] - reference[index]))
            aligned[index] = reference[index] + delta
        return aligned

    def _fuse(
        self,
        estimates: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if not estimates:
            return None
        reference = estimates[0][0]
        information = np.zeros((7, 7), dtype=np.float64)
        weighted = np.zeros(7, dtype=np.float64)
        for vector, covariance in estimates:
            aligned = self._align_angles(vector, reference)
            try:
                inverse = np.linalg.inv(covariance)
            except np.linalg.LinAlgError:
                continue
            information += inverse
            weighted += inverse @ aligned
        if np.linalg.matrix_rank(information) < 7:
            return None
        covariance = np.linalg.inv(information)
        vector = covariance @ weighted
        vector[0] = wrap_angle(float(vector[0]))
        vector[2] = wrap_angle(float(vector[2]))
        return vector, (covariance + covariance.T) / 2.0

    def _process_noise(self, dt_seconds: float) -> np.ndarray:
        return np.diag(
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

    def _rts(
        self,
        measurements: Sequence[Optional[tuple[np.ndarray, np.ndarray]]],
    ) -> list[Optional[tuple[np.ndarray, np.ndarray]]]:
        count = len(measurements)
        output: list[Optional[tuple[np.ndarray, np.ndarray]]] = [None] * count
        first = next((index for index, value in enumerate(measurements) if value), None)
        if first is None:
            return output
        dt = 1.0 / self.config.fps
        transition = BroadcastCameraStateFilter.transition(dt)
        process = self._process_noise(dt)
        filtered_state: list[Optional[np.ndarray]] = [None] * count
        filtered_covariance: list[Optional[np.ndarray]] = [None] * count
        predicted_state: list[Optional[np.ndarray]] = [None] * count
        predicted_covariance: list[Optional[np.ndarray]] = [None] * count
        state, covariance = measurements[first]
        filtered_state[first] = state.copy()
        filtered_covariance[first] = covariance.copy()
        identity = np.eye(7, dtype=np.float64)
        for index in range(first + 1, count):
            prediction = transition @ state
            prediction[0] = wrap_angle(float(prediction[0]))
            prediction[2] = wrap_angle(float(prediction[2]))
            prediction_covariance = transition @ covariance @ transition.T + process
            predicted_state[index] = prediction.copy()
            predicted_covariance[index] = prediction_covariance.copy()
            measurement = measurements[index]
            if measurement is None:
                state, covariance = prediction, prediction_covariance
            else:
                observed, noise = measurement
                innovation = observed - prediction
                innovation[0] = wrap_angle(float(innovation[0]))
                innovation[2] = wrap_angle(float(innovation[2]))
                innovation_covariance = prediction_covariance + noise
                gain = prediction_covariance @ np.linalg.inv(innovation_covariance)
                state = prediction + gain @ innovation
                state[0] = wrap_angle(float(state[0]))
                state[2] = wrap_angle(float(state[2]))
                covariance = (identity - gain) @ prediction_covariance @ (
                    identity - gain
                ).T + gain @ noise @ gain.T
                covariance = (covariance + covariance.T) / 2.0
            filtered_state[index] = state.copy()
            filtered_covariance[index] = covariance.copy()

        smooth_state = filtered_state.copy()
        smooth_covariance = filtered_covariance.copy()
        for index in range(count - 2, first - 1, -1):
            current = filtered_state[index]
            current_covariance = filtered_covariance[index]
            following = smooth_state[index + 1]
            following_covariance = smooth_covariance[index + 1]
            prediction = predicted_state[index + 1]
            prediction_covariance = predicted_covariance[index + 1]
            if any(
                value is None
                for value in (
                    current,
                    current_covariance,
                    following,
                    following_covariance,
                    prediction,
                    prediction_covariance,
                )
            ):
                continue
            gain = current_covariance @ transition.T @ np.linalg.inv(
                prediction_covariance
            )
            difference = following - prediction
            difference[0] = wrap_angle(float(difference[0]))
            difference[2] = wrap_angle(float(difference[2]))
            state = current + gain @ difference
            state[0] = wrap_angle(float(state[0]))
            state[2] = wrap_angle(float(state[2]))
            covariance = current_covariance + gain @ (
                following_covariance - prediction_covariance
            ) @ gain.T
            smooth_state[index] = state
            smooth_covariance[index] = (covariance + covariance.T) / 2.0
        for index in range(first, count):
            if smooth_state[index] is not None and smooth_covariance[index] is not None:
                output[index] = smooth_state[index], smooth_covariance[index]
        return output

    def smooth(
        self,
        forward: Sequence[OfflineCameraObservation],
        backward: Sequence[OfflineCameraObservation],
    ) -> OfflineSmoothingResult:
        if not forward:
            return OfflineSmoothingResult((), (), 0, 0, 0)
        ordered = sorted(forward, key=lambda item: item.frame_index)
        indices = [item.frame_index for item in ordered]
        if len(indices) != len(set(indices)):
            raise ValueError("forward pass contains duplicate frame indices")
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError("forward frame indices must be contiguous")
        backward_by_index = {item.frame_index: item for item in backward}
        if len(backward_by_index) != len(backward):
            raise ValueError("backward pass contains duplicate frame indices")

        output_states = [item.state for item in ordered]
        smoothed_count = 0
        preview_count = 0
        shot_ids: list[int] = []
        for item in ordered:
            if not shot_ids or item.shot_id != shot_ids[-1]:
                shot_ids.append(item.shot_id)
        for shot_id in shot_ids:
            positions = [
                index for index, item in enumerate(ordered) if item.shot_id == shot_id
            ]
            shot_forward = [ordered[index] for index in positions]
            shot_backward = [
                backward_by_index[item.frame_index]
                for item in shot_forward
                if item.frame_index in backward_by_index
            ]
            center = self._shot_center(shot_forward, shot_backward)
            if center is None:
                continue
            measurements: list[Optional[tuple[np.ndarray, np.ndarray]]] = []
            for item in shot_forward:
                estimates: list[tuple[np.ndarray, np.ndarray]] = []
                forward_estimate = self._estimate(
                    item, center, reverse_velocity=False
                )
                if forward_estimate is not None:
                    estimates.append(forward_estimate)
                reverse = backward_by_index.get(item.frame_index)
                if reverse is not None and reverse.shot_id == shot_id:
                    reverse_estimate = self._estimate(
                        reverse, center, reverse_velocity=True
                    )
                    if reverse_estimate is not None:
                        estimates.append(reverse_estimate)
                measurements.append(self._fuse(estimates))
            smoothed = self._rts(measurements)
            for local_index, global_index in enumerate(positions):
                value = smoothed[local_index]
                if value is None:
                    continue
                vector, covariance = value
                item = ordered[global_index]
                forward_state = item.state
                reverse = backward_by_index.get(item.frame_index)
                if reverse is not None and reverse.shot_id != shot_id:
                    reverse = None
                sources = [forward_state] + ([reverse.state] if reverse else [])
                direct_safe = any(
                    state.measurement_tier is MeasurementTier.SAFE
                    and state.status
                    in {
                        CameraTrackingStatus.RELOCALIZED,
                        CameraTrackingStatus.CORRECTED,
                        CameraTrackingStatus.TRACKED,
                    }
                    for state in sources
                )
                status = (
                    CameraTrackingStatus.CORRECTED
                    if direct_safe
                    else CameraTrackingStatus.PREDICTED
                )
                tier = MeasurementTier.SAFE if direct_safe else MeasurementTier.PREVIEW
                try:
                    parameters = BroadcastCameraParameters(
                        center,
                        wrap_angle(float(vector[0])),
                        float(vector[1]),
                        wrap_angle(float(vector[2])),
                        float(vector[3]),
                        item.image_size,
                    )
                    pitch_to_image = parameters.pitch_to_image_homography()
                    image_to_pitch = parameters.image_to_pitch_homography()
                except (ValueError, np.linalg.LinAlgError):
                    continue
                confidence = max(state.confidence for state in sources)
                output_states[global_index] = replace(
                    forward_state,
                    status=status,
                    pan_rad=parameters.pan_rad,
                    pan_velocity_rad_s=float(vector[4]),
                    covariance=covariance[np.ix_([0, 4], [0, 4])],
                    image_to_pitch=image_to_pitch,
                    pitch_to_image=pitch_to_image,
                    confidence=confidence,
                    registration_mode=forward_state.registration_mode,
                    measurement_tier=tier,
                    shot_id=shot_id,
                    camera_model="broadcast_tripod_pan_tilt_zoom_offline_rts",
                    focal_px=parameters.focal_px,
                    tilt_rad=parameters.tilt_rad,
                    roll_rad=parameters.roll_rad,
                    projection_uncertainty=float(np.trace(covariance[:4, :4])),
                    camera_center_xyz_m=center,
                    tilt_velocity_rad_s=float(vector[5]),
                    zoom_velocity_log_s=float(vector[6]),
                    camera_parameter_covariance=covariance,
                )
                smoothed_count += 1
        preview_count = sum(
            state.measurement_tier is MeasurementTier.PREVIEW
            for state in output_states
        )
        return OfflineSmoothingResult(
            states=tuple(output_states),
            frame_indices=tuple(indices),
            smoothed_frame_count=smoothed_count,
            preview_only_frame_count=preview_count,
            shot_count=len(shot_ids),
        )


def save_offline_smoothing_result(
    result: OfflineSmoothingResult,
    path: str | Path,
) -> tuple[Path, Path]:
    """Persist numeric matrices in NPZ and audited metadata in JSON."""

    target = Path(path)
    base = target.with_suffix("") if target.suffix in {".json", ".npz"} else target
    json_path = base.with_suffix(".json")
    npz_path = base.with_suffix(".npz")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    states = result.states
    matrices = np.full((len(states), 3, 3), np.nan, dtype=np.float64)
    parameter_vectors = np.full((len(states), 7), np.nan, dtype=np.float64)
    parameter_covariances = np.full((len(states), 7, 7), np.nan, dtype=np.float64)
    camera_centers = np.full((len(states), 3), np.nan, dtype=np.float64)
    confidence = np.zeros(len(states), dtype=np.float64)
    shot_ids = np.zeros(len(states), dtype=np.int32)
    for index, state in enumerate(states):
        if state.pitch_to_image is not None:
            matrices[index] = state.pitch_to_image
        if (
            state.focal_px is not None
            and np.isfinite(state.pan_rad)
            and np.isfinite(state.tilt_rad)
            and np.isfinite(state.roll_rad)
        ):
            parameter_vectors[index] = (
                state.pan_rad,
                state.tilt_rad,
                state.roll_rad,
                np.log(state.focal_px),
                state.pan_velocity_rad_s,
                state.tilt_velocity_rad_s,
                state.zoom_velocity_log_s,
            )
        if state.camera_parameter_covariance is not None:
            parameter_covariances[index] = state.camera_parameter_covariance
        if state.camera_center_xyz_m is not None:
            camera_centers[index] = state.camera_center_xyz_m
        confidence[index] = state.confidence
        shot_ids[index] = state.shot_id
    np.savez_compressed(
        npz_path,
        frame_indices=np.asarray(result.frame_indices, dtype=np.int64),
        shot_ids=shot_ids,
        pitch_to_image=matrices,
        camera_parameters=parameter_vectors,
        camera_parameter_covariances=parameter_covariances,
        camera_centers_xyz_m=camera_centers,
        confidence=confidence,
    )
    digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    payload = {
        "format_version": 1,
        "arrays_file": npz_path.name,
        "arrays_sha256": digest,
        "frame_count": len(states),
        "smoothed_frame_count": result.smoothed_frame_count,
        "preview_only_frame_count": result.preview_only_frame_count,
        "shot_count": result.shot_count,
        "statuses": [state.status.value for state in states],
        "measurement_tiers": [state.measurement_tier.value for state in states],
        "camera_models": [state.camera_model for state in states],
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return json_path, npz_path


def load_offline_smoothing_result(
    path: str | Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Load a persisted result without enabling NumPy pickle objects."""

    target = Path(path)
    json_path = target.with_suffix(".json") if target.suffix != ".json" else target
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if int(payload.get("format_version", 0)) != 1:
        raise ValueError("unsupported offline registration format")
    npz_path = json_path.parent / str(payload["arrays_file"])
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    if digest != str(payload.get("arrays_sha256", "")):
        raise ValueError("offline registration array SHA-256 mismatch")
    with np.load(npz_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    if arrays["frame_indices"].shape != (int(payload["frame_count"]),):
        raise ValueError("offline registration frame count does not match arrays")
    return payload, arrays
