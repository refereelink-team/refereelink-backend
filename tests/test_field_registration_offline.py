from __future__ import annotations

import numpy as np
import pytest

from app.field_registration.broadcast_camera import BroadcastCameraParameters
from app.field_registration.offline import (
    BidirectionalCameraSmoother,
    OfflineCameraObservation,
    OfflineSmoothingConfig,
    load_offline_smoothing_result,
    save_offline_smoothing_result,
)
from app.field_registration.types import (
    CameraState,
    CameraTrackingStatus,
    MeasurementTier,
    RegistrationMode,
)


IMAGE_SIZE = (1280, 720)


def _state(
    center: np.ndarray,
    pan: float,
    *,
    status: CameraTrackingStatus = CameraTrackingStatus.TRACKED,
    tier: MeasurementTier = MeasurementTier.SAFE,
    velocity: float = 0.01,
    confidence: float = 0.9,
) -> CameraState:
    parameters = BroadcastCameraParameters(
        center,
        pan,
        0.31,
        -0.02,
        float(np.log(1120.0)),
        IMAGE_SIZE,
    )
    covariance = np.diag([2e-4, 2e-4, 2e-4, 6e-4, 1e-2, 1e-2, 1e-2])
    return CameraState(
        status=status,
        pan_rad=pan,
        pan_velocity_rad_s=velocity,
        covariance=covariance[np.ix_([0, 4], [0, 4])],
        image_to_pitch=parameters.image_to_pitch_homography(),
        pitch_to_image=parameters.pitch_to_image_homography(),
        confidence=confidence,
        registration_mode=RegistrationMode.BROADCAST,
        measurement_tier=tier,
        camera_model="broadcast_tripod_pan_tilt_zoom",
        focal_px=parameters.focal_px,
        tilt_rad=parameters.tilt_rad,
        roll_rad=parameters.roll_rad,
        camera_center_xyz_m=center,
        tilt_velocity_rad_s=0.0,
        zoom_velocity_log_s=0.0,
        camera_parameter_covariance=covariance,
    )


def _missing(shot_id: int) -> CameraState:
    return CameraState(
        status=CameraTrackingStatus.LOST,
        pan_rad=float("nan"),
        pan_velocity_rad_s=0.0,
        covariance=np.diag([1e6, 1e6]),
        image_to_pitch=None,
        pitch_to_image=None,
        confidence=0.0,
        registration_mode=RegistrationMode.BROADCAST,
        measurement_tier=MeasurementTier.UNAVAILABLE,
        shot_id=shot_id,
    )


def _observation(index: int, shot_id: int, state: CameraState) -> OfflineCameraObservation:
    if state.shot_id != shot_id:
        state = CameraState(
            **{
                **state.__dict__,
                "shot_id": shot_id,
            }
        )
    return OfflineCameraObservation(index, shot_id, IMAGE_SIZE, state)


def test_bidirectional_rts_reduces_pan_jitter_and_fills_short_gap() -> None:
    center = np.asarray([-18.0, 30.0, 24.0])
    rng = np.random.default_rng(17)
    true_pan = np.linspace(-0.08, 0.12, 30)
    forward = []
    backward = []
    for index, pan in enumerate(true_pan):
        if index in {12, 13, 14}:
            forward.append(_observation(index, 0, _missing(0)))
        else:
            forward.append(
                _observation(index, 0, _state(center, pan + rng.normal(0.0, 0.012)))
            )
        if index in {12, 13, 14, 16, 17}:
            backward.append(_observation(index, 0, _missing(0)))
        else:
            backward.append(
                _observation(
                    index,
                    0,
                    _state(
                        center,
                        pan + rng.normal(0.0, 0.012),
                        velocity=-0.01,
                    ),
                )
            )
    smoother = BidirectionalCameraSmoother(
        config=OfflineSmoothingConfig(fps=25.0)
    )

    result = smoother.smooth(forward, backward)

    observed = np.asarray(
        [item.state.pan_rad for item in forward if np.isfinite(item.state.pan_rad)]
    )
    observed_truth = np.asarray(
        [pan for item, pan in zip(forward, true_pan) if np.isfinite(item.state.pan_rad)]
    )
    smoothed = np.asarray([state.pan_rad for state in result.states])
    assert np.sqrt(np.mean((smoothed - true_pan) ** 2)) < np.sqrt(
        np.mean((observed - observed_truth) ** 2)
    )
    assert all(result.states[index].pitch_to_image is not None for index in (12, 13, 14))
    assert all(
        result.states[index].measurement_tier is MeasurementTier.PREVIEW
        for index in (12, 13, 14)
    )
    assert result.smoothed_frame_count == len(true_pan)
    assert result.shot_count == 1


def test_offline_smoother_never_fuses_across_shot_boundary() -> None:
    first_center = np.asarray([-18.0, 30.0, 24.0])
    second_center = np.asarray([52.0, -18.0, 20.0])
    forward = [
        _observation(index, 0, _state(first_center, -0.2 + index * 0.01))
        for index in range(5)
    ] + [
        _observation(index, 1, _state(second_center, 2.0 + (index - 5) * 0.01))
        for index in range(5, 10)
    ]
    backward = list(forward)

    result = BidirectionalCameraSmoother().smooth(forward, backward)

    assert result.shot_count == 2
    assert result.states[4].shot_id == 0
    assert result.states[5].shot_id == 1
    assert result.states[4].camera_center_xyz_m == pytest.approx(first_center)
    assert result.states[5].camera_center_xyz_m == pytest.approx(second_center)
    assert abs(result.states[5].pan_rad - result.states[4].pan_rad) > 1.0


def test_backward_observation_with_wrong_shot_id_is_ignored() -> None:
    center = np.asarray([-18.0, 30.0, 24.0])
    forward = [
        _observation(index, 0, _state(center, 0.02 * index)) for index in range(6)
    ]
    backward = [
        _observation(index, 1, _state(center, 1.5)) for index in range(6)
    ]

    result = BidirectionalCameraSmoother().smooth(forward, backward)

    assert max(abs(state.pan_rad) for state in result.states) < 0.2


def test_offline_result_round_trip_uses_safe_npz(tmp_path) -> None:
    center = np.asarray([-18.0, 30.0, 24.0])
    observations = [
        _observation(index, 0, _state(center, 0.02 * index)) for index in range(5)
    ]
    result = BidirectionalCameraSmoother().smooth(observations, observations)

    json_path, npz_path = save_offline_smoothing_result(
        result, tmp_path / "registration"
    )
    metadata, arrays = load_offline_smoothing_result(json_path)

    assert npz_path.is_file()
    assert metadata["frame_count"] == 5
    assert arrays["pitch_to_image"].shape == (5, 3, 3)
    assert arrays["camera_parameters"].shape == (5, 7)
    assert np.all(np.isfinite(arrays["camera_centers_xyz_m"]))


def test_late_shot_center_retrospectively_fits_planar_homographies() -> None:
    center = np.asarray([-18.0, 30.0, 24.0])
    physical = _state(center, 0.08)
    planar = CameraState(
        status=CameraTrackingStatus.TRACKED,
        pan_rad=float("nan"),
        pan_velocity_rad_s=0.0,
        covariance=np.diag([1e6, 1e6]),
        image_to_pitch=physical.image_to_pitch,
        pitch_to_image=physical.pitch_to_image,
        confidence=0.85,
        registration_mode=RegistrationMode.BROADCAST,
        measurement_tier=MeasurementTier.SAFE,
        shot_id=0,
        camera_model="broadcast_planar_homography",
    )
    forward = [
        _observation(0, 0, planar),
        _observation(1, 0, physical),
    ]

    result = BidirectionalCameraSmoother().smooth(forward, [])

    assert result.smoothed_frame_count == 2
    assert result.states[0].camera_model.endswith("offline_rts")
    assert result.states[0].focal_px == pytest.approx(physical.focal_px, rel=5e-3)


def test_offline_smoother_keeps_original_state_when_rts_leaves_physical_bounds() -> None:
    center = np.asarray([-18.0, 30.0, 24.0])
    unstable = CameraState(
        **{
            **_state(center, 0.08).__dict__,
            "zoom_velocity_log_s": 1_000_000.0,
        }
    )
    forward = [_observation(0, 0, unstable), _observation(1, 0, _missing(0))]

    result = BidirectionalCameraSmoother().smooth(forward, [])

    assert result.states[0].pitch_to_image is not None
    assert result.states[1].measurement_tier is MeasurementTier.UNAVAILABLE
    assert result.smoothed_frame_count == 1
