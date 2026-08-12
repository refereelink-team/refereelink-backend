from __future__ import annotations

import numpy as np
import pytest

from app.field_registration.broadcast_camera import (
    BroadcastCameraEstimator,
    BroadcastCameraParameters,
    BroadcastCameraStateFilter,
    pan_tilt_roll_from_rotation,
    rotation_from_pan_tilt_roll,
)
from app.field_registration.geometry import transform_points
from app.field_registration.pitch_model import PitchModel


@pytest.mark.parametrize(
    "pan,tilt,roll",
    [
        (0.0, 0.25, 0.0),
        (0.42, 0.31, -0.04),
        (-0.65, 0.48, 0.08),
    ],
)
def test_pan_tilt_roll_rotation_round_trip(
    pan: float, tilt: float, roll: float
) -> None:
    rotation = rotation_from_pan_tilt_roll(pan, tilt, roll)

    recovered = pan_tilt_roll_from_rotation(rotation)

    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-10)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-10)
    assert recovered == pytest.approx((pan, tilt, roll), abs=1e-10)


def _physical_camera(
    *,
    pan: float = 0.04,
    tilt: float = 0.29,
    roll: float = -0.025,
    focal: float = 1120.0,
) -> BroadcastCameraParameters:
    return BroadcastCameraParameters(
        camera_center_xyz_m=np.asarray([-18.0, 32.0, 24.0]),
        pan_rad=pan,
        tilt_rad=tilt,
        roll_rad=roll,
        log_focal_px=float(np.log(focal)),
        image_size=(1280, 720),
    )


def test_broadcast_camera_decomposition_recovers_physical_homography() -> None:
    truth = _physical_camera()
    estimator = BroadcastCameraEstimator()

    recovered = estimator.decompose(
        truth.pitch_to_image_homography(), truth.image_size
    )

    grid = PitchModel().grid(18, 12)
    truth_image = transform_points(grid, truth.pitch_to_image_homography())
    recovered_image = transform_points(grid, recovered.pitch_to_image_homography())
    errors = np.linalg.norm(truth_image - recovered_image, axis=1)
    assert recovered.focal_px == pytest.approx(truth.focal_px, rel=2e-3)
    assert recovered.camera_center_xyz_m == pytest.approx(
        truth.camera_center_xyz_m, abs=0.08
    )
    assert np.median(errors) < 0.05
    assert np.percentile(errors, 95) < 0.15


def test_fixed_center_fit_recovers_pan_tilt_roll_and_zoom() -> None:
    estimator = BroadcastCameraEstimator()
    initial = _physical_camera(pan=0.0, tilt=0.27, roll=0.0, focal=1040.0)
    truth = _physical_camera(pan=0.10, tilt=0.32, roll=0.035, focal=1240.0)

    recovered = estimator.fit_with_fixed_center(
        truth.pitch_to_image_homography(),
        truth.image_size,
        truth.camera_center_xyz_m,
        initial,
    )

    assert recovered.pan_rad == pytest.approx(truth.pan_rad, abs=2e-4)
    assert recovered.tilt_rad == pytest.approx(truth.tilt_rad, abs=2e-4)
    assert recovered.roll_rad == pytest.approx(truth.roll_rad, abs=2e-4)
    assert recovered.focal_px == pytest.approx(truth.focal_px, rel=5e-4)
    assert recovered.fit_p95_error_px < 0.1


def test_broadcast_camera_filter_tracks_motion_and_rejects_outlier() -> None:
    camera_filter = BroadcastCameraStateFilter()
    initial = np.asarray([0.0, 0.30, 0.01, np.log(1050.0)])
    assert camera_filter.update(initial, np.full(4, 1e-4))

    for index in range(1, 20):
        camera_filter.predict(0.04)
        measurement = initial + np.asarray(
            [index * 0.002, index * 0.0005, 0.0, index * 0.001]
        )
        assert camera_filter.update(measurement, np.full(4, 2e-5))

    before = camera_filter.state.copy()
    outlier = before[:4] + np.asarray([2.0, 1.0, 1.0, 1.5])
    assert not camera_filter.update(outlier, np.full(4, 1e-8))
    assert camera_filter.state == pytest.approx(before)
    assert camera_filter.state[4] > 0.0
    assert camera_filter.state[5] > 0.0
    assert camera_filter.state[6] > 0.0


@pytest.mark.parametrize(
    "tilt,log_focal",
    [
        (np.pi / 2.0, np.log(1120.0)),
        (0.3, 1_000.0),
        (0.3, -1_000.0),
    ],
)
def test_broadcast_camera_rejects_nonphysical_parameter_ranges(
    tilt: float,
    log_focal: float,
) -> None:
    with pytest.raises(ValueError):
        BroadcastCameraParameters(
            np.asarray([-18.0, 32.0, 24.0]),
            0.0,
            tilt,
            0.0,
            log_focal,
            (1280, 720),
        )
