from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from app.field_registration.camera_model import CameraRigProfile
from app.field_registration.contact_point import GroundContactPointSelector
from app.field_registration.decode import decode_heatmap_peak, sample_pitch_segment
from app.field_registration.geometry import transform_points
from app.field_registration.initializer import HomographyInitializer
from app.field_registration.lens import LensCalibration, LensCalibrationError, LensModel, LensUndistorter
from app.field_registration.metrics import evaluate_registration, template_jitter_px
from app.field_registration.pan_filter import PanExtendedKalmanFilter
from app.field_registration.optical_flow import MaskedSparseOpticalFlow
from app.field_registration.perception import PitchPerceptionOutput, StaticPerceptionBackend
from app.field_registration.pitch_model import PitchDimensions, PitchModel, VenueProfile
from app.field_registration.point_line_refiner import PanOnlyPointLineRefiner
from app.field_registration.projection import PitchProjector
from app.field_registration.rig_calibration import PanAnchor, calibrate_fixed_pan_rig
from app.field_registration.tracker import FieldRegistrationConfig, FieldRegistrationCore
from app.field_registration.types import (
    CameraState,
    CameraTrackingStatus,
    LineObservation,
    PointObservation,
)


def _look_at_rotation(camera_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    forward = target_xyz - camera_xyz
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.vstack((right, down, forward))


def _rig() -> CameraRigProfile:
    camera = np.array([-12.0, 34.0, 22.0])
    target = np.array([52.5, 34.0, 0.0])
    return CameraRigProfile(
        camera_id="synthetic-main",
        image_size=(1280, 720),
        lens_model=LensModel.PINHOLE,
        camera_matrix=np.array(
            [[920.0, 0.0, 640.0], [0.0, 920.0, 360.0], [0.0, 0.0, 1.0]]
        ),
        distortion_coefficients=np.zeros(5),
        pitch_length_m=105.0,
        pitch_width_m=68.0,
        pan_axis_origin_xyz_m=camera,
        pan_axis_direction=np.array([0.0, 0.0, 1.0]),
        optical_center_offset_m=np.zeros(3),
        base_rotation_world_to_camera=_look_at_rotation(camera, target),
        pan_limits_rad=(-0.8, 0.8),
    )


def _camera_state(rig: CameraRigProfile, pan_rad: float = 0.0) -> CameraState:
    return CameraState(
        status=CameraTrackingStatus.TRACKED,
        pan_rad=pan_rad,
        pan_velocity_rad_s=0.0,
        covariance=np.diag([1e-8, 1e-5]),
        image_to_pitch=rig.image_to_pitch_homography(pan_rad),
        pitch_to_image=rig.pitch_to_image_homography(pan_rad),
        confidence=0.95,
    )


def test_venue_profile_round_trip_uses_metric_dimensions(tmp_path) -> None:
    profile = VenueProfile("stadium-a", PitchDimensions(length_m=104.0, width_m=67.0))
    path = tmp_path / "venue.json"

    profile.save(path)
    loaded = VenueProfile.load(path)

    assert loaded == profile
    assert PitchModel(loaded.pitch).contains((104.0, 67.0))
    assert not PitchModel(loaded.pitch).contains((104.1, 67.0))


@pytest.mark.parametrize("lens_model,distortion", [
    (LensModel.PINHOLE, np.zeros(5)),
    (LensModel.FISHEYE, np.zeros(4)),
])
def test_lens_calibration_round_trip_and_rectification(tmp_path, lens_model, distortion) -> None:
    calibration = LensCalibration(
        lens_model=lens_model,
        camera_matrix=np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        ),
        distortion_coefficients=distortion,
        image_size=(640, 480),
        reprojection_error_px=0.2,
    )
    path = tmp_path / "lens.npz"
    calibration.save(path)
    loaded = LensCalibration.load(path)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    undistorter = LensUndistorter(loaded)

    assert loaded.lens_model is lens_model
    assert undistorter.rectify(frame).shape == frame.shape
    points = np.array([[160.0, 120.0], [100.0, 80.0]])
    rectified_points = undistorter.rectify_points(points, (320, 240))
    assert np.all(np.isfinite(rectified_points))
    assert rectified_points[0] == pytest.approx(points[0], abs=1e-3)
    if lens_model is LensModel.PINHOLE:
        assert np.allclose(rectified_points, points, atol=1e-3)


def test_lens_calibration_rejects_aspect_ratio_change() -> None:
    calibration = LensCalibration(
        LensModel.PINHOLE,
        np.eye(3),
        np.zeros(5),
        (640, 480),
    )
    with pytest.raises(LensCalibrationError, match="aspect ratio"):
        calibration.scaled_camera_matrix((640, 360))


def test_camera_rig_round_trip_and_pan_changes_projection(tmp_path) -> None:
    rig = _rig()
    path = tmp_path / "rig.json"
    rig.save(path)
    loaded = CameraRigProfile.load(path)
    pitch_points = np.array([[20.0, 10.0], [52.5, 34.0], [90.0, 58.0]])

    image_zero = transform_points(pitch_points, loaded.pitch_to_image_homography(0.0))
    recovered = transform_points(image_zero, loaded.image_to_pitch_homography(0.0))
    image_panned = transform_points(pitch_points, loaded.pitch_to_image_homography(0.1))

    assert loaded.camera_id == rig.camera_id
    assert np.allclose(recovered, pitch_points, atol=1e-8)
    assert np.mean(np.linalg.norm(image_panned - image_zero, axis=1)) > 20.0
    assert json.loads(path.read_text())["lens_model"] == "pinhole"


def test_magsac_initializer_rejects_outlier_and_recovers_homography() -> None:
    rig = _rig()
    pitch = PitchModel().grid(8, 6)
    image = transform_points(pitch, rig.pitch_to_image_homography(0.08))
    visible = (
        (image[:, 0] > 5)
        & (image[:, 0] < 1275)
        & (image[:, 1] > 5)
        & (image[:, 1] < 715)
    )
    pitch = pitch[visible]
    image = image[visible]
    observations = [
        PointObservation(str(index), tuple(point), tuple(world), 0.95)
        for index, (point, world) in enumerate(zip(image, pitch))
    ]
    observations.append(PointObservation("outlier", (100.0, 100.0), (100.0, 60.0), 0.99))

    result = HomographyInitializer().estimate(observations, (1280, 720), (105.0, 68.0))

    assert result.success, result.rejection_reason
    assert len(result.inlier_indices) >= len(observations) - 2
    recovered = transform_points(image, result.image_to_pitch)
    assert np.median(np.linalg.norm(recovered - pitch, axis=1)) < 0.02


def test_magsac_initializer_rejects_collinear_geometry() -> None:
    observations = [
        PointObservation(str(index), (100.0 + index * 50, 200.0), (index * 10.0, 0.0))
        for index in range(6)
    ]
    result = HomographyInitializer().estimate(observations, (1280, 720), (105.0, 68.0))
    assert not result.success
    assert result.rejection_reason in {
        "insufficient_image_coverage",
        "insufficient_world_coverage",
        "near_collinear_points",
    }


def test_registration_metrics_measure_known_perturbation() -> None:
    rig = _rig()
    truth = rig.image_to_pitch_homography(0.0)
    predicted = rig.image_to_pitch_homography(0.01)
    pitch = PitchModel().grid(6, 4)
    image = transform_points(pitch, np.linalg.inv(truth))

    metrics = evaluate_registration(predicted, truth, image, pitch, (1280, 720))
    jitter = template_jitter_px(
        [np.linalg.inv(truth), np.linalg.inv(predicted)],
        pitch,
    )

    assert metrics.pitch_projection_m.median is not None
    assert metrics.pitch_projection_m.median > 0.1
    assert metrics.grid_projection_m.p95 is not None
    assert jitter.mean is not None and jitter.mean > 0.0


def test_pan_filter_predicts_updates_and_rejects_large_innovation() -> None:
    pan_filter = PanExtendedKalmanFilter()
    pan_filter.reset(0.0)
    pan_filter.predict(0.04)
    assert pan_filter.update(0.01, 1e-3)
    before = pan_filter.pan_rad
    assert not pan_filter.update(2.0, 1e-6)
    assert pan_filter.pan_rad == before


def test_contact_selector_prefers_head_then_ankles_and_rejects_truncation() -> None:
    selector = GroundContactPointSelector()
    head = selector.select(
        (100, 100, 140, 220),
        (640, 480),
        contact_head_xy=(121.0, 214.0),
        contact_head_confidence=0.9,
    )
    ankles = selector.select(
        (100, 100, 140, 220),
        (640, 480),
        ankles_xyc=np.array([[112.0, 215.0, 0.8], [130.0, 216.0, 0.9]]),
    )
    truncated = selector.select((100, 100, 140, 479), (640, 480))

    assert head.source == "contact_head"
    assert ankles.source == "ankles"
    assert truncated.image_xy is None
    assert truncated.rejection_reason == "truncated_bbox"


def test_pitch_projector_propagates_uncertainty_and_filters_outside_pitch() -> None:
    rig = _rig()
    state = _camera_state(rig)
    projector = PitchProjector(PitchModel(), rig)
    image = transform_points(np.array([[52.5, 34.0]]), state.pitch_to_image)[0]

    coordinate = projector.image_to_pitch(tuple(image), state, image_covariance_px2=np.eye(2))
    outside_image = transform_points(np.array([[150.0, 34.0]]), state.pitch_to_image)[0]
    outside = projector.image_to_pitch(tuple(outside_image), state)

    assert coordinate.xy_m == pytest.approx((52.5, 34.0), abs=1e-6)
    assert coordinate.sigma_m is not None and coordinate.sigma_m > 0.0
    assert outside.xy_m is None


def test_pan_only_point_line_refiner_recovers_global_pan() -> None:
    rig = _rig()
    true_pan = 0.13
    pitch_points = PitchModel().grid(8, 6)
    image_points = transform_points(pitch_points, rig.pitch_to_image_homography(true_pan))
    visible = (
        (image_points[:, 0] > 0)
        & (image_points[:, 0] < 1280)
        & (image_points[:, 1] > 0)
        & (image_points[:, 1] < 720)
    )
    points = [
        PointObservation(str(index), tuple(image), tuple(pitch), 0.95)
        for index, (image, pitch) in enumerate(
            zip(image_points[visible], pitch_points[visible])
        )
    ]
    centre_line_pitch = sample_pitch_segment((52.5, 0.0), (52.5, 68.0), 80)
    centre_line_image = transform_points(
        centre_line_pitch, rig.pitch_to_image_homography(true_pan)
    )
    line = LineObservation("centre_line", centre_line_image, centre_line_pitch, 0.9)

    result = PanOnlyPointLineRefiner(rig).refine(
        points,
        [line],
        (1280, 720),
        global_search=True,
    )

    assert result.success, result.rejection_reason
    assert result.pan_rad == pytest.approx(true_pan, abs=2e-4)
    assert result.mean_point_error_px is not None
    assert result.mean_point_error_px < 0.2


def test_masked_sparse_flow_tracks_translation_and_excludes_dynamic_box() -> None:
    previous = np.zeros((240, 320), dtype=np.uint8)
    for y_coord in range(50, 220, 25):
        for x_coord in range(30, 300, 30):
            cv2.circle(previous, (x_coord, y_coord), 3, 255, -1)
    transform = np.float32([[1.0, 0.0, 4.0], [0.0, 1.0, 2.0]])
    current = cv2.warpAffine(previous, transform, (320, 240))
    flow_tracker = MaskedSparseOpticalFlow()

    flow = flow_tracker.track(previous, current, dynamic_boxes_xyxy=np.array([[0, 0, 40, 240]]))

    assert flow.count >= flow_tracker.config.minimum_tracks
    displacement = np.median(flow.current_xy - flow.previous_xy, axis=0)
    assert displacement == pytest.approx((4.0, 2.0), abs=0.2)
    assert np.all(flow.previous_xy[:, 0] > 40)


def test_heatmap_decoder_returns_subpixel_local_centroid() -> None:
    heatmap = np.zeros((8, 10), dtype=np.float32)
    heatmap[3, 4] = 4.0
    heatmap[3, 5] = 3.0

    point, confidence = decode_heatmap_peak(heatmap, output_size=(100, 80))

    assert 44.0 < point[0] < 50.0
    assert 30.0 < point[1] < 40.0
    assert confidence == 4.0


def test_field_registration_core_relocalizes_then_expires_prediction() -> None:
    rig = _rig()
    true_pan = 0.07
    pitch_points = PitchModel().grid(7, 5)
    image_points = transform_points(pitch_points, rig.pitch_to_image_homography(true_pan))
    visible = (
        (image_points[:, 0] > 0)
        & (image_points[:, 0] < 1280)
        & (image_points[:, 1] > 0)
        & (image_points[:, 1] < 720)
    )
    observations = tuple(
        PointObservation(str(index), tuple(image), tuple(pitch), 0.95)
        for index, (image, pitch) in enumerate(
            zip(image_points[visible], pitch_points[visible])
        )
    )
    backend = StaticPerceptionBackend(
        [PitchPerceptionOutput(points=observations), PitchPerceptionOutput()]
    )
    core = FieldRegistrationCore(
        PitchModel(),
        backend,
        rig,
        FieldRegistrationConfig(
            fps=25.0,
            normal_semantic_interval=100,
            stable_semantic_interval=100,
            max_prediction_frames=2,
        ),
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    first = core.process(frame, 0)
    second = core.process(frame, 1)
    third = core.process(frame, 2)
    fourth = core.process(frame, 3)

    assert first.camera_state.status is CameraTrackingStatus.RELOCALIZED
    assert first.camera_state.pan_rad == pytest.approx(true_pan, abs=2e-4)
    assert second.camera_state.status is CameraTrackingStatus.PREDICTED
    assert third.camera_state.status is CameraTrackingStatus.PREDICTED
    assert fourth.camera_state.status is CameraTrackingStatus.LOST
    assert fourth.camera_state.image_to_pitch is None


def test_multi_anchor_rig_calibration_recovers_fixed_camera_and_relative_pan() -> None:
    rig = _rig()
    pitch_points = PitchModel().grid(8, 6)
    anchors = []
    for anchor_id, pan in (("left", -0.12), ("right", 0.11)):
        image_points = transform_points(pitch_points, rig.pitch_to_image_homography(pan))
        visible = (
            (image_points[:, 0] > -100.0)
            & (image_points[:, 0] < 1380.0)
            & (image_points[:, 1] > -100.0)
            & (image_points[:, 1] < 820.0)
        )
        observations = tuple(
            PointObservation(str(index), tuple(image), tuple(pitch), 0.95)
            for index, (image, pitch) in enumerate(
                zip(image_points[visible], pitch_points[visible])
            )
        )
        anchors.append(PanAnchor(anchor_id, observations))
    lens = LensCalibration(
        LensModel.PINHOLE,
        rig.camera_matrix,
        rig.distortion_coefficients,
        rig.image_size,
    )

    result = calibrate_fixed_pan_rig(
        "recovered",
        lens,
        anchors,
        (105.0, 68.0),
        (-0.8, 0.8),
    )

    expected_delta = 0.23
    recovered_delta = result.anchor_pan_rad["right"] - result.anchor_pan_rad["left"]
    assert recovered_delta == pytest.approx(expected_delta, abs=2e-3)
    assert result.profile.pan_axis_origin_xyz_m == pytest.approx(
        rig.pan_axis_origin_xyz_m, abs=1e-3
    )
    assert result.camera_centre_spread_m < 1e-3
    assert max(result.anchor_reprojection_error_px.values()) < 1e-3
