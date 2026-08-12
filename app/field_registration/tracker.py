"""Constrained pitch tracker state machine and adaptive refresh scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.field_registration.broadcast_camera import (
    BroadcastCameraEstimator,
    BroadcastCameraParameters,
    BroadcastCameraStateFilter,
)
from app.field_registration.camera_model import CameraRigProfile
from app.field_registration.full_homography_refiner import (
    FullHomographyPointLineRefiner,
)
from app.field_registration.geometry import transform_points
from app.field_registration.initializer import HomographyInitializer
from app.field_registration.optical_flow import MaskedSparseOpticalFlow, OpticalFlowResult
from app.field_registration.pan_filter import PanExtendedKalmanFilter
from app.field_registration.perception import PitchPerceptionBackend, PitchPerceptionOutput
from app.field_registration.pitch_model import PitchModel
from app.field_registration.point_line_refiner import PanOnlyPointLineRefiner
from app.field_registration.shot import ShotBoundaryDetector
from app.field_registration.types import (
    CameraState,
    CameraTrackingStatus,
    FieldRegistrationFrame,
    MeasurementTier,
    PointObservation,
    RegistrationMode,
)


@dataclass(frozen=True)
class FieldRegistrationConfig:
    fps: float = 25.0
    normal_semantic_interval: int = 5
    stable_semantic_interval: int = 10
    fast_pan_velocity_rad_s: float = 0.20
    low_confidence_threshold: float = 0.45
    measurement_confidence_threshold: float = 0.55
    max_prediction_frames: int = 6
    minimum_flow_tracks: int = 12
    fast_semantic_interval: int = 3
    safe_flow_without_semantic_seconds: float = 0.5
    preview_flow_without_semantic_seconds: float = 1.0
    require_physical_camera_for_safe: bool = True
    minimum_safe_semantic_lines: int = 2
    allow_point_only_safe: bool = False
    registration_mode: RegistrationMode | str | None = None


class FieldRegistrationCore:
    """Track camera pan and derive a guarded image↔pitch transform per frame."""

    def __init__(
        self,
        pitch_model: PitchModel,
        perception_backend: PitchPerceptionBackend,
        rig_profile: Optional[CameraRigProfile] = None,
        config: FieldRegistrationConfig | None = None,
        initializer: Optional[HomographyInitializer] = None,
        optical_flow: Optional[MaskedSparseOpticalFlow] = None,
        pan_filter: Optional[PanExtendedKalmanFilter] = None,
        full_homography_refiner: Optional[FullHomographyPointLineRefiner] = None,
        shot_detector: Optional[ShotBoundaryDetector] = None,
        broadcast_camera_estimator: Optional[BroadcastCameraEstimator] = None,
        broadcast_camera_filter: Optional[BroadcastCameraStateFilter] = None,
    ) -> None:
        self.pitch_model = pitch_model
        self.perception_backend = perception_backend
        self.rig_profile = rig_profile
        self.config = config or FieldRegistrationConfig()
        if self.config.fps <= 0:
            raise ValueError("registration fps must be positive")
        configured_mode = self.config.registration_mode
        if configured_mode is None:
            self.registration_mode = (
                RegistrationMode.RIG_PAN
                if rig_profile is not None
                else RegistrationMode.BROADCAST
            )
        else:
            self.registration_mode = RegistrationMode(configured_mode)
        if self.registration_mode is RegistrationMode.RIG_PAN and rig_profile is None:
            raise ValueError("rig_pan registration requires a camera rig profile")
        self.initializer = initializer or HomographyInitializer()
        self.optical_flow = optical_flow or MaskedSparseOpticalFlow()
        self.pan_filter = pan_filter or PanExtendedKalmanFilter()
        self.full_homography_refiner = (
            full_homography_refiner or FullHomographyPointLineRefiner()
        )
        self.shot_detector = shot_detector or ShotBoundaryDetector()
        self.broadcast_camera_estimator = (
            broadcast_camera_estimator or BroadcastCameraEstimator(pitch_model)
        )
        self.broadcast_camera_filter = (
            broadcast_camera_filter or BroadcastCameraStateFilter()
        )
        self.pan_refiner = (
            PanOnlyPointLineRefiner(rig_profile)
            if self.registration_mode is RegistrationMode.RIG_PAN
            and rig_profile is not None
            else None
        )
        self._previous_frame: Optional[np.ndarray] = None
        self._previous_dynamic_boxes: Optional[np.ndarray] = None
        self._last_state: Optional[CameraState] = None
        self._last_semantic_frame: Optional[int] = None
        self._prediction_age = 0
        self._last_frame_index: Optional[int] = None
        self._shot_id = 0
        self._shot_camera_center: Optional[np.ndarray] = None
        self.semantic_inference_count = 0
        self.flow_update_count = 0
        self.prediction_count = 0
        self.lost_count = 0
        self.relocalization_count = 0
        self.shot_cut_count = 0

    @property
    def state(self) -> Optional[CameraState]:
        return self._last_state

    def reset(self) -> None:
        self._reset_tracking_state()
        self._last_frame_index = None
        self._shot_id = 0
        self.shot_detector.reset()

    def _reset_tracking_state(self) -> None:
        self._previous_frame = None
        self._previous_dynamic_boxes = None
        self._last_state = None
        self._last_semantic_frame = None
        self._prediction_age = 0
        self.pan_filter = PanExtendedKalmanFilter(self.pan_filter.config)
        self.broadcast_camera_filter = BroadcastCameraStateFilter(
            self.broadcast_camera_filter.config
        )
        self._shot_camera_center = None

    def _semantic_interval(self) -> int:
        state = self._last_state
        if state is None or state.status in {
            CameraTrackingStatus.INITIALIZING,
            CameraTrackingStatus.LOST,
        }:
            return 1
        if (
            state.confidence < self.config.low_confidence_threshold
            or abs(state.pan_velocity_rad_s) >= self.config.fast_pan_velocity_rad_s
        ):
            return max(self.config.fast_semantic_interval, 1)
        if state.confidence >= 0.80 and abs(state.pan_velocity_rad_s) < 0.05:
            return self.config.stable_semantic_interval
        return self.config.normal_semantic_interval

    def _should_run_semantic(self, frame_index: int) -> bool:
        return (
            self._last_semantic_frame is None
            or frame_index - self._last_semantic_frame >= self._semantic_interval()
        )

    def _field_polygon(self, state: CameraState) -> Optional[np.ndarray]:
        if state.pitch_to_image is None:
            return None
        dimensions = self.pitch_model.dimensions
        corners = np.array(
            [
                [0.0, 0.0],
                [dimensions.length_m, 0.0],
                [dimensions.length_m, dimensions.width_m],
                [0.0, dimensions.width_m],
            ],
            dtype=np.float64,
        )
        projected = transform_points(corners, state.pitch_to_image)
        if not np.all(np.isfinite(projected)):
            return None
        # A valid projective camera may place invisible field corners far
        # outside the image. Bound polygon coordinates before OpenCV converts
        # them to int32 so extreme partial views cannot overflow fillPoly.
        maximum = 1_000_000.0
        return np.clip(projected, -maximum, maximum)

    def _flow_result(
        self,
        frame: np.ndarray,
    ) -> tuple[OpticalFlowResult, tuple[PointObservation, ...]]:
        empty = OpticalFlowResult(
            np.empty((0, 2)), np.empty((0, 2)), np.empty(0), 0.0
        )
        if (
            self._previous_frame is None
            or self._last_state is None
            or self._last_state.image_to_pitch is None
        ):
            return empty, ()
        flow = self.optical_flow.track(
            self._previous_frame,
            frame,
            dynamic_boxes_xyxy=self._previous_dynamic_boxes,
            field_polygon_xy=self._field_polygon(self._last_state),
        )
        pitch_xy, current_xy = self.optical_flow.field_correspondences(
            flow,
            self._last_state.image_to_pitch,
            self.pitch_model,
        )
        observations = tuple(
            PointObservation(
                label=f"flow-{index}",
                image_xy=(float(image[0]), float(image[1])),
                pitch_xy_m=(float(pitch[0]), float(pitch[1])),
                confidence=float(np.clip(flow.valid_ratio, 0.1, 1.0)),
                sigma_px=max(float(flow.mean_error_px or 0.5), 0.5),
                source="optical_flow",
            )
            for index, (pitch, image) in enumerate(zip(pitch_xy, current_xy))
        )
        return flow, observations

    def _state_from_pan(
        self,
        status: CameraTrackingStatus,
        confidence: float,
        point_inliers: int,
        mean_error_px: Optional[float],
        p95_error_px: Optional[float],
        semantic_age: int,
        measurement_tier: MeasurementTier = MeasurementTier.SAFE,
        flow_inliers: int = 0,
    ) -> CameraState:
        assert self.rig_profile is not None
        pan = self.pan_filter.pan_rad
        return CameraState(
            status=status,
            pan_rad=pan,
            pan_velocity_rad_s=self.pan_filter.velocity_rad_s,
            covariance=self.pan_filter.covariance.copy(),
            image_to_pitch=self.rig_profile.image_to_pitch_homography(pan),
            pitch_to_image=self.rig_profile.pitch_to_image_homography(pan),
            point_inliers=point_inliers,
            spatial_coverage=0.0,
            mean_segment_error_px=mean_error_px,
            p95_segment_error_px=p95_error_px,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            age_since_semantic_update=semantic_age,
            registration_mode=self.registration_mode,
            measurement_tier=measurement_tier,
            shot_id=self._shot_id,
            camera_model="fixed_rig_pan",
            focal_px=float(self.rig_profile.camera_matrix[0, 0]),
            flow_inliers=flow_inliers,
            projection_uncertainty=float(np.trace(self.pan_filter.covariance)),
        )

    def _broadcast_state_from_initialization(
        self,
        initialization: object,
        image_size: tuple[int, int],
        status: CameraTrackingStatus,
        confidence: float,
        semantic_age: int,
        measurement_tier: MeasurementTier,
        flow_inliers: int = 0,
        line_result: object | None = None,
    ) -> CameraState:
        image_to_pitch = getattr(line_result, "image_to_pitch", None)
        pitch_to_image = getattr(line_result, "pitch_to_image", None)
        if image_to_pitch is None:
            image_to_pitch = initialization.image_to_pitch
        if pitch_to_image is None:
            pitch_to_image = initialization.pitch_to_image
        mean_error = getattr(line_result, "mean_line_error_px", None)
        p95_error = getattr(line_result, "p95_line_error_px", None)
        point_inliers = getattr(line_result, "point_inliers", None)
        if point_inliers is None:
            point_inliers = len(initialization.inlier_indices)
        visible_segments = int(getattr(line_result, "visible_segment_count", 0) or 0)
        uncertainty = initialization.p95_reprojection_error_px
        if p95_error is not None:
            uncertainty = max(float(uncertainty or 0.0), float(p95_error))
        physical = self._update_broadcast_camera(
            pitch_to_image,
            image_size,
            float(uncertainty or 2.0),
        )
        camera_model = "broadcast_planar_homography"
        pan_rad = float("nan")
        pan_velocity = 0.0
        tilt_rad = float("nan")
        roll_rad = float("nan")
        focal_px = None
        center = None
        parameter_covariance = None
        tilt_velocity = 0.0
        zoom_velocity = 0.0
        covariance = np.diag([1e6, 1e6])
        if physical is not None:
            pitch_to_image = physical.pitch_to_image_homography()
            image_to_pitch = physical.image_to_pitch_homography()
            camera_model = "broadcast_tripod_pan_tilt_zoom"
            pan_rad = physical.pan_rad
            tilt_rad = physical.tilt_rad
            roll_rad = physical.roll_rad
            focal_px = physical.focal_px
            center = physical.camera_center_xyz_m.copy()
            filter_state = self.broadcast_camera_filter.state
            filter_covariance = self.broadcast_camera_filter.covariance
            pan_velocity = float(filter_state[4])
            tilt_velocity = float(filter_state[5])
            zoom_velocity = float(filter_state[6])
            covariance = filter_covariance[np.ix_([0, 4], [0, 4])]
            parameter_covariance = filter_covariance.copy()
            uncertainty = max(float(uncertainty or 0.0), physical.fit_p95_error_px)
        elif (
            self.registration_mode is RegistrationMode.BROADCAST
            and self.config.require_physical_camera_for_safe
            and measurement_tier is MeasurementTier.SAFE
        ):
            # A projective H can satisfy its own point correspondences while
            # still placing the field lines far from the image evidence (for
            # example after a systematic landmark-ID error).  Do not expose
            # such a planar-only estimate to metric/high-risk consumers.  It
            # remains available as PREVIEW for diagnostics and can recover to
            # SAFE as soon as the shot-local physical camera fit succeeds.
            measurement_tier = MeasurementTier.PREVIEW
        return CameraState(
            status=status,
            pan_rad=pan_rad,
            pan_velocity_rad_s=pan_velocity,
            covariance=covariance,
            image_to_pitch=image_to_pitch,
            pitch_to_image=pitch_to_image,
            point_inliers=int(point_inliers),
            visible_segment_count=visible_segments,
            spatial_coverage=initialization.image_coverage,
            mean_segment_error_px=mean_error,
            p95_segment_error_px=p95_error,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            age_since_semantic_update=semantic_age,
            registration_mode=self.registration_mode,
            measurement_tier=measurement_tier,
            shot_id=self._shot_id,
            camera_model=camera_model,
            focal_px=focal_px,
            tilt_rad=tilt_rad,
            roll_rad=roll_rad,
            flow_inliers=flow_inliers,
            projection_uncertainty=(
                float(uncertainty) if uncertainty is not None else None
            ),
            camera_center_xyz_m=center,
            tilt_velocity_rad_s=tilt_velocity,
            zoom_velocity_log_s=zoom_velocity,
            camera_parameter_covariance=parameter_covariance,
        )

    def _update_broadcast_camera(
        self,
        pitch_to_image: np.ndarray,
        image_size: tuple[int, int],
        reprojection_error_px: float,
    ) -> Optional[BroadcastCameraParameters]:
        """Factor a trusted H into one shot-local tripod camera observation."""

        try:
            if self._shot_camera_center is None:
                measured = self.broadcast_camera_estimator.decompose(
                    pitch_to_image, image_size
                )
                if (
                    measured.fit_p95_error_px
                    > self.broadcast_camera_estimator.config.maximum_physical_fit_p95_px
                ):
                    return None
                self._shot_camera_center = measured.camera_center_xyz_m.copy()
                self.broadcast_camera_filter.reset(measured.observation_vector)
            else:
                initial = (
                    self.broadcast_camera_filter.parameters(
                        self._shot_camera_center, image_size
                    )
                    if self.broadcast_camera_filter.initialized
                    else None
                )
                measured = self.broadcast_camera_estimator.fit_with_fixed_center(
                    pitch_to_image,
                    image_size,
                    self._shot_camera_center,
                    initial,
                )
                if (
                    measured.fit_p95_error_px
                    > self.broadcast_camera_estimator.config.maximum_physical_fit_p95_px
                ):
                    return None
                normalized_error = max(
                    reprojection_error_px / max(image_size),
                    5e-5,
                )
                variances = np.asarray(
                    [
                        normalized_error**2,
                        normalized_error**2,
                        normalized_error**2,
                        (2.0 * normalized_error) ** 2,
                    ],
                    dtype=np.float64,
                )
                if not self.broadcast_camera_filter.update(
                    measured.observation_vector, variances
                ):
                    return None
            filtered = self.broadcast_camera_filter.parameters(
                self._shot_camera_center,
                image_size,
            )
            fitted = self.broadcast_camera_estimator.with_fit_error(
                filtered, pitch_to_image
            )
            if (
                fitted.fit_p95_error_px
                > self.broadcast_camera_estimator.config.maximum_physical_fit_p95_px
            ):
                # Preserve the projective measurement for this frame instead
                # of turning filter lag into a false SAFE physical matrix.
                return None
            return fitted
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return None

    def _broadcast_prediction_state(
        self,
        confidence: float,
        semantic_age: int,
    ) -> Optional[CameraState]:
        if (
            self._shot_camera_center is None
            or not self.broadcast_camera_filter.initialized
            or self._previous_frame is None
        ):
            return None
        image_size = (self._previous_frame.shape[1], self._previous_frame.shape[0])
        try:
            parameters = self.broadcast_camera_filter.parameters(
                self._shot_camera_center, image_size
            )
            pitch_to_image = parameters.pitch_to_image_homography()
            image_to_pitch = parameters.image_to_pitch_homography()
        except (ValueError, np.linalg.LinAlgError):
            return None
        filter_state = self.broadcast_camera_filter.state
        filter_covariance = self.broadcast_camera_filter.covariance
        return CameraState(
            status=CameraTrackingStatus.PREDICTED,
            pan_rad=parameters.pan_rad,
            pan_velocity_rad_s=float(filter_state[4]),
            covariance=filter_covariance[np.ix_([0, 4], [0, 4])],
            image_to_pitch=image_to_pitch,
            pitch_to_image=pitch_to_image,
            confidence=confidence,
            age_since_semantic_update=semantic_age,
            registration_mode=self.registration_mode,
            measurement_tier=MeasurementTier.PREVIEW,
            shot_id=self._shot_id,
            camera_model="broadcast_tripod_pan_tilt_zoom",
            focal_px=parameters.focal_px,
            tilt_rad=parameters.tilt_rad,
            roll_rad=parameters.roll_rad,
            projection_uncertainty=float(np.trace(filter_covariance[:4, :4])),
            camera_center_xyz_m=parameters.camera_center_xyz_m,
            tilt_velocity_rad_s=float(filter_state[5]),
            zoom_velocity_log_s=float(filter_state[6]),
            camera_parameter_covariance=filter_covariance.copy(),
        )

    def _semantic_state(
        self,
        output: PitchPerceptionOutput,
        image_size: tuple[int, int],
    ) -> Optional[CameraState]:
        if self.registration_mode is RegistrationMode.RIG_PAN and self.pan_refiner is not None:
            global_search = self._last_state is None or self._last_state.status is CameraTrackingStatus.LOST
            initial_pan = None if global_search else self.pan_filter.pan_rad
            result = self.pan_refiner.refine(
                output.points,
                output.lines,
                image_size,
                initial_pan_rad=initial_pan,
                global_search=global_search,
            )
            if not result.success or result.pan_rad is None:
                return None
            variance = max(((result.mean_point_error_px or 2.0) / 1000.0) ** 2, 1e-7)
            if global_search:
                self.pan_filter.reset(result.pan_rad)
                self.pan_filter.covariance[0, 0] = variance
                status = CameraTrackingStatus.RELOCALIZED
                self.relocalization_count += 1
            else:
                if not self.pan_filter.update(result.pan_rad, variance):
                    return None
                status = CameraTrackingStatus.CORRECTED
            return self._state_from_pan(
                status,
                result.confidence,
                result.point_inliers,
                result.mean_line_error_px,
                result.p95_line_error_px,
                0,
            )

        initialization = self.initializer.estimate(
            output.points,
            image_size=image_size,
            pitch_size_m=(
                self.pitch_model.dimensions.length_m,
                self.pitch_model.dimensions.width_m,
            ),
        )
        if not initialization.success:
            return None
        status = (
            CameraTrackingStatus.RELOCALIZED
            if self._last_state is None or self._last_state.status is CameraTrackingStatus.LOST
            else CameraTrackingStatus.CORRECTED
        )
        if status is CameraTrackingStatus.RELOCALIZED:
            self.relocalization_count += 1
        confidence = float(
            np.clip(
                0.35
                + 0.35 * initialization.image_coverage
                + 0.30
                * min(len(initialization.inlier_indices) / max(len(output.points), 1), 1.0),
                0.0,
                1.0,
            )
        )
        line_result = None
        if output.lines and initialization.pitch_to_image is not None:
            candidate = self.full_homography_refiner.refine(
                initialization.pitch_to_image,
                output.points,
                output.lines,
                image_size,
            )
            if candidate.success:
                line_result = candidate
                confidence = max(confidence, candidate.confidence)
        safe_line_evidence = (
            line_result is not None
            and line_result.success
            and line_result.visible_segment_count
            >= self.config.minimum_safe_semantic_lines
        )
        measurement_tier = (
            MeasurementTier.SAFE
            if safe_line_evidence or self.config.allow_point_only_safe
            else MeasurementTier.PREVIEW
        )
        return self._broadcast_state_from_initialization(
            initialization,
            image_size,
            status,
            confidence,
            0,
            measurement_tier,
            line_result=line_result,
        )

    def _broadcast_flow_state(
        self,
        flow: OpticalFlowResult,
        observations: tuple[PointObservation, ...],
        image_size: tuple[int, int],
        frame_index: int,
    ) -> Optional[CameraState]:
        if len(observations) < self.config.minimum_flow_tracks:
            return None
        initialization = self.initializer.estimate(
            observations,
            image_size=image_size,
            pitch_size_m=(
                self.pitch_model.dimensions.length_m,
                self.pitch_model.dimensions.width_m,
            ),
        )
        if not initialization.success:
            return None
        semantic_age = (
            frame_index - self._last_semantic_frame
            if self._last_semantic_frame is not None
            else frame_index + 1
        )
        safe_limit = max(
            int(round(self.config.safe_flow_without_semantic_seconds * self.config.fps)),
            1,
        )
        preview_limit = max(
            int(round(self.config.preview_flow_without_semantic_seconds * self.config.fps)),
            safe_limit,
        )
        if semantic_age > preview_limit:
            return None
        inherited_safe = (
            self._last_state is not None
            and self._last_state.measurement_tier is MeasurementTier.SAFE
        )
        tier = (
            MeasurementTier.SAFE
            if semantic_age <= safe_limit and inherited_safe
            else MeasurementTier.PREVIEW
        )
        previous_confidence = self._last_state.confidence if self._last_state else 0.5
        geometric_confidence = float(
            np.clip(
                0.35
                + 0.35 * initialization.image_coverage
                + 0.30
                * min(
                    len(initialization.inlier_indices) / max(len(observations), 1),
                    1.0,
                ),
                0.0,
                1.0,
            )
        )
        confidence = min(previous_confidence * 0.997, geometric_confidence)
        if tier is MeasurementTier.PREVIEW:
            confidence *= 0.85
        self.flow_update_count += 1
        return self._broadcast_state_from_initialization(
            initialization,
            image_size,
            CameraTrackingStatus.TRACKED,
            confidence,
            semantic_age,
            tier,
            flow_inliers=len(initialization.inlier_indices),
        )

    def process(
        self,
        frame: np.ndarray,
        frame_index: int,
        person_masks_or_boxes: object | None = None,
    ) -> FieldRegistrationFrame:
        if frame.ndim < 2:
            raise ValueError("frame must have at least two dimensions")
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("frame_index must increase monotonically")
        height, width = frame.shape[:2]
        dynamic_boxes = None
        if person_masks_or_boxes is not None:
            candidate = np.asarray(person_masks_or_boxes, dtype=np.float64)
            if candidate.size:
                dynamic_boxes = candidate.reshape(-1, 4)

        shot = self.shot_detector.update(frame)
        if shot.is_cut:
            self._shot_id += 1
            self.shot_cut_count += 1
            self._reset_tracking_state()

        if self.registration_mode is RegistrationMode.RIG_PAN:
            self.pan_filter.predict(1.0 / self.config.fps)
        elif (
            self.registration_mode is RegistrationMode.BROADCAST
            and self.broadcast_camera_filter.initialized
        ):
            self.broadcast_camera_filter.predict(1.0 / self.config.fps)
        flow, flow_observations = self._flow_result(frame)
        state: Optional[CameraState] = None
        output = PitchPerceptionOutput()
        refresh_reason: Optional[str] = None
        semantic_due = self._should_run_semantic(frame_index)
        if semantic_due:
            output = self.perception_backend.predict(frame)
            self.semantic_inference_count += 1
            self._last_semantic_frame = frame_index
            state = self._semantic_state(output, (width, height))
            refresh_reason = (
                "hard_cut"
                if shot.is_cut
                else "scheduled" if self._last_state is not None else "initialization"
            )

        if (
            state is None
            and self.rig_profile is not None
            and self.pan_refiner is not None
            and len(flow_observations) >= max(
                self.config.minimum_flow_tracks,
                self.pan_refiner.config.minimum_point_inliers,
            )
        ):
            flow_refinement = self.pan_refiner.refine(
                flow_observations,
                (),
                (width, height),
                initial_pan_rad=self.pan_filter.pan_rad,
                global_search=False,
            )
            if flow_refinement.success and flow_refinement.pan_rad is not None:
                variance = max(
                    ((flow_refinement.mean_point_error_px or 2.0) / 900.0) ** 2,
                    2e-7,
                )
                if self.pan_filter.update(flow_refinement.pan_rad, variance):
                    previous_confidence = self._last_state.confidence if self._last_state else 0.5
                    state = self._state_from_pan(
                        CameraTrackingStatus.TRACKED,
                        min(previous_confidence * 0.995, flow_refinement.confidence),
                        flow_refinement.point_inliers,
                        flow.mean_error_px,
                        None,
                        (
                            frame_index - self._last_semantic_frame
                            if self._last_semantic_frame is not None
                            else 0
                        ),
                    )
                    self.flow_update_count += 1

        if (
            state is None
            and self.registration_mode is RegistrationMode.BROADCAST
        ):
            state = self._broadcast_flow_state(
                flow,
                flow_observations,
                (width, height),
                frame_index,
            )

        if state is None:
            self._prediction_age += 1
            if (
                self._last_state is not None
                and self._last_state.image_to_pitch is not None
                and self._prediction_age <= self.config.max_prediction_frames
            ):
                confidence = self._last_state.confidence * (0.82**self._prediction_age)
                if self.registration_mode is RegistrationMode.RIG_PAN:
                    state = self._state_from_pan(
                        CameraTrackingStatus.PREDICTED,
                        confidence,
                        0,
                        None,
                        None,
                        (
                            frame_index - self._last_semantic_frame
                            if self._last_semantic_frame is not None
                            else self._prediction_age
                        ),
                        MeasurementTier.PREVIEW,
                    )
                else:
                    semantic_age = (
                        frame_index - self._last_semantic_frame
                        if self._last_semantic_frame is not None
                        else self._prediction_age
                    )
                    state = self._broadcast_prediction_state(
                        confidence, semantic_age
                    ) or CameraState(
                        status=CameraTrackingStatus.PREDICTED,
                        pan_rad=self._last_state.pan_rad,
                        pan_velocity_rad_s=0.0,
                        covariance=self._last_state.covariance * 1.5,
                        image_to_pitch=self._last_state.image_to_pitch,
                        pitch_to_image=self._last_state.pitch_to_image,
                        confidence=confidence,
                        age_since_semantic_update=semantic_age,
                        registration_mode=self.registration_mode,
                        measurement_tier=MeasurementTier.PREVIEW,
                        shot_id=self._shot_id,
                        camera_model="broadcast_planar_homography",
                        projection_uncertainty=float(
                            (self._last_state.projection_uncertainty or 1.0)
                            * (1.5**self._prediction_age)
                        ),
                    )
                self.prediction_count += 1
            else:
                state = CameraState(
                    status=CameraTrackingStatus.LOST,
                    pan_rad=(
                        self.pan_filter.pan_rad
                        if self.registration_mode is RegistrationMode.RIG_PAN
                        else float("nan")
                    ),
                    pan_velocity_rad_s=(
                        self.pan_filter.velocity_rad_s
                        if self.registration_mode is RegistrationMode.RIG_PAN
                        else 0.0
                    ),
                    covariance=(
                        self.pan_filter.covariance.copy()
                        if self.registration_mode is RegistrationMode.RIG_PAN
                        else np.diag([1e6, 1e6])
                    ),
                    image_to_pitch=None,
                    pitch_to_image=None,
                    confidence=0.0,
                    age_since_semantic_update=self._prediction_age,
                    registration_mode=self.registration_mode,
                    measurement_tier=MeasurementTier.UNAVAILABLE,
                    shot_id=self._shot_id,
                    camera_model=(
                        "fixed_rig_pan"
                        if self.registration_mode is RegistrationMode.RIG_PAN
                        else "broadcast_planar_homography"
                    ),
                )
                self.lost_count += 1
        else:
            self._prediction_age = 0

        self._last_state = state
        self._previous_frame = frame.copy()
        self._previous_dynamic_boxes = None if dynamic_boxes is None else dynamic_boxes.copy()
        self._last_frame_index = frame_index
        return FieldRegistrationFrame(
            frame_index=frame_index,
            camera_state=state,
            point_observations=output.points,
            line_observations=output.lines,
            refresh_reason=refresh_reason,
            diagnostics={
                "flow_track_count": flow.count,
                "flow_valid_ratio": flow.valid_ratio,
                "semantic_interval": self._semantic_interval(),
                "semantic_inference_count": self.semantic_inference_count,
                "semantic_inference_time_ms": output.inference_time_ms,
                "registration_mode": self.registration_mode.value,
                "measurement_tier": state.measurement_tier.value,
                "shot_id": self._shot_id,
                "shot_cut": int(shot.is_cut),
                "shot_cut_score": shot.score,
                "shot_histogram_distance": shot.histogram_distance,
                "field_view": shot.field_view.value,
                "green_ratio": shot.green_ratio,
            },
        )
