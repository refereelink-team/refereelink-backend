"""Constrained pitch tracker state machine and adaptive refresh scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.field_registration.camera_model import CameraRigProfile
from app.field_registration.geometry import transform_points
from app.field_registration.initializer import HomographyInitializer
from app.field_registration.optical_flow import MaskedSparseOpticalFlow, OpticalFlowResult
from app.field_registration.pan_filter import PanExtendedKalmanFilter
from app.field_registration.perception import PitchPerceptionBackend, PitchPerceptionOutput
from app.field_registration.pitch_model import PitchModel
from app.field_registration.point_line_refiner import PanOnlyPointLineRefiner
from app.field_registration.types import (
    CameraState,
    CameraTrackingStatus,
    FieldRegistrationFrame,
    PointObservation,
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
    ) -> None:
        self.pitch_model = pitch_model
        self.perception_backend = perception_backend
        self.rig_profile = rig_profile
        self.config = config or FieldRegistrationConfig()
        if self.config.fps <= 0:
            raise ValueError("registration fps must be positive")
        self.initializer = initializer or HomographyInitializer()
        self.optical_flow = optical_flow or MaskedSparseOpticalFlow()
        self.pan_filter = pan_filter or PanExtendedKalmanFilter()
        self.pan_refiner = (
            PanOnlyPointLineRefiner(rig_profile) if rig_profile is not None else None
        )
        self._previous_frame: Optional[np.ndarray] = None
        self._previous_dynamic_boxes: Optional[np.ndarray] = None
        self._last_state: Optional[CameraState] = None
        self._last_semantic_frame: Optional[int] = None
        self._prediction_age = 0
        self._last_frame_index: Optional[int] = None
        self.semantic_inference_count = 0
        self.flow_update_count = 0
        self.prediction_count = 0
        self.lost_count = 0
        self.relocalization_count = 0

    @property
    def state(self) -> Optional[CameraState]:
        return self._last_state

    def reset(self) -> None:
        self._previous_frame = None
        self._previous_dynamic_boxes = None
        self._last_state = None
        self._last_semantic_frame = None
        self._prediction_age = 0
        self._last_frame_index = None
        self.pan_filter = PanExtendedKalmanFilter(self.pan_filter.config)

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
            return 1
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
        return projected

    def _flow_result(
        self,
        frame: np.ndarray,
    ) -> tuple[OpticalFlowResult, tuple[PointObservation, ...]]:
        empty = OpticalFlowResult(
            np.empty((0, 2)), np.empty((0, 2)), np.empty(0), 0.0
        )
        if (
            self.rig_profile is None
            or self.pan_refiner is None
            or self._previous_frame is None
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
        )

    def _semantic_state(
        self,
        output: PitchPerceptionOutput,
        image_size: tuple[int, int],
    ) -> Optional[CameraState]:
        if self.rig_profile is not None and self.pan_refiner is not None:
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
        return CameraState(
            status=status,
            pan_rad=float("nan"),
            pan_velocity_rad_s=0.0,
            covariance=np.diag([1e6, 1e6]),
            image_to_pitch=initialization.image_to_pitch,
            pitch_to_image=initialization.pitch_to_image,
            point_inliers=len(initialization.inlier_indices),
            spatial_coverage=initialization.image_coverage,
            confidence=confidence,
            age_since_semantic_update=0,
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

        if self.rig_profile is not None:
            self.pan_filter.predict(1.0 / self.config.fps)
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
            refresh_reason = "scheduled" if self._last_state is not None else "initialization"

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

        if state is None:
            self._prediction_age += 1
            if (
                self._last_state is not None
                and self._last_state.image_to_pitch is not None
                and self._prediction_age <= self.config.max_prediction_frames
            ):
                confidence = self._last_state.confidence * (0.82**self._prediction_age)
                if self.rig_profile is not None:
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
                    )
                else:
                    state = CameraState(
                        status=CameraTrackingStatus.PREDICTED,
                        pan_rad=self._last_state.pan_rad,
                        pan_velocity_rad_s=0.0,
                        covariance=self._last_state.covariance * 1.5,
                        image_to_pitch=self._last_state.image_to_pitch,
                        pitch_to_image=self._last_state.pitch_to_image,
                        confidence=confidence,
                        age_since_semantic_update=self._prediction_age,
                    )
                self.prediction_count += 1
            else:
                state = CameraState(
                    status=CameraTrackingStatus.LOST,
                    pan_rad=self.pan_filter.pan_rad if self.rig_profile is not None else float("nan"),
                    pan_velocity_rad_s=(
                        self.pan_filter.velocity_rad_s if self.rig_profile is not None else 0.0
                    ),
                    covariance=(
                        self.pan_filter.covariance.copy()
                        if self.rig_profile is not None
                        else np.diag([1e6, 1e6])
                    ),
                    image_to_pitch=None,
                    pitch_to_image=None,
                    confidence=0.0,
                    age_since_semantic_update=self._prediction_age,
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
            },
        )
