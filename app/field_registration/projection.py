"""Safe image/pitch projection and first-order uncertainty propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from app.field_registration.camera_model import CameraRigProfile
from app.field_registration.geometry import transform_points
from app.field_registration.pitch_model import PitchModel
from app.field_registration.types import CameraState, PitchCoordinate


@dataclass(frozen=True)
class ProjectionUncertaintyConfig:
    finite_difference_px: float = 0.5
    finite_difference_pan_rad: float = 1e-4
    max_sigma_m: float = 3.0
    pitch_margin_m: float = 0.5


class PitchProjector:
    def __init__(
        self,
        pitch_model: PitchModel,
        rig_profile: Optional[CameraRigProfile] = None,
        config: ProjectionUncertaintyConfig | None = None,
    ) -> None:
        self.pitch_model = pitch_model
        self.rig_profile = rig_profile
        self.config = config or ProjectionUncertaintyConfig()

    def image_to_pitch(
        self,
        image_xy: Tuple[float, float],
        camera_state: CameraState,
        source: str = "bbox_bottom",
        image_covariance_px2: Optional[np.ndarray] = None,
    ) -> PitchCoordinate:
        if camera_state.image_to_pitch is None:
            return PitchCoordinate(None, None, "none", camera_state.status)
        projected = transform_points(
            np.asarray([image_xy], dtype=np.float64), camera_state.image_to_pitch
        )[0]
        if not np.all(np.isfinite(projected)) or not self.pitch_model.contains(
            (float(projected[0]), float(projected[1])),
            margin_m=self.config.pitch_margin_m,
        ):
            return PitchCoordinate(None, None, "none", camera_state.status)

        covariance = self._propagate_covariance(
            image_xy,
            camera_state,
            image_covariance_px2,
        )
        sigma = None
        if covariance is not None:
            eigenvalues = np.linalg.eigvalsh(covariance)
            sigma = float(np.sqrt(max(float(np.max(eigenvalues)), 0.0)))
            if not np.isfinite(sigma) or sigma > self.config.max_sigma_m:
                return PitchCoordinate(None, sigma, "none", camera_state.status, covariance)
        return PitchCoordinate(
            xy_m=(float(projected[0]), float(projected[1])),
            sigma_m=sigma,
            source=source,  # type: ignore[arg-type]
            camera_status=camera_state.status,
            covariance_m2=covariance,
        )

    def _propagate_covariance(
        self,
        image_xy: Tuple[float, float],
        camera_state: CameraState,
        image_covariance_px2: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        covariance_px = (
            np.eye(2, dtype=np.float64)
            if image_covariance_px2 is None
            else np.asarray(image_covariance_px2, dtype=np.float64)
        )
        if covariance_px.shape != (2, 2) or not np.all(np.isfinite(covariance_px)):
            raise ValueError("image covariance must be a finite 2x2 matrix")
        step = self.config.finite_difference_px
        base = np.asarray(image_xy, dtype=np.float64)
        plus_x = transform_points(
            np.asarray([base + (step, 0.0)]), camera_state.image_to_pitch
        )[0]
        minus_x = transform_points(
            np.asarray([base - (step, 0.0)]), camera_state.image_to_pitch
        )[0]
        plus_y = transform_points(
            np.asarray([base + (0.0, step)]), camera_state.image_to_pitch
        )[0]
        minus_y = transform_points(
            np.asarray([base - (0.0, step)]), camera_state.image_to_pitch
        )[0]
        jacobian_image = np.column_stack(
            ((plus_x - minus_x) / (2.0 * step), (plus_y - minus_y) / (2.0 * step))
        )
        covariance = jacobian_image @ covariance_px @ jacobian_image.T

        if self.rig_profile is not None and np.isfinite(camera_state.pan_rad):
            pan_step = self.config.finite_difference_pan_rad
            plus_h = self.rig_profile.image_to_pitch_homography(
                camera_state.pan_rad + pan_step
            )
            minus_h = self.rig_profile.image_to_pitch_homography(
                camera_state.pan_rad - pan_step
            )
            plus_pan = transform_points(np.asarray([base]), plus_h)[0]
            minus_pan = transform_points(np.asarray([base]), minus_h)[0]
            jacobian_pan = ((plus_pan - minus_pan) / (2.0 * pan_step)).reshape(2, 1)
            pan_variance = max(float(camera_state.covariance[0, 0]), 0.0)
            covariance += jacobian_pan * pan_variance @ jacobian_pan.T
        return (covariance + covariance.T) / 2.0
