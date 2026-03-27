# -*- coding: utf-8 -*-
"""
SN-style calibration projector using the new calibration module.

Uses SNCalibrationEngine (DeepLabV3 segmentation + camera pose estimation)
instead of the old YOLO keypoint-based approach.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

from calibration import SNCalibrationEngine
from projection.dynamic_projector import create_dynamic_projector
from projection.homography import (
    COORD_SYSTEM_METER_CENTER,
    HomographyAdapter,
    build_default_meter_homography,
    load_calibration,
)

# Default path to SN segmentation model
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _TrackedKeypointLite:
    """Lightweight TrackedKeypoint-like object for draw_keypoints_on_frame compatibility."""
    __slots__ = ('pt', 'visible')

    def __init__(self, pt, visible=True):
        self.pt = pt  # (x, y) in pixels
        self.visible = visible


class SNCalibrationProjector:
    """
    Wraps SNCalibrationEngine with the HomographyAdapter interface.

    Uses semantic segmentation (DeepLabV3) to detect pitch line extremities,
    matches them to 3D pitch geometry to estimate homography, and exposes
    a HomographyAdapter for use by projection/modeling.py.
    """

    def __init__(
        self,
        backend_name: str = "nbjw",
        recalib_interval: int = 10,
        use_prev_homography: bool = True,
        debug: bool = False,
    ):
        if backend_name not in {"nbjw", "pnl"}:
            raise ValueError(f"Unsupported backend: {backend_name}")

        self.backend_name = backend_name
        self.recalib_interval = max(1, int(recalib_interval))
        self.use_prev_homography = bool(use_prev_homography)
        self.debug = debug

        # Build paths to SN calibration resources
        model_path = os.path.join(
            PROJECT_ROOT, "sn-calibration-main", "resources", "soccer_pitch_segmentation.pth"
        )
        mean_path = os.path.join(PROJECT_ROOT, "sn-calibration-main", "resources", "mean.npy")
        std_path = os.path.join(PROJECT_ROOT, "sn-calibration-main", "resources", "std.npy")

        # SN calibration engine
        self.engine = SNCalibrationEngine(
            model_path=model_path,
            mean_path=mean_path,
            std_path=std_path,
            recalib_interval=recalib_interval,
            debug=debug,
        )

        self.default_homography = build_default_meter_homography()
        self.last_good_H: Optional[np.ndarray] = None
        self.validated = False

    def update(self, frame: np.ndarray) -> HomographyAdapter:
        """Process one frame and return HomographyAdapter."""
        camera = self.engine.update(frame)
        self.validated = self.engine.validated

        # Get homography from engine (H: world meters → image pixels)
        H = self.engine.get_homography()
        if H is not None:
            self.last_good_H = H.copy()
            return HomographyAdapter(
                H_matrix=H,
                coord_system=COORD_SYSTEM_METER_CENTER,
            )

        if self.use_prev_homography and self.last_good_H is not None:
            return HomographyAdapter(
                H_matrix=self.last_good_H,
                coord_system=COORD_SYSTEM_METER_CENTER,
            )

        return self.default_homography

    @property
    def keypoint_manager(self):
        """Dummy keypoint_manager for interface compatibility."""
        return self.engine


class ProjectionEngine:
    """
    Unified runtime projection engine with switchable calibration backends.
    """

    def __init__(
        self,
        calib_backend: str = "nbjw",
        dynamic: bool = False,
        recalib_interval: int = 10,
        calibration_path: str = "",
        field_path: str = "field_map.png",
        debug: bool = False,
        use_prev_homography: bool = True,
    ):
        self.calib_backend = calib_backend
        self.debug = debug
        self.dynamic_projector = None
        self.sn_projector = None
        self.static_h = None

        # Static calibration takes priority
        if calibration_path:
            self.static_h = load_calibration(calibration_path)
            self._use_static = True
            print(f"[calibration] Loaded static homography from {calibration_path}")
            return
        self._use_static = False

        if calib_backend in {"nbjw", "pnl"}:
            self.sn_projector = SNCalibrationProjector(
                backend_name=calib_backend,
                recalib_interval=recalib_interval,
                use_prev_homography=use_prev_homography,
                debug=debug,
            )
            return

        # Legacy path
        if dynamic:
            self.dynamic_projector = create_dynamic_projector(
                recalib_interval=recalib_interval,
                field_path=field_path,
                debug=debug,
                min_inliers=4,
                min_inlier_ratio=0.35,
                max_reproj_err=20.0,
                max_consecutive_fails=10,
            )
        else:
            self.static_h = build_default_meter_homography()

    def update(self, frame) -> HomographyAdapter:
        if self.sn_projector is not None:
            return self.sn_projector.update(frame)
        if self.dynamic_projector is not None:
            return self.dynamic_projector.update(frame)
        if self.static_h is None:
            self.static_h = build_default_meter_homography()
        return self.static_h

    def get_keypoints(self) -> Dict:
        if self.sn_projector is not None:
            km = self.sn_projector.keypoint_manager
            if hasattr(km, "get_extremities_2d"):
                # Convert extremities {class: [{'x': norm, 'y': norm}, ...]} to
                # TrackedKeypoint-like {class: TrackedKeypoint(pt, visible)} for
                # compatibility with draw_keypoints_on_frame
                extremities = km.get_extremities_2d()
                if not extremities:
                    return {}
                # Use image dimensions from engine
                cam = km.camera
                w, h = cam.image_width, cam.image_height
                result = {}
                for class_name, pts in extremities.items():
                    if len(pts) >= 2:
                        # Take first endpoint as the keypoint position
                        pt = pts[0]
                        result[class_name] = _TrackedKeypointLite(
                            pt=(float(pt['x']) * w, float(pt['y']) * h),
                            visible=True,
                        )
                return result
            return {}
        if self.dynamic_projector is not None:
            return self.dynamic_projector.keypoint_manager.keypoints
        return {}


def create_projection_engine(
    calib_backend: str = "nbjw",
    dynamic: bool = False,
    recalib_interval: int = 10,
    calibration_path: str = "",
    field_path: str = "field_map.png",
    debug: bool = False,
    use_prev_homography: bool = True,
) -> ProjectionEngine:
    return ProjectionEngine(
        calib_backend=calib_backend,
        dynamic=dynamic,
        recalib_interval=recalib_interval,
        calibration_path=calibration_path,
        field_path=field_path,
        debug=debug,
        use_prev_homography=use_prev_homography,
    )
