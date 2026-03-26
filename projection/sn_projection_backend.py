from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from projection.calibrator import LineFallbackDetector, PitchKeypointDetector
from projection.dynamic_projector import create_dynamic_projector
from projection.homography import (
    COORD_SYSTEM_METER_CENTER,
    HomographyAdapter,
    HomographyEstimator,
    HomographyQuality,
    HomographyQualityGate,
    HomographySmoother,
    HomographyState,
    build_default_homography,
    build_default_meter_homography,
    load_calibration,
    template_to_field_points_center,
)
from projection.keypoint_manager import KeypointManager
from projection.shot_change_detector import ShotChangeDetector


class SNCalibrationProjector:
    """
    Lightweight clean-room implementation inspired by SN-style calibration flow:
    keypoint/line evidence -> robust homography estimation -> temporal fallback.
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

        # Stage-1 evidence
        self.keypoint_detector = PitchKeypointDetector()
        self.line_fallback_detector = LineFallbackDetector()
        self.keypoint_manager = KeypointManager(
            max_age=45 if backend_name == "pnl" else 30,
            min_confidence=0.25 if backend_name == "pnl" else 0.3,
        )
        self.template_pts = template_to_field_points_center()

        # Stage-2 calibration
        self.quality_gate = HomographyQualityGate(
            min_inliers=5 if backend_name == "pnl" else 4,
            min_inlier_ratio=0.40 if backend_name == "pnl" else 0.35,
            max_reproj_err_px=16.0 if backend_name == "pnl" else 20.0,
        )
        self.smoother = HomographySmoother(alpha=0.55 if backend_name == "pnl" else 0.6)

        # Runtime stability
        self.shot_detector = ShotChangeDetector(
            hist_threshold=0.28 if backend_name == "pnl" else 0.30,
            pixel_threshold=0.40,
            min_changed_pixels=10000,
        )
        self.h_state = HomographyState.default()
        self.h_state.H = build_default_meter_homography().get_matrix()
        self.default_homography = build_default_meter_homography()
        self.last_good_H: Optional[np.ndarray] = None
        self.prev_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        self.last_detect_frame = 0
        self.validated = False

    def update(self, frame: np.ndarray) -> HomographyAdapter:
        self.frame_count += 1

        if self.shot_detector.update(frame):
            self._handle_shot_change()

        if self._should_detect():
            self._run_detection(frame)
            self.last_detect_frame = self.frame_count
        elif self.prev_frame is not None:
            self.keypoint_manager.track(self.prev_frame, frame)
            self._estimate_from_keypoints()

        self.prev_frame = frame.copy()

        if self.validated and self.h_state.H is not None:
            H_smooth = self.smoother.update(self.h_state.H)
            return HomographyAdapter(H_matrix=H_smooth, coord_system=COORD_SYSTEM_METER_CENTER)

        if self.use_prev_homography and self.last_good_H is not None:
            return HomographyAdapter(H_matrix=self.last_good_H, coord_system=COORD_SYSTEM_METER_CENTER)

        return self.default_homography

    def _should_detect(self) -> bool:
        if not self.validated:
            return True
        if (self.frame_count - self.last_detect_frame) >= self.recalib_interval:
            return True
        if self.keypoint_manager.get_confident_count() < 4:
            return True
        if self.h_state.fails >= 10:
            return True
        return False

    def _run_detection(self, frame: np.ndarray) -> None:
        detections = self.keypoint_detector.detect_labeled_keypoints(frame)
        if detections:
            self.keypoint_manager.fuse_detected_keypoints(detections)

        if self.keypoint_manager.get_confident_count() < 4:
            fallback = self.line_fallback_detector.detect_keypoints(frame)
            if fallback:
                self.keypoint_manager.fuse_detected_keypoints(fallback)

        self._estimate_from_keypoints()

    def _estimate_from_keypoints(self) -> None:
        src_pts, dst_pts, _ = self.keypoint_manager.build_correspondences(self.template_pts)
        if len(src_pts) < 4:
            self._handle_failure()
            return

        candidate = self._estimate_with_voting(src_pts, dst_pts)
        if candidate is None:
            self._handle_failure()
            return

        H_candidate, quality = candidate
        if not self.quality_gate.accept(quality):
            self._handle_failure()
            return

        self.h_state.H = H_candidate
        self.h_state.inliers = quality.inliers
        self.h_state.inlier_ratio = quality.inlier_ratio
        self.h_state.reproj_err_px_med = quality.reproj_err_px_med
        self.h_state.reproj_err_px_p90 = quality.reproj_err_px_p90
        self.h_state.fails = 0
        self.h_state.mode = "TRACK"
        self.h_state.last_good_ts = time.time()
        self.last_good_H = H_candidate.copy()
        self.validated = True

    def _estimate_with_voting(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
    ) -> Optional[tuple[np.ndarray, HomographyQuality]]:
        # PnL-style path favors tighter thresholds first.
        thresholds = [5.0, 3.0, 10.0, 15.0, 25.0, 50.0] if self.backend_name == "pnl" else [10.0, 5.0, 3.0, 15.0, 25.0, 50.0]

        best_H: Optional[np.ndarray] = None
        best_q: Optional[HomographyQuality] = None
        for thresh in thresholds:
            estimator = HomographyEstimator(
                ransac_reproj_threshold=thresh,
                max_iters=1000,
                confidence=0.995,
            )
            H, _, quality = estimator.estimate(src_pts, dst_pts)
            if H is None or quality is None:
                continue
            if best_q is None or quality.reproj_err_px_p90 < best_q.reproj_err_px_p90:
                best_H = H
                best_q = quality

        if best_H is None or best_q is None:
            return None
        return best_H, best_q

    def _handle_failure(self) -> None:
        self.h_state.fails += 1
        self.h_state.mode = "FROZEN"
        if self.h_state.fails >= 10:
            self.validated = False
            self.h_state.mode = "DETECT"
            self.smoother.reset()

    def _handle_shot_change(self) -> None:
        self.keypoint_manager.clear()
        self.prev_frame = None
        self.validated = False
        self.h_state = HomographyState.default()
        self.h_state.H = self.default_homography.get_matrix()
        self.h_state.mode = "DETECT"
        self.h_state.fails = 0
        self.smoother.reset()


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

        if calib_backend in {"nbjw", "pnl"}:
            self.sn_projector = SNCalibrationProjector(
                backend_name=calib_backend,
                recalib_interval=recalib_interval,
                use_prev_homography=use_prev_homography,
                debug=debug,
            )
            return

        # legacy path
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
        elif calibration_path:
            self.static_h = load_calibration(calibration_path)
        else:
            self.static_h = build_default_homography()

    def update(self, frame: np.ndarray) -> HomographyAdapter:
        if self.sn_projector is not None:
            return self.sn_projector.update(frame)
        if self.dynamic_projector is not None:
            return self.dynamic_projector.update(frame)
        if self.static_h is None:
            self.static_h = build_default_homography()
        return self.static_h

    def get_keypoints(self) -> Dict:
        if self.sn_projector is not None:
            return self.sn_projector.keypoint_manager.keypoints
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

