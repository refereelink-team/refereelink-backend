# -*- coding: utf-8 -*-
"""
SN-style field calibration engine.

Uses semantic segmentation (DeepLabV3) to detect pitch line extremities,
matches them to 3D pitch geometry to estimate homography, then derives
camera pose from the homography.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from calibration.camera import Camera
from calibration.detect import (
    SegmentationNetwork,
    generate_class_synthesis,
    get_line_extremities,
)
from calibration.pitch import SoccerPitch

# Default paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SNCalibrationEngine:
    """
    Per-frame field calibration using SN-style segmentation + homography.

    Flow per frame:
    1. If detect frame: run segmentation → extract extremities → match to pitch → estimate homography → update camera
    2. Else: track extremities with optical flow → update homography incrementally
    """

    def __init__(
        self,
        model_path: str = "",
        mean_path: str = "",
        std_path: str = "",
        recalib_interval: int = 10,
        debug: bool = False,
        min_inliers: int = 4,
        min_inlier_ratio: float = 0.35,
        max_reproj_err: float = 20.0,
    ):
        """
        Args:
            model_path: path to soccer_pitch_segmentation.pth
            mean_path: path to mean.npy
            std_path: path to std.npy
            recalib_interval: run full detection every N frames
            debug: enable debug output
            min_inliers: minimum RANSAC inliers for homography acceptance
            min_inlier_ratio: minimum ratio of inliers / total points
            max_reproj_err: max median reprojection error (pixels)
        """
        self.debug = debug

        # Resolve resource paths
        if not model_path:
            model_path = os.path.join(PROJECT_ROOT, "sn-calibration-main", "resources", "soccer_pitch_segmentation.pth")
        if not mean_path:
            mean_path = os.path.join(PROJECT_ROOT, "sn-calibration-main", "resources", "mean.npy")
        if not std_path:
            std_path = os.path.join(PROJECT_ROOT, "sn-calibration-main", "resources", "std.npy")

        self.model_path = model_path
        self.mean_path = mean_path
        self.std_path = std_path
        self.recalib_interval = max(1, recalib_interval)
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.max_reproj_err = max_reproj_err

        # Segmentation model
        self.segmenter: Optional[SegmentationNetwork] = None
        self._segmenter_loaded = False

        # Pitch geometry
        self.pitch = SoccerPitch()

        # Camera pose
        self.camera = Camera()

        # Runtime state
        self.frame_count = 0
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_extremities_2d: Dict[str, List[Dict[str, float]]] = {}
        self.prev_pix_pts: Optional[np.ndarray] = None  # tracked 2D points for optical flow
        self.prev_pix_pts_3d: Dict[str, List[np.ndarray]] = {}  # corresponding 3D points
        self.last_good_H: Optional[np.ndarray] = None
        self.validated = False

        # Optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def _ensure_segmenter(self):
        """Lazy-load segmentation model."""
        if self._segmenter_loaded:
            return
        self._segmenter_loaded = True
        if not os.path.exists(self.model_path):
            if self.debug:
                print(f"[SNCalibrationEngine] Model not found: {self.model_path}, segmentation disabled")
            return
        try:
            self.segmenter = SegmentationNetwork(
                self.model_path, self.mean_path, self.std_path
            )
            if self.debug:
                print("[SNCalibrationEngine] SegmentationNetwork loaded")
        except Exception as e:
            print(f"[SNCalibrationEngine] Failed to load SegmentationNetwork: {e}")
            self.segmenter = None

    def update(self, frame: np.ndarray) -> Camera:
        """
        Process one frame and update camera pose.

        Args:
            frame: BGR image

        Returns:
            Camera with updated pose
        """
        self.frame_count += 1
        h, w = frame.shape[:2]
        if self.camera.image_width == 0:
            self.camera = Camera(iwidth=w, iheight=h)

        if self._should_detect():
            self._run_detection(frame)
        else:
            self._track_extremities(frame)

        self.prev_frame = frame.copy()
        return self.camera

    def _should_detect(self) -> bool:
        """Return True if we should run full segmentation detection this frame."""
        if not self.validated:
            return True
        if (self.frame_count - 1) % self.recalib_interval == 0:
            return True
        return False

    def _run_detection(self, frame: np.ndarray) -> None:
        """Run full segmentation + extremity detection + homography estimation."""
        self._ensure_segmenter()
        if self.segmenter is None:
            return

        h, w = frame.shape[:2]

        # 1. Segment the frame
        semantic_mask = self.segmenter.analyse_image(frame)

        # 2. Generate per-class circle synthesis
        buckets = generate_class_synthesis(semantic_mask, radius=6)

        # 3. Extract extremities (endpoints) for each line
        extremities_2d = get_line_extremities(buckets, 40, w, h)

        if not extremities_2d:
            if self.debug:
                print("[SNCalibrationEngine] No extremities detected")
            return

        self.prev_extremities_2d = extremities_2d

        # 4. Match 2D extremities to 3D pitch geometry → homography
        H, inliers = self._match_to_pitch(extremities_2d, w, h)

        if H is None:
            if self.debug:
                print("[SNCalibrationEngine] Homography estimation failed")
            return

        self.last_good_H = H.copy()

        # 5. Estimate camera pose from homography
        if self.camera.from_homography(H):
            self.validated = True
            if self.debug:
                print(f"[SNCalibrationEngine] Camera validated, inliers={inliers.sum() if inliers is not None else 0}")
        else:
            if self.debug:
                print("[SNCalibrationEngine] Camera.from_homography failed")

    def _match_to_pitch(
        self,
        extremities_2d: Dict[str, List[Dict[str, float]]],
        image_w: int,
        image_h: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Build 2D↔3D correspondences from detected extremities and estimate homography.

        Args:
            extremities_2d: {class_name: [{'x': norm, 'y': norm}, ...]}
            image_w, image_h: image dimensions

        Returns:
            (H, inlier_mask) or (None, None) if insufficient correspondences
        """
        img_pts = []
        world_pts = []

        for class_name, pts in extremities_2d.items():
            if len(pts) < 2:
                continue
            if class_name not in self.pitch.line_extremities:
                continue

            start_3D, end_3D = self.pitch.line_extremities[class_name]

            # Image coordinates (normalized → pixel)
            pt0 = pts[0]
            pt1 = pts[1]
            u0, v0 = pt0['x'] * image_w, pt0['y'] * image_h
            u1, v1 = pt1['x'] * image_w, pt1['y'] * image_h

            img_pts.append([u0, v0])
            world_pts.append([start_3D[0], start_3D[1]])
            img_pts.append([u1, v1])
            world_pts.append([end_3D[0], end_3D[1]])

        if len(img_pts) < 4:
            return None, None

        img_pts = np.array(img_pts, dtype=np.float32)
        world_pts = np.array(world_pts, dtype=np.float32)

        # Add Z=0 for homography (pitch plane)
        world_pts_hom = np.column_stack([world_pts, np.zeros(len(world_pts))]).astype(np.float32)

        # RANSAC homography
        H, mask = cv2.findHomography(world_pts_hom, img_pts, cv2.RANSAC, 5.0, confidence=0.995)

        if H is None:
            return None, None

        # Check quality
        if mask is not None:
            inliers = mask.flatten() == 1
            inlier_count = int(np.sum(inliers))
            inlier_ratio = inlier_count / len(img_pts) if len(img_pts) > 0 else 0.0
        else:
            inliers = np.ones(len(img_pts), dtype=bool)
            inlier_count = len(img_pts)
            inlier_ratio = 1.0

        # Compute median reprojection error
        if inlier_count >= 4:
            # Project world points through H (3x3 homography acts as 2D transform on Z=0 plane)
            src_2d = world_pts_hom[inliers][:, :2]  # (N, 2) - ignore Z=0
            dst_2d = img_pts[inliers]
            # Apply homography: projected = H @ [x; y; 1]
            ones = np.ones((src_2d.shape[0], 1), dtype=np.float32)
            src_h = np.hstack([src_2d, ones])  # (N, 3)
            projected_h = (H @ src_h.T).T  # (N, 3)
            projected = projected_h[:, :2] / projected_h[:, 2:3]  # (N, 2) - divide by homogeneous coord
            errors = np.linalg.norm(projected - dst_2d, axis=1)
            median_err = float(np.median(errors))
        else:
            median_err = float('inf')

        if inlier_count < self.min_inliers or inlier_ratio < self.min_inlier_ratio:
            if self.debug:
                print(f"[SNCalibrationEngine] Quality rejected: inliers={inlier_count}, ratio={inlier_ratio:.2f}")
            return None, None

        if median_err > self.max_reproj_err:
            if self.debug:
                print(f"[SNCalibrationEngine] Quality rejected: median_err={median_err:.1f}")
            return None, None

        return H, (mask.flatten() == 1 if mask is not None else None)

    def _track_extremities(self, frame: np.ndarray) -> None:
        """Track extremity points using KLT optical flow between detection frames."""
        if self.prev_frame is None or not self.prev_extremities_2d:
            return

        # Build 2D points + corresponding 3D points from previous extremities
        prev_pts_2d = []
        corresponding_3d: List[np.ndarray] = []

        for class_name, pts in self.prev_extremities_2d.items():
            if len(pts) < 2 or class_name not in self.pitch.line_extremities:
                continue
            start_3D, end_3D = self.pitch.line_extremities[class_name]
            h, w = self.prev_frame.shape[:2]
            u0, v0 = pts[0]['x'] * w, pts[0]['y'] * h
            u1, v1 = pts[1]['x'] * w, pts[1]['y'] * h
            prev_pts_2d.append([u0, v0])
            corresponding_3d.append(np.array([start_3D[0], start_3D[1], 0.0], dtype=np.float32))
            prev_pts_2d.append([u1, v1])
            corresponding_3d.append(np.array([end_3D[0], end_3D[1], 0.0], dtype=np.float32))

        if not prev_pts_2d:
            return

        prev_pts_2d = np.array(prev_pts_2d, dtype=np.float32).reshape(-1, 1, 2)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)

        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts_2d, None, **self.lk_params)

        if curr_pts is None:
            return

        valid = status.flatten() == 1
        good_prev = prev_pts_2d[valid]
        good_curr = curr_pts[valid]
        good_3d = [corresponding_3d[i] for i, v in enumerate(valid) if v]

        if len(good_prev) < 4:
            return

        # Re-estimate homography from tracked points
        good_3d_arr = np.array(good_3d, dtype=np.float32)
        H, mask = cv2.findHomography(good_3d_arr, good_curr, cv2.RANSAC, 5.0)

        if H is not None:
            self.last_good_H = H.copy()
            if self.camera.from_homography(H):
                self.validated = True

    def get_extremities_2d(self) -> Dict[str, List[Dict[str, float]]]:
        """Return current detected 2D extremities (for visualization)."""
        return self.prev_extremities_2d

    def get_camera(self) -> Camera:
        """Return current camera."""
        return self.camera

    def get_homography(self) -> Optional[np.ndarray]:
        """Return last good homography matrix."""
        return self.last_good_H.copy() if self.last_good_H is not None else None

    def reset(self) -> None:
        """Reset engine state."""
        self.frame_count = 0
        self.prev_frame = None
        self.prev_extremities_2d = {}
        self.prev_pix_pts = None
        self.prev_pix_pts_3d = {}
        self.last_good_H = None
        self.validated = False
        self.camera = Camera()
