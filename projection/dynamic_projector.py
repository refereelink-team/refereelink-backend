# -*- coding: utf-8 -*-
"""
动态投影器。

整合球场自动标定、光流跟踪、时序平滑、质量门控，实现动态相机下的实时投影。

重构后的三模式运行：
- DETECT: 重定位/强制检测模式
- TRACK: 检测间隙用跟踪维持模式
- FROZEN: 短时冻结 H 等待恢复模式
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from projection.calibrator import (
    FieldCalibrator,
    LineFallbackDetector,
    PitchKeypointDetector,
)
from projection.homography import (
    DEFAULT_DST_PTS,
    HomographyAdapter,
    HomographyEstimator,
    HomographyQuality,
    HomographyQualityGate,
    HomographyState,
    HomographySmoother,
    build_default_homography,
    template_to_image_points,
)
from projection.keypoint_manager import KeypointManager, PITCH_KEYPOINT_IDS
from projection.shot_change_detector import ShotChangeDetector


class DynamicProjector:
    """动态投影器：自动标定 + 光流更新 + 时序平滑 + 质量门控

    使用三级策略：
    - 低频（每N帧）：精确重标定 + 质量门控
    - 高频（中间帧）：光流传播更新
    - 始终：时序平滑 + 失败重定位

    Args:
        recalib_interval: 重标定间隔帧数（默认5，即5-10Hz）
        field_path: 球场模板图片路径
        smooth_alpha: 平滑系数 (0,1)
        debug: 是否输出调试信息
    """

    def __init__(
        self,
        recalib_interval: int = 5,
        field_path: str = "field_map.png",
        smooth_alpha: float = 0.6,
        debug: bool = False,
        # 质量门控参数
        min_inliers: int = 4,
        min_inlier_ratio: float = 0.35,
        max_reproj_err: float = 20.0,
        # 失败重定位参数
        max_consecutive_fails: int = 10,
        # 光流参数
        flow_interval: int = 1,
    ):
        self.recalib_interval = recalib_interval
        self.field_path = field_path
        self.debug = debug
        self.smooth_alpha = smooth_alpha

        # 组件 - 关键点检测
        self.keypoint_detector = PitchKeypointDetector()
        self.line_fallback_detector = LineFallbackDetector()

        # 组件 - 标定器（用于光流跟踪）
        self.calibrator = FieldCalibrator(use_yolo=True)

        # 组件 - 质量评估
        self.estimator = HomographyEstimator(
            ransac_reproj_threshold=5.0,
            max_iters=1000,
            confidence=0.995,
        )
        self.quality_gate = HomographyQualityGate(
            min_inliers=min_inliers,
            min_inlier_ratio=min_inlier_ratio,
            max_reproj_err_px=max_reproj_err,
        )

        # 组件 - 关键点管理器
        self.keypoint_manager = KeypointManager(
            max_age=30,
            min_confidence=0.3,
        )

        # 组件 - 镜头切换检测
        self.shot_detector = ShotChangeDetector(
            hist_threshold=0.3,
            pixel_threshold=0.4,
            min_changed_pixels=10000,
        )

        # 组件 - 时序平滑
        self.smoother = HomographySmoother(alpha=smooth_alpha)

        # 状态
        self.frame_count = 0
        self.prev_frame: Optional[np.ndarray] = None
        self.h_state: HomographyState = HomographyState.default()
        self.default_homography = build_default_homography()
        self._initialized = False  # 是否已经初始化过有效的 H
        self._h_validated = False  # H 是否已经通过质量验证
        self._consecutive_rejections = 0  # 连续被拒绝的次数

        # 模板关键点坐标
        self.template_pts = template_to_image_points()

        # 参数
        self.max_consecutive_fails = max_consecutive_fails
        self.flow_interval = flow_interval
        self.last_detect_frame = 0

    def update(self, frame: np.ndarray) -> HomographyAdapter:
        """更新并返回当前帧的单应矩阵适配器

        Args:
            frame: 当前视频帧 (BGR)

        Returns:
            HomographyAdapter: 当前可用的单应矩阵适配器
        """
        self.frame_count += 1

        # 1) 检测镜头切换
        if self.shot_detector.update(frame):
            if self.debug:
                print(f"[DynamicProjector] Frame {self.frame_count}: SHOT CHANGE DETECTED")
            self._handle_shot_change()

        # 2) 决定运行模式
        should_detect = self._should_run_detection()

        if should_detect:
            self._run_detection(frame)
        elif self.prev_frame is not None and self.h_state.H is not None:
            # 3) 光流跟踪更新
            self._run_tracking(frame)

        # 为下一帧缓存当前原始帧，供光流跟踪使用
        self.prev_frame = frame.copy()

        # 4) 时序平滑
        if self._h_validated and self.h_state.H is not None:
            self._initialized = True
            H_smooth = self.smoother.update(self.h_state.H)
            return HomographyAdapter(H_matrix=H_smooth)

        # 如果还没有初始化，返回默认
        if not self._initialized:
            return self.default_homography

        # 兜底：返回默认（不应该到这里）
        return self.default_homography

    def _should_run_detection(self) -> bool:
        """判断是否应该运行关键点检测"""
        # 初始帧或未初始化
        if not self._initialized or not self._h_validated:
            return True

        # 连续失败后的重定位
        if self.h_state.mode == "DETECT":
            return True

        # 定期检测
        if (self.frame_count - self.last_detect_frame) >= self.recalib_interval:
            return True

        # 连续失败过多，强制重定位
        if self.h_state.fails >= self.max_consecutive_fails:
            return True

        # 关键点不足时
        if self.keypoint_manager.get_confident_count() < 4:
            return True

        return False

    def _run_detection(self, frame: np.ndarray) -> None:
        """运行关键点检测并更新单应矩阵"""
        if self.debug:
            print(f"[DynamicProjector] Frame {self.frame_count}: Running DETECTION")

        self.last_detect_frame = self.frame_count

        # 优先使用带固定语义 ID 的 YOLO 检测
        detections = self.keypoint_detector.detect_labeled_keypoints(frame)

        if detections:
            self.keypoint_manager.fuse_detected_keypoints(detections)

        if self.keypoint_manager.get_confident_count() < 4:
            # 回退到线段检测
            if self.debug:
                print("[DynamicProjector] YOLO detection failed, trying line fallback")
            line_keypoints = self.line_fallback_detector.detect_keypoints(frame)
            if line_keypoints:
                self.keypoint_manager.fuse_detected_keypoints(line_keypoints)

        # 估计单应矩阵
        self._estimate_homography_from_keypoints(frame)

    def _run_tracking(self, frame: np.ndarray) -> None:
        """使用光流跟踪更新关键点和单应矩阵"""
        if self.prev_frame is None:
            return

        # 跟踪关键点
        self.keypoint_manager.track(self.prev_frame, frame)

        # 检查关键点数量
        confident_count = self.keypoint_manager.get_confident_count()
        if confident_count < 4:
            # 关键点不足，触发重新检测
            if self.debug:
                print(f"[DynamicProjector] Frame {self.frame_count}: Low keypoints ({confident_count}), triggering redetection")
            self.h_state.mode = "DETECT"
            return

        # 从跟踪的关键点估计单应矩阵
        self._estimate_homography_from_keypoints(frame)

    def _estimate_homography_from_keypoints(self, frame: Optional[np.ndarray] = None) -> None:
        """从关键点管理器估计单应矩阵"""
        src_pts, dst_pts, weights = self.keypoint_manager.build_correspondences(self.template_pts)

        if len(src_pts) < 4:
            self._handle_estimation_failure()
            return

        # 估计单应矩阵
        H_candidate, mask, quality = self.estimator.estimate(src_pts, dst_pts)

        if H_candidate is None or quality is None:
            self._handle_estimation_failure()
            return

        if frame is not None and not self.calibrator.validate_homography(H_candidate, frame):
            if self.debug:
                print("[DynamicProjector] H rejected: failed geometric validation")
            self._handle_estimation_failure()
            return

        # 质量门控
        if self.quality_gate.accept(quality):
            # 更新状态
            self.h_state.H = H_candidate
            self.h_state.inliers = quality.inliers
            self.h_state.inlier_ratio = quality.inlier_ratio
            self.h_state.reproj_err_px_med = quality.reproj_err_px_med
            self.h_state.reproj_err_px_p90 = quality.reproj_err_px_p90
            self.h_state.fails = 0
            self.h_state.mode = "TRACK"
            self.h_state.last_good_ts = time.time()

            if self.debug:
                print(f"[DynamicProjector] H update: inliers={quality.inliers}, "
                      f"ratio={quality.inlier_ratio:.2f}, err_p90={quality.reproj_err_px_p90:.1f}")
            # 更新成功，标记为已验证
            self._h_validated = True
            self._consecutive_rejections = 0
        else:
            # 质量不通过，拒绝更新
            reason = self.quality_gate.get_rejection_reason(quality)
            if self.debug:
                print(f"[DynamicProjector] H rejected: {reason}")
            self._handle_estimation_failure()

    def _handle_estimation_failure(self) -> None:
        """处理单应估计失败（无法估计新 H）"""
        self.h_state.fails += 1
        self._consecutive_rejections += 1

        if self.h_state.fails >= self.max_consecutive_fails:
            if self.debug:
                print("[DynamicProjector] Too many failures, forcing relocalization")
            self.h_state.mode = "DETECT"
            self._h_validated = False
            self.smoother.reset()
        else:
            self.h_state.mode = "FROZEN"

    def _handle_shot_change(self) -> None:
        """处理镜头切换"""
        # 清空关键点状态
        self.keypoint_manager.clear()
        # 强制重检测
        self.h_state.mode = "DETECT"
        self.h_state.fails = 0
        # 重置平滑器
        self.smoother.reset()
        self.h_state = HomographyState.default()
        self.h_state.mode = "DETECT"
        # 重置状态
        self._initialized = False
        self._h_validated = False
        self._consecutive_rejections = 0
        self.prev_frame = None

    def get_state(self) -> HomographyState:
        """获取当前单应状态"""
        return self.h_state

    def reset(self) -> None:
        """重置状态"""
        self.frame_count = 0
        self.prev_frame = None
        self.h_state = HomographyState.default()
        self.keypoint_manager.clear()
        self.smoother.reset()
        self.shot_detector.reset()
        self._initialized = False
        self._h_validated = False
        self._consecutive_rejections = 0

    def force_recalibrate(self, frame: np.ndarray) -> HomographyAdapter:
        """强制重标定"""
        self.h_state.mode = "DETECT"
        self._run_detection(frame)

        if self.h_state.H is not None:
            return HomographyAdapter(H_matrix=self.h_state.H)

        return self.default_homography


def create_dynamic_projector(
    recalib_interval: int = 5,
    field_path: str = "field_map.png",
    smooth_alpha: float = 0.6,
    debug: bool = False,
    # 质量门控参数
    min_inliers: int = 4,
    min_inlier_ratio: float = 0.35,
    max_reproj_err: float = 20.0,
    # 失败重定位参数
    max_consecutive_fails: int = 10,
) -> DynamicProjector:
    """创建动态投影器的工厂函数

    Args:
        recalib_interval: 重标定间隔帧数（默认5，即5-10Hz）
        field_path: 球场模板图片路径
        smooth_alpha: 平滑系数
        debug: 是否输出调试信息
        min_inliers: 最小内点数
        min_inlier_ratio: 最小内点比例
        max_reproj_err: 最大重投影误差
        max_consecutive_fails: 最大连续失败次数

    Returns:
        DynamicProjector 实例
    """
    return DynamicProjector(
        recalib_interval=recalib_interval,
        field_path=field_path,
        smooth_alpha=smooth_alpha,
        debug=debug,
        min_inliers=min_inliers,
        min_inlier_ratio=min_inlier_ratio,
        max_reproj_err=max_reproj_err,
        max_consecutive_fails=max_consecutive_fails,
    )
