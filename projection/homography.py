# -*- coding: utf-8 -*-
"""
透视变换（Homography）接口模块。
用于将视频像素坐标映射到战术板/世界坐标系，便于多视角扩展。

增强功能：
- HomographyState: 带质量指标的单应矩阵状态
- HomographyEstimator: 带 RANSAC 质量评估的估计器
- HomographyQualityGate: 质量门控
"""

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class HomographyAdapter:
    """
    透视变换适配器。
    当前实现：单应矩阵 H，将画面四点映射到战术板四点。
    """

    def __init__(
        self,
        src_pts: Optional[np.ndarray] = None,
        dst_pts: Optional[np.ndarray] = None,
        H_matrix: Optional[np.ndarray] = None,
    ):
        if H_matrix is not None:
            self.H = np.array(H_matrix, dtype=np.float32)
        elif src_pts is not None and dst_pts is not None:
            src_pts = np.array(src_pts, dtype=np.float32)
            dst_pts = np.array(dst_pts, dtype=np.float32)
            self.H, _ = cv2.findHomography(src_pts, dst_pts)
        else:
            raise ValueError("请提供 (src_pts, dst_pts) 或 H_matrix")

    def pixel_to_map(self, u: float, v: float) -> Tuple[float, float]:
        """将像素坐标 (u, v) 映射到战术板/世界坐标 (wx, wy)。"""
        pt = np.array([[[u, v]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self.H)[0][0]
        return float(mapped[0]), float(mapped[1])

    def pixel_points_to_map(self, points: np.ndarray) -> np.ndarray:
        """批量像素点映射。输入/输出均为 (N, 2)。"""
        if points.ndim == 1:
            points = points.reshape(1, -1)
        pts = points.astype(np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.H)
        return out.reshape(-1, 2)

    def set_homography(self, H: np.ndarray) -> None:
        """更新单应矩阵（例如切换相机或重新标定）。"""
        self.H = np.array(H, dtype=np.float32)

    def get_matrix(self) -> np.ndarray:
        """返回当前 3x3 单应矩阵。"""
        return self.H.copy()


# field_map.png 尺寸: 1200 x 800 (宽 x 高)
FIELD_MAP_WIDTH = 1200
FIELD_MAP_HEIGHT = 800

# field_map.png 上白线球场的真实边界（与图像绘制保持一致）
PITCH_LEFT = 75.0
PITCH_TOP = 60.0
PITCH_RIGHT = 1125.0
PITCH_BOTTOM = 740.0
PITCH_WIDTH = PITCH_RIGHT - PITCH_LEFT
PITCH_HEIGHT = PITCH_BOTTOM - PITCH_TOP

# 标准 11 人制足球场尺寸（米）
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
PENALTY_BOX_DEPTH_M = 16.5
PENALTY_BOX_WIDTH_M = 40.32
HALF_PITCH_LENGTH_M = PITCH_LENGTH_M / 2.0
HALF_PITCH_WIDTH_M = PITCH_WIDTH_M / 2.0
HALF_PENALTY_BOX_WIDTH_M = PENALTY_BOX_WIDTH_M / 2.0

# 默认目标映射：精确对应到 field_map.png 的球场白线四角
DEFAULT_DST_PTS = np.array([
    [PITCH_LEFT, PITCH_BOTTOM],
    [PITCH_RIGHT, PITCH_BOTTOM],
    [PITCH_RIGHT, PITCH_TOP],
    [PITCH_LEFT, PITCH_TOP],
], dtype=np.float32)


def _meters_to_map(x_m: float, y_m: float) -> Tuple[float, float]:
    """把标准球场米制坐标映射到 field_map.png 上的像素坐标。"""
    x = PITCH_LEFT + (x_m / PITCH_LENGTH_M) * PITCH_WIDTH
    y = PITCH_TOP + (y_m / PITCH_WIDTH_M) * PITCH_HEIGHT
    return x, y


# 标准球场关键点模板，全部锚定到“全场”几何而不是局部禁区区域。
PITCH_KEYPOINT_TEMPLATE: Dict[str, Tuple[float, float]] = {
    # 4 个球场角点
    "top_left_corner": (0.0, 0.0),
    "top_right_corner": (PITCH_LENGTH_M, 0.0),
    "bottom_right_corner": (PITCH_LENGTH_M, PITCH_WIDTH_M),
    "bottom_left_corner": (0.0, PITCH_WIDTH_M),
    # 4 个禁区角点
    "penalty_top_left": (PENALTY_BOX_DEPTH_M, HALF_PITCH_WIDTH_M - HALF_PENALTY_BOX_WIDTH_M),
    "penalty_top_right": (PITCH_LENGTH_M - PENALTY_BOX_DEPTH_M, HALF_PITCH_WIDTH_M - HALF_PENALTY_BOX_WIDTH_M),
    "penalty_bottom_right": (PITCH_LENGTH_M - PENALTY_BOX_DEPTH_M, HALF_PITCH_WIDTH_M + HALF_PENALTY_BOX_WIDTH_M),
    "penalty_bottom_left": (PENALTY_BOX_DEPTH_M, HALF_PITCH_WIDTH_M + HALF_PENALTY_BOX_WIDTH_M),
    # 4 个中线/边线中点
    "mid_top": (HALF_PITCH_LENGTH_M, 0.0),
    "mid_bottom": (HALF_PITCH_LENGTH_M, PITCH_WIDTH_M),
    "mid_left": (0.0, HALF_PITCH_WIDTH_M),
    "mid_right": (PITCH_LENGTH_M, HALF_PITCH_WIDTH_M),
}


def template_to_dst_points(keypoint_id: str) -> np.ndarray:
    """将模板坐标转换为 field_map.png 上的实际像素坐标。"""
    if keypoint_id not in PITCH_KEYPOINT_TEMPLATE:
        return None
    x_m, y_m = PITCH_KEYPOINT_TEMPLATE[keypoint_id]
    x, y = _meters_to_map(x_m, y_m)
    return np.array([x, y], dtype=np.float32)


def template_to_image_points() -> Dict[str, np.ndarray]:
    """将所有模板坐标转换为 DEFAULT_DST_PTS 坐标系下的实际像素坐标

    Returns:
        {keypoint_id: np.array([x, y])}
    """
    result = {}
    for keypoint_id in PITCH_KEYPOINT_TEMPLATE:
        pt = template_to_dst_points(keypoint_id)
        if pt is not None:
            result[keypoint_id] = pt
    return result


@dataclass
class HomographyState:
    """单应矩阵状态，包含质量指标"""
    H: np.ndarray                    # 3x3 单应矩阵
    inliers: int = 0                 # 内点数量
    inlier_ratio: float = 0.0        # 内点比例
    reproj_err_px_med: float = 0.0   # 中位重投影误差
    reproj_err_px_p90: float = 0.0   # P90 重投影误差
    fails: int = 0                   # 连续失败帧数
    mode: str = "TRACK"              # "TRACK" | "DETECT" | "FROZEN"
    last_good_ts: float = 0.0        # 上次成功时间戳

    @staticmethod
    def default() -> "HomographyState":
        """创建默认状态（使用默认单应矩阵）"""
        H = build_default_homography().get_matrix()
        return HomographyState(H=H, mode="TRACK")


@dataclass
class HomographyQuality:
    """单应矩阵质量指标"""
    inliers: int
    inlier_ratio: float
    reproj_err_px_med: float
    reproj_err_px_p90: float


class HomographyEstimator:
    """单应矩阵估计器，带 RANSAC 质量评估"""

    def __init__(
        self,
        ransac_reproj_threshold: float = 5.0,
        max_iters: int = 1000,
        confidence: float = 0.995,
    ):
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.max_iters = max_iters
        self.confidence = confidence

    def estimate(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[HomographyQuality]]:
        """估计单应矩阵并返回质量指标

        Args:
            src_pts: 源点 (N, 2)
            dst_pts: 目标点 (N, 2)

        Returns:
            (H, mask, quality) 或 (None, None, None) 如果失败
        """
        if len(src_pts) < 4 or len(dst_pts) < 4:
            return None, None, None

        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)

        try:
            H, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                self.ransac_reproj_threshold,
                maxIters=self.max_iters,
                confidence=self.confidence,
            )
        except Exception:
            return None, None, None

        if H is None:
            return None, None, None

        # 计算质量指标
        quality = self._evaluate_quality(src_pts, dst_pts, H, mask)

        return H, mask, quality

    def _evaluate_quality(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        H: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> HomographyQuality:
        """评估单应矩阵质量"""
        # 转换mask为bool数组
        if mask is not None:
            inliers = mask.flatten() == 1
            inlier_count = np.sum(inliers)
            inlier_ratio = inlier_count / len(src_pts) if len(src_pts) > 0 else 0.0
        else:
            inliers = np.ones(len(src_pts), dtype=bool)
            inlier_count = len(src_pts)
            inlier_ratio = 1.0

        # 计算重投影误差
        src_pts_np = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(src_pts_np, H)
        errors = np.linalg.norm(projected.reshape(-1, 2) - dst_pts, axis=1)

        # 只计算内点的误差
        inlier_errors = errors[inliers] if np.any(inliers) else errors
        reproj_err_med = float(np.median(inlier_errors)) if len(inlier_errors) > 0 else 0.0
        reproj_err_p90 = float(np.percentile(inlier_errors, 90)) if len(inlier_errors) > 0 else 0.0

        return HomographyQuality(
            inliers=inlier_count,
            inlier_ratio=inlier_ratio,
            reproj_err_px_med=reproj_err_med,
            reproj_err_px_p90=reproj_err_p90,
        )


class HomographyQualityGate:
    """单应矩阵质量门控"""

    def __init__(
        self,
        min_inliers: int = 6,
        min_inlier_ratio: float = 0.4,
        max_reproj_err_px: float = 10.0,
    ):
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.max_reproj_err_px = max_reproj_err_px

    def accept(self, quality: Optional[HomographyQuality]) -> bool:
        """判断质量是否可接受"""
        if quality is None:
            return False

        if quality.inliers < self.min_inliers:
            return False
        if quality.inlier_ratio < self.min_inlier_ratio:
            return False
        if quality.reproj_err_px_p90 > self.max_reproj_err_px:
            return False

        return True

    def get_rejection_reason(self, quality: Optional[HomographyQuality]) -> str:
        """获取拒绝原因"""
        if quality is None:
            return "quality is None"
        if quality.inliers < self.min_inliers:
            return f"inliers {quality.inliers} < {self.min_inliers}"
        if quality.inlier_ratio < self.min_inlier_ratio:
            return f"inlier_ratio {quality.inlier_ratio:.2f} < {self.min_inlier_ratio}"
        if quality.reproj_err_px_p90 > self.max_reproj_err_px:
            return f"reproj_err_p90 {quality.reproj_err_px_p90:.1f} > {self.max_reproj_err_px}"
        return "accepted"

# 默认源点：适配一般视频的中央区域（视频分辨率假设 1280x720 或类似）
# 这些点会在运行时自动调整
DEFAULT_SRC_PTS = np.array([
    [320, 540],     # 画面底部1/4处
    [960, 540],     # 画面底部1/4处
    [960, 180],     # 画面顶部1/4处
    [320, 180],     # 画面顶部1/4处
], dtype=np.float32)


def load_calibration(calibration_path: str) -> HomographyAdapter:
    """从 JSON 文件加载单应矩阵"""
    with open(calibration_path) as f:
        data = json.load(f)
    return HomographyAdapter(H_matrix=np.array(data["H"]))


def build_default_homography() -> HomographyAdapter:
    """使用默认标定构建透视变换器。"""
    return HomographyAdapter(src_pts=DEFAULT_SRC_PTS, dst_pts=DEFAULT_DST_PTS)


class HomographySmoother:
    """单应矩阵时序平滑器。

    使用指数滑动平均(EMA)对连续帧的单应矩阵进行平滑，
    减少由自动标定引起的抖动。
    """

    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: 平滑系数，取值(0,1)。值越大对新帧响应越快，值越大对旧帧保留越多
        """
        self.alpha = alpha
        self.H_prev: Optional[np.ndarray] = None

    def update(self, H_new: np.ndarray) -> np.ndarray:
        """更新并返回平滑后的单应矩阵"""
        H_new = np.array(H_new, dtype=np.float32)

        if self.H_prev is None:
            self.H_prev = H_new
            return H_new.copy()

        # 指数滑动平均: H_smooth = alpha * H_new + (1 - alpha) * H_prev
        H_smooth = self.alpha * H_new + (1 - self.alpha) * self.H_prev
        self.H_prev = H_smooth

        return H_smooth

    def reset(self) -> None:
        """重置平滑器状态"""
        self.H_prev = None
