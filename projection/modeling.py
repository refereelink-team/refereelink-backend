# -*- coding: utf-8 -*-
"""
projection 建模能力：
- 将 tracking 底座输出的像素 tracklet 映射为 2D 球场平面模型

增强功能：
- TrackSmoother: 球员轨迹时序平滑（EMA/Kalman）
- 增强脚点提取（支持姿态关键点）
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from projection.homography import HomographyAdapter
from tracking.backend import TrackedObject


@dataclass
class ProjectedTracklet:
    track_id: int
    class_id: int
    xyxy: Tuple[int, int, int, int]
    confidence: float
    map_x: float
    map_y: float
    team: str


@dataclass
class SmoothedPosition:
    """平滑后的位置"""
    x: float
    y: float
    vx: float = 0.0  # 速度 x
    vy: float = 0.0  # 速度 y
    confidence: float = 1.0


class TrackSmoother:
    """球员轨迹平滑器

    使用指数移动平均（EMA）或卡尔曼滤波对球员2D轨迹进行平滑。
    对每个 track_id 维护独立状态。
    """

    def __init__(
        self,
        method: str = "ema",
        alpha: float = 0.7,  # EMA 系数
        process_noise: float = 1.0,  # 卡尔曼过程噪声
        measurement_noise: float = 5.0,  # 卡尔曼观测噪声
    ):
        """
        Args:
            method: 平滑方法 "ema" 或 "kalman"
            alpha: EMA 系数 (0-1)，越大对新值响应越快
            process_noise: 卡尔曼过程噪声
            measurement_noise: 卡尔曼观测噪声
        """
        self.method = method
        self.alpha = alpha
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # 每个 track_id 的状态
        self.track_states: Dict[int, Dict] = {}

    def update(self, track_id: int, raw_x: float, raw_y: float) -> SmoothedPosition:
        """更新并返回平滑后的位置

        Args:
            track_id: 球员 track ID
            raw_x: 原始投影 x 坐标
            raw_y: 原始投影 y 坐标

        Returns:
            SmoothedPosition: 平滑后的位置
        """
        if track_id not in self.track_states:
            # 初始化状态
            self.track_states[track_id] = self._init_state(raw_x, raw_y)
            return SmoothedPosition(x=raw_x, y=raw_y, confidence=1.0)

        state = self.track_states[track_id]

        if self.method == "ema":
            smoothed = self._update_ema(state, raw_x, raw_y)
        else:
            smoothed = self._update_kalman(state, raw_x, raw_y)

        # 更新状态
        state["prev_x"] = smoothed.x
        state["prev_y"] = smoothed.y

        return smoothed

    def _init_state(self, x: float, y: float) -> Dict:
        """初始化状态"""
        if self.method == "kalman":
            # 状态: [x, y, vx, vy]
            state = np.array([x, y, 0.0, 0.0], dtype=np.float32)
            # 卡尔曼滤波器
            kf = cv2.KalmanFilter(4, 2, 0)
            kf.transitionMatrix = np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)
            kf.measurementMatrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ], dtype=np.float32)
            kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.process_noise
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.measurement_noise
            kf.errorCovPost = np.eye(4, dtype=np.float32)

            return {"kf": kf, "prev_x": x, "prev_y": y, "initialized": True}
        else:
            return {"prev_x": x, "prev_y": y, "initialized": True}

    def _update_ema(self, state: Dict, x: float, y: float) -> SmoothedPosition:
        """EMA 更新"""
        prev_x = state.get("prev_x", x)
        prev_y = state.get("prev_y", y)

        # EMA 平滑
        smooth_x = self.alpha * x + (1 - self.alpha) * prev_x
        smooth_y = self.alpha * y + (1 - self.alpha) * prev_y

        # 计算速度
        vx = smooth_x - prev_x
        vy = smooth_y - prev_y

        return SmoothedPosition(x=smooth_x, y=smooth_y, vx=vx, vy=vy)

    def _update_kalman(self, state: Dict, x: float, y: float) -> SmoothedPosition:
        """卡尔曼更新"""
        kf: cv2.KalmanFilter = state["kf"]

        # 预测
        prediction = kf.predict()

        # 修正
        measurement = np.array([[x], [y]], dtype=np.float32)
        estimated = kf.correct(measurement)

        smooth_x = float(estimated[0])
        smooth_y = float(estimated[1])
        vx = float(estimated[2])
        vy = float(estimated[3])

        return SmoothedPosition(x=smooth_x, y=smooth_y, vx=vx, vy=vy)

    def reset(self, track_id: Optional[int] = None) -> None:
        """重置状态

        Args:
            track_id: 如果指定，只重置该 track_id；否则重置所有
        """
        if track_id is not None:
            if track_id in self.track_states:
                del self.track_states[track_id]
        else:
            self.track_states.clear()


def extract_footpoint(obj: TrackedObject) -> Tuple[float, float]:
    """提取球员脚点

    当前实现：使用 bbox 底边中点
    增强方案：如果有姿态数据，可使用踝关键点

    Args:
        obj: 跟踪对象

    Returns:
        (u, v): 脚点像素坐标
    """
    x1, y1, x2, y2 = obj.xyxy
    foot_u = (x1 + x2) / 2.0
    foot_v = float(y2)
    return foot_u, foot_v


def project_single_bbox_multi(
    bbox_ltrb: Tuple[float, float, float, float],
    homography: HomographyAdapter,
) -> Tuple[float, float]:
    """将单个 bbox 的 3 个脚点投影到 2D 球场并取平均。

    使用 HomographyAdapter.pixel_to_field_meters 直接将图像像素投影到球场米坐标，
    比单点点投影更鲁棒。

    Args:
        bbox_ltrb: (left, top, right, bottom) 像素坐标
        homography: 单应矩阵适配器

    Returns:
        (map_x, map_y): 球场坐标（米，中心原点）
    """
    l, t, r, b = bbox_ltrb
    # 3 个脚点：左下角、右下角、底边中点
    bl_u, bl_v = l, b
    br_u, br_v = r, b
    bm_u, bm_v = l + (r - l) / 2.0, b

    # 直接投影到球场米坐标并取平均
    x_bl, y_bl = homography.pixel_to_field_meters(bl_u, bl_v)
    x_br, y_br = homography.pixel_to_field_meters(br_u, br_v)
    x_bm, y_bm = homography.pixel_to_field_meters(bm_u, bm_v)

    return (x_bl + x_br + x_bm) / 3.0, (y_bl + y_br + y_bm) / 3.0


def project_tracked_objects(
    tracked_objects: List[TrackedObject],
    homography: HomographyAdapter,
    track_smoother: Optional[TrackSmoother] = None,
) -> List[ProjectedTracklet]:
    """把 tracking 底座输出映射为 2D 平面 tracklet。

    Args:
        tracked_objects: 跟踪对象列表
        homography: 单应矩阵适配器
        track_smoother: 轨迹平滑器（可选）

    Returns:
        投影后的 tracklet 列表
    """
    projected: List[ProjectedTracklet] = []

    for obj in tracked_objects:
        # 使用 SN-style 多点点投影（3 点平均，更鲁棒）
        map_x, map_y = project_single_bbox_multi(obj.xyxy, homography)

        # 可选：轨迹平滑
        if track_smoother is not None:
            smoothed = track_smoother.update(obj.track_id, map_x, map_y)
            map_x = smoothed.x
            map_y = smoothed.y

        projected.append(
            ProjectedTracklet(
                track_id=obj.track_id,
                class_id=obj.class_id,
                xyxy=obj.xyxy,
                confidence=obj.confidence,
                map_x=map_x,
                map_y=map_y,
                team=obj.team,
            )
        )

    return projected


def create_track_smoother(
    method: str = "ema",
    alpha: float = 0.7,
) -> TrackSmoother:
    """创建轨迹平滑器

    Args:
        method: 平滑方法 "ema" 或 "kalman"
        alpha: EMA 系数

    Returns:
        TrackSmoother 实例
    """
    return TrackSmoother(method=method, alpha=alpha)
