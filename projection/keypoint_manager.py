# -*- coding: utf-8 -*-
"""
关键点管理器。

管理球场语义关键点的检测、跟踪和融合。
支持低频检测（YOLO）和高频光流跟踪的混合策略。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from projection.homography import PITCH_KEYPOINT_TEMPLATE

# 球场语义关键点定义
# 注意：YOLO 模型输出的顺序与此不同，需要通过 KEYPOINT_ORDER_MAP 映射
PITCH_KEYPOINT_IDS = [
    # 4 个球场角点 (YOLO 输出顺序可能是不同的)
    "top_left_corner",
    "top_right_corner",
    "bottom_right_corner",
    "bottom_left_corner",
    # 4 个禁区角点
    "penalty_top_left",
    "penalty_top_right",
    "penalty_bottom_right",
    "penalty_bottom_left",
    # 4 个中线点
    "mid_top",
    "mid_bottom",
    "mid_left",
    "mid_right",
]

# YOLO 模型输出的关键点顺序（需要根据实际模型调整）
# 这个顺序是基于观察的，实际使用中需要根据具体模型输出调整
# 当前使用置信度过滤：只保留置信度 >= 0.3 的关键点
YOLO_KEYPOINT_ORDER = [
    "top_left_corner",    # 0: 可能是顶部角点
    "top_right_corner",   # 1: 可能是右侧角点
    "bottom_left_corner", # 2: 可能是左下角点
    "bottom_right_corner", # 3: 可能是右下角点
    "penalty_bottom_left", # 4: 可能是禁区下角
    "mid_top",            # 5: 可能是中线顶部
    "penalty_bottom_right", # 6: 可能是禁区内角
    "penalty_top_left",   # 7: 可能是禁区内上角
    "mid_bottom",         # 8: 可能是中线下部
    "mid_right",          # 9: 可能是中线右侧
    "mid_left",           # 10: 可能是中线左侧
    "penalty_top_right",  # 11: 可能是禁区内右上
]

@dataclass
class TrackedKeypoint:
    """单个跟踪的关键点"""
    pt: Tuple[float, float]          # 图像坐标 (x, y)
    confidence: float = 1.0          # 置信度 0-1
    age: int = 0                     # 年龄（帧数）
    src: str = "det"                 # 来源: "det" | "flow"
    visible: bool = True              # 是否可见
    matched: bool = False            # 当前帧是否匹配成功


class KeypointManager:
    """关键点管理器

    管理语义关键点的生命周期：
    - 融合检测到的关键点
    - 使用光流跟踪关键点
    - 处理关键点可见性
    - 维护关键点年龄用于信任度衰减
    """

    def __init__(
        self,
        max_age: int = 30,           # 最大年龄，超过后删除
        min_confidence: float = 0.3, # 最小置信度
        flow_window_size: int = 21,  # 光流窗口大小
        flow_max_level: int = 3,     # 金字塔层数
    ):
        self.max_age = max_age
        self.min_confidence = min_confidence

        # 光流参数
        self.lk_params = dict(
            winSize=(flow_window_size, flow_window_size),
            maxLevel=flow_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )

        # 关键点池
        self.keypoints: Dict[str, TrackedKeypoint] = {}

        # 缓存
        self.prev_gray: Optional[np.ndarray] = None

    def fuse_detected_keypoints(
        self,
        detections: Dict[str, Tuple[float, float, float]],
    ) -> None:
        """融合检测到的关键点

        Args:
            detections: {keypoint_id: (x, y, confidence)}
        """
        for keypoint_id, (x, y, conf) in detections.items():
            if conf < self.min_confidence:
                continue

            if keypoint_id in self.keypoints:
                # 更新现有关键点
                kp = self.keypoints[keypoint_id]
                kp.pt = (x, y)
                kp.confidence = conf
                kp.age = 0
                kp.src = "det"
                kp.visible = True
                kp.matched = True
            else:
                # 创建新关键点
                self.keypoints[keypoint_id] = TrackedKeypoint(
                    pt=(x, y),
                    confidence=conf,
                    age=0,
                    src="det",
                    visible=True,
                    matched=True,
                )

    def track(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
    ) -> Dict[str, TrackedKeypoint]:
        """使用光流跟踪关键点

        Args:
            prev_frame: 上一帧
            curr_frame: 当前帧

        Returns:
            更新后的关键点字典
        """
        if not self.keypoints:
            return self.keypoints

        # 转换为灰度图
        if prev_frame.ndim == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame

        if curr_frame.ndim == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame

        self.prev_gray = curr_gray

        # 提取需要跟踪的点
        pts = np.array(
            [[kp.pt[0], kp.pt[1]] for kp in self.keypoints.values()],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        if len(pts) == 0:
            return self.keypoints

        # KLT 光流跟踪
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts, None, **self.lk_params
        )

        if curr_pts is None:
            return self.keypoints

        # 更新关键点状态
        for i, (keypoint_id, kp) in enumerate(self.keypoints.items()):
            if i >= len(status):
                break

            if status[i][0] == 1:
                # 跟踪成功
                kp.pt = (float(curr_pts[i][0][0]), float(curr_pts[i][0][1]))
                kp.age += 1
                kp.src = "flow"
                kp.visible = True
                kp.matched = True

                # 根据年龄衰减置信度
                kp.confidence = max(0.3, kp.confidence * 0.98)
            else:
                # 跟踪失败
                kp.age += 1
                kp.matched = False
                # 跟踪失败时置信度下降更快
                kp.confidence = max(0.1, kp.confidence * 0.8)

            # 超过最大年龄，标记为不可见
            if kp.age > self.max_age:
                kp.visible = False

        return self.keypoints

    def build_correspondences(
        self,
        template_pts: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构建关键点对应关系

        Args:
            template_pts: {keypoint_id: template_image_point}

        Returns:
            (src_pts, dst_pts, weights) 用于单应估计
        """
        src_pts = []
        dst_pts = []
        weights = []

        for keypoint_id, kp in self.keypoints.items():
            if not kp.visible or kp.confidence < self.min_confidence:
                continue

            if keypoint_id in template_pts:
                src_pts.append(kp.pt)
                dst_pts.append(template_pts[keypoint_id])
                weights.append(kp.confidence)

        if not src_pts:
            return np.array([]), np.array([]), np.array([])

        return (
            np.array(src_pts, dtype=np.float32),
            np.array(dst_pts, dtype=np.float32),
            np.array(weights, dtype=np.float32),
        )

    def get_visible_count(self) -> int:
        """获取可见关键点数量"""
        return sum(1 for kp in self.keypoints.values() if kp.visible)

    def get_confident_count(self) -> int:
        """获取高置信度关键点数量"""
        return sum(
            1 for kp in self.keypoints.values()
            if kp.visible and kp.confidence >= self.min_confidence
        )

    def clear(self) -> None:
        """清空所有关键点"""
        self.keypoints.clear()
        self.prev_gray = None

    def reset_age(self) -> None:
        """重置所有关键点年龄"""
        for kp in self.keypoints.values():
            kp.age = 0


def template_to_image_points() -> Dict[str, np.ndarray]:
    """兼容旧接口，复用 homography 中的模板定义。"""
    from projection.homography import template_to_image_points as build_template_points

    return build_template_points()
