# -*- coding: utf-8 -*-
"""
透视变换（Homography）接口模块。
用于将视频像素坐标映射到战术板/世界坐标系，便于多视角扩展。
"""

from typing import Optional, Tuple

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


# 默认标定（左上, 右上, 右下, 左下）
DEFAULT_SRC_PTS = np.array(
    [[1297, 1217], [920, 803], [1929, 700], [2559, 823]], dtype=np.float32
)
DEFAULT_DST_PTS = np.array(
    [[240, 603], [240, 201], [963, 199], [960, 601]], dtype=np.float32
)


def build_default_homography() -> HomographyAdapter:
    """使用默认标定构建透视变换器。"""
    return HomographyAdapter(src_pts=DEFAULT_SRC_PTS, dst_pts=DEFAULT_DST_PTS)
