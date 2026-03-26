# -*- coding: utf-8 -*-
"""
镜头切换检测器。

基于内容差分检测镜头切换（cut），用于触发重定位。
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class ShotChangeDetector:
    """镜头切换检测器

    使用内容差分（直方图/像素差异）检测镜头切换。
    检测到切镜时触发重定位，避免错误投影污染下游。
    """

    def __init__(
        self,
        hist_threshold: float = 0.3,    # 直方图差异阈值
        pixel_threshold: float = 0.4,    # 像素差异阈值
        min_changed_pixels: int = 10000,  # 最小变化像素数
        roi_y_start: float = 0.3,       # ROI 起始 y 坐标
        roi_y_end: float = 1.0,          # ROI 结束 y 坐标
    ):
        self.hist_threshold = hist_threshold
        self.pixel_threshold = pixel_threshold
        self.min_changed_pixels = min_changed_pixels
        self.roi_y_start = roi_y_start
        self.roi_y_end = roi_y_end

        self.prev_frame: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_hist: Optional[np.ndarray] = None

    def update(self, frame: np.ndarray) -> bool:
        """检测当前帧是否发生了镜头切换

        Args:
            frame: 当前帧 (BGR)

        Returns:
            True 如果检测到镜头切换
        """
        # 提取 ROI（球场区域）
        h, w = frame.shape[:2]
        roi = frame[int(h * self.roi_y_start):int(h * self.roi_y_end), :]

        # 转换为灰度
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_frame = frame.copy()
            return False

        # 方法1: 直方图差异
        hist_diff = self._compute_hist_diff(gray, self.prev_gray)

        # 方法2: 像素差异
        pixel_diff, changed_pixels = self._compute_pixel_diff(gray, self.prev_gray)

        # 判断是否发生切镜
        cut_detected = (
            hist_diff > self.hist_threshold or
            (pixel_diff > self.pixel_threshold and changed_pixels > self.min_changed_pixels)
        )

        # 更新状态
        self.prev_gray = gray
        self.prev_frame = frame.copy()

        return cut_detected

    def _compute_hist_diff(self, gray1: np.ndarray, gray2: np.ndarray) -> float:
        """计算直方图差异"""
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

        # 归一化
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()

        # 计算巴氏距离
        hist_diff = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)

        return float(hist_diff)

    def _compute_pixel_diff(
        self,
        gray1: np.ndarray,
        gray2: np.ndarray,
    ) -> tuple[float, int]:
        """计算像素差异"""
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

        diff = cv2.absdiff(gray1, gray2)
        thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]

        changed_pixels = cv2.countNonZero(thresholded)
        total_pixels = gray1.shape[0] * gray1.shape[1]

        pixel_diff = changed_pixels / total_pixels if total_pixels > 0 else 0.0

        return float(pixel_diff), int(changed_pixels)

    def reset(self) -> None:
        """重置检测器状态"""
        self.prev_frame = None
        self.prev_gray = None
        self.prev_hist = None