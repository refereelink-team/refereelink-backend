# -*- coding: utf-8 -*-
"""
球场自动标定模块。

提供基于球场线检测的自动标定功能，以及基于光流的帧间跟踪更新。
支持使用YOLO模型检测关键点或简化版角点检测。

增强功能：
- 质量评估：返回 inlier mask 和重投影误差
- 线段检测回退：当关键点不足时使用 LSD 线段检测
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from projection.homography import DEFAULT_DST_PTS, HomographyQuality, template_to_image_points

# 足球场绿色 (HSV范围)
FIELD_GREEN_LOWER = np.array([25, 40, 40])
FIELD_GREEN_UPPER = np.array([85, 255, 255])

# 白线阈值
WHITE_THRESHOLD = 200


class PitchKeypointDetector:
    """球场关键点检测器。

    优先使用YOLO模型检测12个关键点（4球场角 + 4禁区角 + 4中线点）。
    备选方案：简化版角点检测（基于HSV颜色分割）。
    """

    # Keypoint indices for standard pitch detection
    # 0-3: Field corners (top-left, top-right, bottom-right, bottom-left)
    # 4-7: Penalty area corners
    # 8-11: Midline points
    KEYPOINT_CLASSES = [
        "top_left_corner", "top_right_corner", "bottom_right_corner", "bottom_left_corner",
        "penalty_top_left", "penalty_top_right", "penalty_bottom_right", "penalty_bottom_left",
        "mid_top", "mid_bottom", "mid_left", "mid_right"
    ]

    def __init__(self, model_path: str = "tracking/data/football-pitch-detection.pt"):
        self.model_path = model_path
        self.model = None
        self._model_loaded = False

    def _load_model(self):
        """Lazy load YOLO model on first use"""
        if self._model_loaded:
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self._model_loaded = True
            print("[PitchKeypointDetector] YOLO model loaded successfully")
        except Exception as e:
            print(f"[PitchKeypointDetector] Failed to load YOLO model: {e}")
            self._model_loaded = True  # Mark as tried, won't retry

    def detect_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """检测球场关键点

        Args:
            frame: BGR图像

        Returns:
            关键点坐标数组 (12, 2) 或 None
        """
        labeled = self.detect_labeled_keypoints(frame)
        if labeled:
            coords = []
            for keypoint_id in self.KEYPOINT_CLASSES:
                if keypoint_id not in labeled:
                    continue
                x, y, conf = labeled[keypoint_id]
                if conf >= 0.3:
                    coords.append((x, y))
            if len(coords) >= 4:
                return np.array(coords, dtype=np.float32)

        # Fallback: use simplified corner detection
        return self.detect_corners_simple(frame)

    def detect_labeled_keypoints(
        self,
        frame: np.ndarray,
    ) -> Dict[str, Tuple[float, float, float]]:
        """按固定语义 ID 解码 YOLO 输出的球场关键点。"""
        self._load_model()

        if self.model is not None:
            try:
                results = self.model(frame, verbose=False)[0]
                keypoints = results.keypoints

                if keypoints is not None and len(keypoints) > 0:
                    kp_data = keypoints.data[0]
                    coords = kp_data[:, :2].cpu().numpy()
                    if kp_data.shape[1] >= 3:
                        confidences = kp_data[:, 2].cpu().numpy()
                    else:
                        confidences = np.ones(len(coords), dtype=np.float32) * 0.9

                    labeled = {}
                    for idx, keypoint_id in enumerate(self.KEYPOINT_CLASSES):
                        if idx >= len(coords):
                            break
                        x, y = coords[idx]
                        conf = float(confidences[idx]) if idx < len(confidences) else 0.9
                        labeled[keypoint_id] = (float(x), float(y), conf)
                    return labeled
            except Exception as e:
                print(f"[PitchKeypointDetector] YOLO detection failed: {e}")

        corners = self.detect_corners_simple(frame)
        if corners is None or len(corners) < 4:
            return {}

        corner_ids = [
            "top_left_corner",
            "top_right_corner",
            "bottom_right_corner",
            "bottom_left_corner",
        ]
        labeled = {}
        for keypoint_id, (x, y) in zip(corner_ids, corners):
            labeled[keypoint_id] = (float(x), float(y), 0.7)
        return labeled

    def detect_keypoints_with_confidence(
        self,
        frame: np.ndarray,
    ) -> Optional[List[Tuple[float, float, float]]]:
        """检测球场关键点（带置信度）

        Returns:
            [(x, y, confidence), ...] 或 None
        """
        labeled = self.detect_labeled_keypoints(frame)
        if not labeled:
            return None

        result = []
        for keypoint_id in self.KEYPOINT_CLASSES:
            if keypoint_id not in labeled:
                continue
            result.append(labeled[keypoint_id])
        return result or None

    def detect_corners_simple(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """简化版角点检测（备选方案）

        只检测球场四角，使用HSV颜色分割和白线检测。
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. 提取绿色区域（球场区域）
        green_mask = cv2.inRange(hsv, FIELD_GREEN_LOWER, FIELD_GREEN_UPPER)

        # 2. 提取白线
        _, white_mask = cv2.threshold(gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)

        # 3. 取交集
        field_mask = cv2.bitwise_and(white_mask, green_mask)

        # 4. 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 5. 找到轮廓
        contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find the largest contour (likely the field)
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approx) >= 4:
            # Get corners from the approximated polygon
            corners = approx.squeeze()
            if len(corners) == 4:
                # Sort corners: top-left, top-right, bottom-right, bottom-left
                corners = self._sort_corners(corners)
                return corners.astype(np.float32)

        # Fallback: use extreme points
        h, w = frame.shape[:2]
        return np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype=np.float32)

    def _sort_corners(self, corners: np.ndarray) -> np.ndarray:
        """Sort corners in order: top-left, top-right, bottom-right, bottom-left"""
        # Calculate sum and difference for sorting
        sums = corners.sum(axis=1)
        diffs = corners[:, 1] - corners[:, 0]

        # Order: top-left (min sum), top-right (min diff), bottom-right (max sum), bottom-left (max diff)
        sorted_indices = np.argsort(sums)
        tl = sorted_indices[0]  # top-left

        remaining = [i for i in sorted_indices if i != tl]
        tr = remaining[0] if corners[remaining[0], 0] > corners[remaining[1], 0] else remaining[1]

        return corners[[tl, tr, tr, tl]]


class FieldLineDetector:
    """球场线检测器"""

    def __init__(self):
        self.roi_y_start = 0.3  # 假设球场在画面中下部

    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        检测球场线段。

        Args:
            frame: BGR图像

        Returns:
            线段数组，每条线段为 [x1, y1, x2, y2]
        """
        # 转灰度和HSV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. 提取绿色区域（球场区域）
        green_mask = cv2.inRange(hsv, FIELD_GREEN_LOWER, FIELD_GREEN_UPPER)

        # 2. 提取白线
        _, white_mask = cv2.threshold(gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)

        # 3. 取交集：只在球场区域内找白线
        field_mask = cv2.bitwise_and(white_mask, green_mask)

        # 4. 形态学处理去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 5. 霍夫线检测
        lines = cv2.HoughLinesP(
            field_mask,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )

        if lines is None:
            return None

        # 过滤并返回线段
        valid_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > 30:  # 过滤短线
                valid_lines.append([x1, y1, x2, y2])

        return np.array(valid_lines) if valid_lines else None


class FieldCalibrator:
    """球场标定器

    使用球场线检测和光流跟踪来动态估计单应矩阵。
    优先使用PitchKeypointDetector检测关键点，备选方案使用传统HSV+霍夫线检测。
    """

    def __init__(self, use_yolo: bool = True):
        self.line_detector = FieldLineDetector()
        self.keypoint_detector = PitchKeypointDetector() if use_yolo else None
        self.template_pts = template_to_image_points()
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def _build_semantic_correspondences(
        self,
        detections: Dict[str, Tuple[float, float, float]],
        min_confidence: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """把带语义的关键点检测转换为图像点与模板点的对应。"""
        src_pts = []
        dst_pts = []

        for keypoint_id, (x, y, conf) in detections.items():
            if conf < min_confidence or keypoint_id not in self.template_pts:
                continue
            src_pts.append((x, y))
            dst_pts.append(self.template_pts[keypoint_id])

        if not src_pts:
            return np.array([]), np.array([])

        return np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32)

    def detect_corners(self, lines: np.ndarray) -> Optional[np.ndarray]:
        """从检测到的线段中提取角点（简化版）"""
        if lines is None or len(lines) < 4:
            return None

        # 收集所有端点
        points = []
        for line in lines:
            x1, y1, x2, y2 = line
            points.append((x1, y1))
            points.append((x2, y2))

        points = np.array(points)

        # 使用简单的几何规则找角落：
        # 找画面中最上/下/左/右的点作为候选角点
        ys = points[:, 1]
        xs = points[:, 0]

        # 粗略角点：基于分位数
        top_y = np.percentile(ys, 20)
        bottom_y = np.percentile(ys, 80)
        left_x = np.percentile(xs, 20)
        right_x = np.percentile(xs, 80)

        # 找最接近这些位置的点
        def find_closest(points, target_y, target_x):
            distances = (points[:, 1] - target_y) ** 2 + (points[:, 0] - target_x) ** 2
            return points[np.argmin(distances)]

        corners = [
            find_closest(points, bottom_y, left_x),   # 左下
            find_closest(points, bottom_y, right_x),  # 右下
            find_closest(points, top_y, right_x),     # 右上
            find_closest(points, top_y, left_x),     # 左上
        ]

        return np.array(corners, dtype=np.float32)

    def estimate_homography(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """从当前帧估计单应矩阵

        优先使用YOLO关键点检测（如果可用），失败时回退到传统HSV+霍夫线检测。
        """
        # Try YOLO keypoint detection first (if enabled)
        if self.keypoint_detector is not None:
            detections = self.keypoint_detector.detect_labeled_keypoints(frame)
            src_pts, dst_pts = self._build_semantic_correspondences(detections)
            if len(src_pts) >= 4:
                try:
                    H, _ = cv2.findHomography(src_pts, dst_pts)
                    return H
                except Exception as e:
                    print(f"[FieldCalibrator] YOLO homography failed: {e}")

        # Fallback: traditional line detection
        lines = self.line_detector.detect(frame)
        corners = self.detect_corners(lines)

        if corners is None or len(corners) < 4:
            return None

        # 目标点：field_map的角点
        from projection.homography import DEFAULT_DST_PTS

        try:
            H, _ = cv2.findHomography(corners, DEFAULT_DST_PTS)
            return H
        except:
            return None

    def track_with_flow(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_pts: np.ndarray
    ) -> Optional[np.ndarray]:
        """使用光流跟踪更新单应矩阵

        Args:
            prev_frame: 上一帧
            curr_frame: 当前帧
            prev_pts: 上一帧跟踪点

        Returns:
            更新后的单应矩阵，或None如果跟踪失败
        """
        if prev_pts is None or len(prev_pts) < 4:
            return None

        # 转灰度
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # KLT光流跟踪
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **self.lk_params
        )

        if curr_pts is None:
            return None

        # 过滤成功跟踪的点
        valid = status.flatten() == 1
        good_prev = prev_pts[valid]
        good_curr = curr_pts[valid]

        if len(good_prev) < 4:
            return None

        try:
            # 估计单应矩阵
            H, _ = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)
            return H
        except:
            return None

    def extract_tracking_points(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """提取用于光流跟踪的特征点"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 在画面中下部提取特征点（球场区域）
        h, w = gray.shape
        roi = gray[int(h * 0.3):, :]  # 只看下半部分

        # GoodFeaturesToTrack
        corners = cv2.goodFeaturesToTrack(roi, maxCorners=50, qualityLevel=0.01, minDistance=20)

        if corners is None:
            return None

        # 转换到全图坐标
        corners[:, 0, 1] += int(h * 0.3)
        return corners

    # ========== 增强功能 ==========

    def estimate_homography_with_quality(
        self,
        frame: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[HomographyQuality]]:
        """从当前帧估计单应矩阵（带质量评估）

        Returns:
            (H, mask, quality)
        """
        # Try YOLO keypoint detection first (if enabled)
        if self.keypoint_detector is not None:
            detections = self.keypoint_detector.detect_labeled_keypoints(frame)
            src_pts, dst_pts = self._build_semantic_correspondences(detections)
            if len(src_pts) >= 4:
                try:
                    from projection.homography import HomographyEstimator

                    estimator = HomographyEstimator()
                    H, mask, quality = estimator.estimate(src_pts, dst_pts)
                    if H is not None:
                        return H, mask, quality
                except Exception as e:
                    print(f"[FieldCalibrator] YOLO homography failed: {e}")

        # Fallback: traditional line detection
        lines = self.line_detector.detect(frame)
        corners = self.detect_corners(lines)

        if corners is None or len(corners) < 4:
            return None, None, None

        try:
            from projection.homography import HomographyEstimator

            estimator = HomographyEstimator()
            H, mask, quality = estimator.estimate(corners, DEFAULT_DST_PTS)
            return H, mask, quality
        except:
            return None, None, None

    def track_with_flow_quality(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_pts: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[HomographyQuality]]:
        """使用光流跟踪更新单应矩阵（带质量评估）

        Returns:
            (H, mask, quality)
        """
        if prev_pts is None or len(prev_pts) < 4:
            return None, None, None

        # 转灰度
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # KLT光流跟踪
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **self.lk_params
        )

        if curr_pts is None:
            return None, None, None

        # 过滤成功跟踪的点
        valid = status.flatten() == 1
        good_prev = prev_pts[valid]
        good_curr = curr_pts[valid]

        if len(good_prev) < 4:
            return None, None, None

        try:
            from projection.homography import HomographyEstimator

            estimator = HomographyEstimator()
            H, mask, quality = estimator.estimate(good_prev, good_curr)
            return H, mask, quality
        except:
            return None, None, None

    def validate_homography(self, H: np.ndarray, frame: np.ndarray) -> bool:
        """验证单应矩阵的有效性

        通过将球场角点投影回图像，检查是否在合理范围内
        """
        if H is None:
            return False

        try:
            # 将模板角点投影回图像
            template_corners = DEFAULT_DST_PTS.copy()
            template_corners = template_corners.reshape(-1, 1, 2).astype(np.float32)
            projected = cv2.perspectiveTransform(template_corners, np.linalg.inv(H))

            h, w = frame.shape[:2]

            # 检查所有角点是否在图像范围内
            for pt in projected:
                x, y = pt[0]
                if x < -w * 0.5 or x > w * 1.5 or y < -h * 0.5 or y > h * 1.5:
                    return False

            return True
        except:
            return False


class LineFallbackDetector:
    """线段检测回退检测器

    当关键点数量不足时，使用 LSD 线段检测生成额外关键点
    """

    def __init__(self):
        self.field_line_detector = FieldLineDetector()

    def detect_keypoints(
        self,
        frame: np.ndarray,
    ) -> Dict[str, Tuple[float, float, float]]:
        """从线段中提取关键点

        Returns:
            {keypoint_id: (x, y, confidence)}
        """
        lines = self.field_line_detector.detect(frame)
        if lines is None or len(lines) < 2:
            return {}

        # 收集所有端点
        points = []
        for line in lines:
            x1, y1, x2, y2 = line
            points.append((x1, y1))
            points.append((x2, y2))

        if len(points) < 4:
            return {}

        points = np.array(points)
        h, w = frame.shape[:2]

        # 基于几何特征估计关键点
        keypoints = {}

        # 1. 估计四个角点（基于图像边界）
        ys = points[:, 1]
        xs = points[:, 0]

        top_y = np.percentile(ys, 15)
        bottom_y = np.percentile(ys, 85)
        left_x = np.percentile(xs, 15)
        right_x = np.percentile(xs, 85)

        def find_closest_point(points, target_y, target_x):
            distances = (points[:, 1] - target_y) ** 2 + (points[:, 0] - target_x) ** 2
            idx = np.argmin(distances)
            return points[idx]

        # 分配角点
        corner_candidates = {
            "bottom_left_corner": find_closest_point(points, bottom_y, left_x),
            "bottom_right_corner": find_closest_point(points, bottom_y, right_x),
            "top_right_corner": find_closest_point(points, top_y, right_x),
            "top_left_corner": find_closest_point(points, top_y, left_x),
        }

        for keypoint_id, pt in corner_candidates.items():
            conf = 0.6  # 较低置信度，因为是从线段推断的
            keypoints[keypoint_id] = (float(pt[0]), float(pt[1]), conf)

        # 2. 估计中线点（基于水平线的中点）
        horizontal_lines = [line for line in lines if abs(line[3] - line[1]) < abs(line[2] - line[0])]
        if len(horizontal_lines) >= 2:
            top_line_y = min(l[1] for l in horizontal_lines[:3])
            bottom_line_y = max(l[1] for l in horizontal_lines[:3])
            mid_y = (top_line_y + bottom_line_y) / 2
            mid_x = w / 2
            keypoints["mid_top"] = (mid_x, float(top_line_y), 0.5)
            keypoints["mid_bottom"] = (mid_x, float(bottom_line_y), 0.5)

        return keypoints
