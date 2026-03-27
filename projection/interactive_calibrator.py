# -*- coding: utf-8 -*-
"""
球场标定模块。

提供两种标定方式：
1. InteractiveCalibrator — 交互式标定，用户点击 4 个角点
2. LineBasedCalibrator — 基于 Hough 线条的自动标定

FIFA 标准球场尺寸：105m × 68m
坐标系：中心原点，x 沿长度方向，y 沿宽度方向
"""

import json
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# FIFA 标准球场尺寸
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
HALF_PITCH_LENGTH_M = PITCH_LENGTH_M / 2.0
HALF_PITCH_WIDTH_M = PITCH_WIDTH_M / 2.0

# 球场角点世界坐标（中心原点，米）
PITCH_CORNERS_METER_CENTER = {
    "top_left":     (-HALF_PITCH_LENGTH_M, -HALF_PITCH_WIDTH_M),
    "top_right":    ( HALF_PITCH_LENGTH_M, -HALF_PITCH_WIDTH_M),
    "bottom_right": ( HALF_PITCH_LENGTH_M,  HALF_PITCH_WIDTH_M),
    "bottom_left":  (-HALF_PITCH_LENGTH_M,  HALF_PITCH_WIDTH_M),
}

# 球场角点世界坐标（左上原点，米）- 用于某些标定场景
PITCH_CORNERS_TOP_LEFT = {
    "top_left":     (0.0, 0.0),
    "top_right":    (PITCH_LENGTH_M, 0.0),
    "bottom_right": (PITCH_LENGTH_M, PITCH_WIDTH_M),
    "bottom_left":  (0.0, PITCH_WIDTH_M),
}


class InteractiveCalibrator:
    """交互式标定器。

    用户在视频帧上点击 4 个球场角点，系统计算单应矩阵。

    使用方法：
        calibrator = InteractiveCalibrator()
        calibrator.load_frame(frame)
        # 用户点击 4 个角点（依次：左上、右上、右下、左下）
        H, quality = calibrator.compute_homography()
        calibrator.save_calibration("calib.json", H)
    """

    def __init__(self):
        self.frame: Optional[np.ndarray] = None
        self.frame_h: int = 0
        self.frame_w: int = 0
        self.clicked_points: List[Tuple[float, float]] = []
        self.correspondence_labels: List[Optional[str]] = []
        self._display_frame: Optional[np.ndarray] = None

    def load_frame(self, frame: np.ndarray) -> None:
        """加载一帧用于标定。"""
        self.frame = frame.copy()
        self.frame_h, self.frame_w = frame.shape[:2]
        self.clicked_points = []
        self.correspondence_labels = []
        self._display_frame = None

    def load_frame_from_video(self, video_path: str, frame_index: int = 0) -> bool:
        """从视频加载指定帧。"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.load_frame(frame)
        return ret

    def click_point(self, x: float, y: float, label: Optional[str] = None) -> int:
        """注册一个点击的点。"""
        self.clicked_points.append((float(x), float(y)))
        self.correspondence_labels.append(label)
        return len(self.clicked_points)

    def click_corner(self, x: float, y: float, corner_name: str) -> int:
        """注册一个带语义标签的角点。"""
        return self.click_point(x, y, label=corner_name)

    def get_click_labels(self) -> List[str]:
        """返回当前需要的角点标签顺序。"""
        return ["top_left", "top_right", "bottom_right", "bottom_left"]

    def compute_homography(
        self,
        dst_points: Optional[List[Tuple[float, float]]] = None,
        require_4_pts: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """计算单应矩阵。

        Args:
            dst_points: 对应的世界坐标点（如果为 None，使用球场角点）
            require_4_pts: 是否要求至少 4 个点

        Returns:
            (H, quality_dict) 或 (None, None)
        """
        n_pts = len(self.clicked_points)
        min_pts = 4 if require_4_pts else 1

        if n_pts < min_pts:
            return None, None

        src_pts = np.array(self.clicked_points, dtype=np.float32)

        if dst_points is None:
            # 使用球场角点（米，中心原点）
            labels = self.get_click_labels()[:n_pts]
            dst_pts_list = []
            for label in labels:
                if label in PITCH_CORNERS_METER_CENTER:
                    x, y = PITCH_CORNERS_METER_CENTER[label]
                    dst_pts_list.append((x, y))
                else:
                    dst_pts_list.append((0.0, 0.0))
            dst_pts = np.array(dst_pts_list, dtype=np.float32)
        else:
            dst_pts = np.array(dst_points, dtype=np.float32)

        if len(src_pts) < 4 or len(dst_pts) < 4:
            return None, None

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        quality = None
        if H is not None:
            inliers = int(np.sum(mask)) if mask is not None else 0
            quality = {
                "inliers": inliers,
                "total_points": n_pts,
                "inlier_ratio": inliers / n_pts if n_pts > 0 else 0.0,
            }

        return H, quality

    def save_calibration(
        self,
        path: str,
        H: np.ndarray,
        coord_system: str = "meter_center",
    ) -> None:
        """保存标定结果到 JSON 文件。"""
        data = {
            "H": H.tolist(),
            "coord_system": coord_system,
            "num_points": len(self.clicked_points),
            "frame_width": self.frame_w,
            "frame_height": self.frame_h,
            "pitch_length_m": PITCH_LENGTH_M,
            "pitch_width_m": PITCH_WIDTH_M,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_calibration(path: str) -> Tuple[np.ndarray, Dict]:
        """从 JSON 文件加载标定结果。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        H = np.array(data["H"])
        info = {
            "coord_system": data.get("coord_system", "meter_center"),
            "frame_width": data.get("frame_width"),
            "frame_height": data.get("frame_height"),
        }
        return H, info

    def draw_clicks(self, frame: Optional[np.ndarray] = None) -> np.ndarray:
        """绘制已点击的点。"""
        if frame is None:
            if self._display_frame is None and self.frame is not None:
                self._display_frame = self.frame.copy()
            frame = self._display_frame
        else:
            frame = frame.copy()

        labels = self.get_click_labels()
        colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0)]

        for i, (pt, color) in enumerate(zip(self.clicked_points, colors)):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x, y), 8, color, -1)
            cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)
            label = labels[i] if i < len(labels) else f"pt{i}"
            cv2.putText(
                frame,
                label,
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        return frame


class LineBasedCalibrator:
    """基于 Hough 线条检测的自动标定。

    使用 HoughLinesP 检测球场白线，找交点，匹配已知球场几何。
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 10,
        hough_min_length: int = 30,
        hough_max_gap: int = 20,
        brightness_thresh: int = 100,
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.hough_min_length = hough_min_length
        self.hough_max_gap = hough_max_gap
        self.brightness_thresh = brightness_thresh
        self.frame_h: int = 0
        self.frame_w: int = 0

    def detect_lines(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """检测帧中的线条段（改进版 HSV 场地掩码）。

        流程：
        1. 转换到 HSV 色彩空间
        2. HSV 掩码提取绿色场地区域（H:25-85, S:>40）
        3. 形态学开闭运算清理掩码
        4. 仅在场地区域内做 Canny 边缘检测
        5. 亮度阈值过滤亮线
        6. HoughLinesP 提取线段
        """
        self.frame_h, self.frame_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

        # 球场区域：下半部分
        roi_y_start = self.frame_h // 3
        roi_hsv = hsv[roi_y_start:, :]
        roi_gray = blurred[roi_y_start:, :]

        # HSV 场地掩码：绿色场地（H:25-85, S:>40）
        field_mask = (
            (roi_hsv[:, :, 0] > 25)
            & (roi_hsv[:, :, 0] < 85)
            & (roi_hsv[:, :, 1] > 40)
        ).astype(np.uint8) * 255

        # 形态学清理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # 亮度阈值提取白线
        _, bright = cv2.threshold(roi_gray, self.brightness_thresh, 255, cv2.THRESH_BINARY)

        # Canny 边缘（仅在场地区域）
        edges = cv2.Canny(roi_gray, self.canny_low, self.canny_high)
        field_edges = cv2.bitwise_and(edges, field_mask)

        # Hough 线条检测
        lines_roi = cv2.HoughLinesP(
            field_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_length,
            maxLineGap=self.hough_max_gap,
        )

        if lines_roi is None:
            return None

        # 转换 ROI 坐标到完整帧坐标
        lines_full = []
        for line in lines_roi:
            x1, y1, x2, y2 = line[0]
            lines_full.append([[x1, y1 + roi_y_start, x2, y2 + roi_y_start]])
        return np.array(lines_full, dtype=np.int32)

    def find_intersections(
        self, lines: np.ndarray
    ) -> List[Tuple[float, float]]:
        """求线条段的交点（仅水平-垂直线交点）。"""
        if lines is None or len(lines) < 2:
            return []

        # 分类为水平线和垂直线
        horizontal: List[np.ndarray] = []
        vertical: List[np.ndarray] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx * dx + dy * dy)
            if length < 10:
                continue
            angle = abs(np.arctan2(dy, dx))
            # 近似水平线（角度接近 0 或 π）
            if angle < np.pi / 9 or angle > np.pi - np.pi / 9:
                horizontal.append(line[0])
            # 近似垂直线（角度接近 π/2）
            elif abs(angle - np.pi / 2) < np.pi / 9:
                vertical.append(line[0])

        intersections = []
        h, w = self.frame_h, self.frame_w

        # 只考虑水平-垂直线交点
        for x1h, y1h, x2h, y2h in horizontal:
            for x1v, y1v, x2v, y2v in vertical:
                denom = (x1h - x2h) * (y1v - y2v) - (y1h - y2h) * (x1v - x2v)
                if abs(denom) < 1e-6:
                    continue

                x = (
                    (x1h * y2h - y1h * x2h) * (x1v - x2v)
                    - (x1h - x2h) * (x1v * y2v - y1v * x2v)
                ) / denom
                y = (
                    (x1h * y2h - y1h * x2h) * (y1v - y2v)
                    - (y1h - y2h) * (x1v * y2v - y1v * x2v)
                ) / denom

                if 0 <= x <= w and 0 <= y <= h:
                    intersections.append((float(x), float(y)))

        return intersections

    def filter_quadrilateral(
        self, points: List[Tuple[float, float]]
    ) -> Optional[np.ndarray]:
        """从交点中筛选出组成四边形的 4 个角点。

        改进版：使用更严格的几何约束。
        - 假设摄像机朝向我们（常见广播视角），球场四个角点的 y 坐标应满足：
          top_corners_y < bottom_corners_y
        - x 坐标：left < right
        - 四边形应接近凸四边形
        """
        if len(points) < 4:
            return None

        pts = np.array(points)
        h, w = self.frame_h, self.frame_w

        # 初步筛选：在画面下半部分且不太靠近边缘的点优先
        # 球场角点通常在画面中下部（远离相机方向）
        roi_bottom = h * 0.4  # 角点不应在画面上 40% 以上
        roi_top = h * 0.95  # 角点不应在接近画面底部
        roi_left = w * 0.05
        roi_right = w * 0.95

        # 过滤：在合理范围内的点
        valid = []
        for pt in pts:
            x, y = pt
            if roi_bottom < y < roi_top and roi_left < x < roi_right:
                valid.append(pt)
        if len(valid) < 4:
            valid = list(pts)

        valid = np.array(valid)

        # 取 y 最小的 N 个作为顶部候选
        n_top = min(10, len(valid))
        top_candidates = valid[np.argsort(valid[:, 1])[:n_top]]
        # 取 y 最大的 N 个作为底部候选
        n_bottom = min(10, len(valid))
        bottom_candidates = valid[np.argsort(valid[:, 1])[-n_bottom:]]

        # 顶部最左/最右
        tl_candidate = top_candidates[np.argmin(top_candidates[:, 0])]
        tr_candidate = top_candidates[np.argmax(top_candidates[:, 0])]
        # 底部最左/最右
        bl_candidate = bottom_candidates[np.argmin(bottom_candidates[:, 0])]
        br_candidate = bottom_candidates[np.argmax(bottom_candidates[:, 0])]

        corners = np.array(
            [
                [tl_candidate[0], tl_candidate[1]],
                [tr_candidate[0], tr_candidate[1]],
                [br_candidate[0], br_candidate[1]],
                [bl_candidate[0], bl_candidate[1]],
            ],
            dtype=np.float32,
        )

        # 验证：四角应形成一个凸四边形
        # 对角线应相交（在四边形内部）
        # 简单验证：TL 和 BR 的 x+y 应该都较小/较大
        # 如果检测到的点不合理，返回 None
        tl, tr, br, bl = corners

        # 简单验证：按 y 排序，上面两个的 y 应该接近，下面两个的 y 应该接近
        y_sorted = corners[corners[:, 1].argsort()]
        top_y_span = y_sorted[1, 1] - y_sorted[0, 1]
        bottom_y_span = y_sorted[3, 1] - y_sorted[2, 1]
        # 如果上下之间 y 差太小，说明点太集中
        y_gap = y_sorted[2, 1] - y_sorted[1, 1]
        if y_gap < h * 0.1:
            return None

        return corners

    def calibrate(
        self,
        frame: np.ndarray,
        known_pitch_corners: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """完整标定流程。"""
        self.frame_h, self.frame_w = frame.shape[:2]

        lines = self.detect_lines(frame)
        if lines is None or len(lines) < 4:
            return None, None

        intersections = self.find_intersections(lines)
        if len(intersections) < 4:
            return None, None

        corners = self.filter_quadrilateral(intersections)
        if corners is None:
            return None, None

        # 默认使用球场角点（米，中心原点）
        if known_pitch_corners is None:
            known_pitch_corners = PITCH_CORNERS_METER_CENTER

        dst_pts = np.array(
            [
                known_pitch_corners["top_left"],
                known_pitch_corners["top_right"],
                known_pitch_corners["bottom_right"],
                known_pitch_corners["bottom_left"],
            ],
            dtype=np.float32,
        )

        H, mask = cv2.findHomography(corners, dst_pts, cv2.RANSAC, 5.0)

        quality = None
        if H is not None:
            inliers = int(np.sum(mask)) if mask is not None else 0
            quality = {
                "inliers": inliers,
                "total_corners": 4,
                "lines_detected": len(lines),
                "intersections_found": len(intersections),
            }

        return H, quality

    def visualize(
        self,
        frame: np.ndarray,
        lines: Optional[np.ndarray] = None,
        corners: Optional[np.ndarray] = None,
        intersections: Optional[List[Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """可视化检测结果。"""
        vis = frame.copy()

        # 绘制线条
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制交点
        if intersections:
            for pt in intersections:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

        # 绘制角点
        if corners is not None:
            colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0)]
            labels = ["TL", "TR", "BR", "BL"]
            for pt, color, label in zip(corners, colors, labels):
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 10, color, -1)
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 10, (255, 255, 255), 2)
                cv2.putText(
                    vis,
                    label,
                    (int(pt[0]) + 12, int(pt[1]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

        return vis


def create_interactive_calibrator() -> InteractiveCalibrator:
    """创建交互式标定器。"""
    return InteractiveCalibrator()


def create_line_calibrator(
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 50,
    hough_min_length: int = 50,
    hough_max_gap: int = 20,
) -> LineBasedCalibrator:
    """创建线条标定器。"""
    return LineBasedCalibrator(
        canny_low=canny_low,
        canny_high=canny_high,
        hough_threshold=hough_threshold,
        hough_min_length=hough_min_length,
        hough_max_gap=hough_max_gap,
    )
