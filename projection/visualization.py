# -*- coding: utf-8 -*-
"""
projection 可视化能力：
- 将 tracking 输出投影到 2D 球场平面
- 导出仅包含 2D 球场视角的视频（不含越位业务判定）
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, Optional

import cv2
import numpy as np

from projection.coords import (
    FIELD_MAP_HEIGHT,
    FIELD_MAP_WIDTH,
    PITCH_BOTTOM,
    PITCH_LEFT,
    PITCH_RIGHT,
    PITCH_TOP,
    field_meter_center_to_map_pixel,
)
from projection.homography import HomographyAdapter, build_default_homography
from projection.modeling import ProjectedTracklet, project_tracked_objects
from projection.sn_projection_backend import create_projection_engine
from core import ObjectTrack, ProjectedObject
from tracking.backend import (
    BALL_CLASS_ID,
    build_detector_and_tracker,
    run_detection_and_tracking,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "tracking", "data", "football-player-detection.pt"
)


def _resolve_device(device: str = "auto") -> str:
    normalized = (device or "auto").strip().lower()
    if normalized != "auto":
        return normalized

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def load_field_map(field_path: str = "field_map.png") -> np.ndarray:
    img = cv2.imread(field_path)
    if img is not None:
        return img
    w, h = FIELD_MAP_WIDTH, FIELD_MAP_HEIGHT
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (50, 150, 50)
    cv2.rectangle(
        img,
        (int(PITCH_LEFT), int(PITCH_TOP)),
        (int(PITCH_RIGHT), int(PITCH_BOTTOM)),
        (255, 255, 255),
        2,
    )
    return img


def open_video_capture(source: str):
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(source)
    return cap


def draw_projected_tracklets(
    display_map: np.ndarray,
    tracklets: list[ProjectedTracklet],
) -> None:
    for t in tracklets:
        mx_f, my_f = field_meter_center_to_map_pixel(t.map_x, t.map_y)
        mx, my = int(mx_f), int(my_f)
        if mx < 0 or mx >= display_map.shape[1] or my < 0 or my >= display_map.shape[0]:
            continue
        if t.class_id == BALL_CLASS_ID or t.team == "BALL":
            cv2.circle(display_map, (mx, my), 8, (0, 215, 255), -1)
            cv2.putText(
                display_map,
                "BALL",
                (mx + 8, my - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            continue

        color = (0, 0, 255) if t.team == "RED" else (255, 0, 0)
        if t.team == "WHITE":
            color = (220, 220, 220)
        cv2.circle(display_map, (mx, my), 6, color, -1)
        cv2.putText(
            display_map,
            f"#{t.track_id}",
            (mx + 5, my),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )


# 颜色定义 for keypoint visualization
KEYPOINT_COLORS = {
    # 角点 - 红色
    "top_left_corner": (0, 0, 255),
    "top_right_corner": (0, 0, 255),
    "bottom_left_corner": (0, 0, 255),
    "bottom_right_corner": (0, 0, 255),
    # 禁区角点 - 黄色
    "penalty_top_left": (0, 255, 255),
    "penalty_top_right": (0, 255, 255),
    "penalty_bottom_left": (0, 255, 255),
    "penalty_bottom_right": (0, 255, 255),
    # 中线点 - 绿色
    "mid_top": (0, 255, 0),
    "mid_bottom": (0, 255, 0),
    "mid_left": (0, 255, 0),
    "mid_right": (0, 255, 0),
}


def draw_keypoints_on_frame(
    frame: np.ndarray,
    keypoints: dict,
    show_labels: bool = True,
    radius: int = 8,
) -> np.ndarray:
    """在原始视频帧上绘制检测到的球场关键点

    Args:
        frame: BGR 视频帧
        keypoints: {keypoint_id: TrackedKeypoint}
        show_labels: 是否显示关键点标签
        radius: 绘制半径

    Returns:
        绘制了关键点的帧
    """
    display = frame.copy()

    for keypoint_id, kp in keypoints.items():
        if not kp.visible:
            continue

        x, y = int(kp.pt[0]), int(kp.pt[1])

        # 确保在帧范围内
        if x < 0 or x >= display.shape[1] or y < 0 or y >= display.shape[0]:
            continue

        # 获取颜色
        color = KEYPOINT_COLORS.get(keypoint_id, (255, 255, 255))

        # 绘制圆圈
        cv2.circle(display, (x, y), radius, color, -1)

        # 绘制边框
        cv2.circle(display, (x, y), radius, (255, 255, 255), 2)

        if show_labels:
            # 绘制标签
            label = keypoint_id.replace("_", " ")
            cv2.putText(
                display,
                label,
                (x + radius + 4, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )

    return display


def draw_keypoints_on_field(
    field_img: np.ndarray,
    keypoints: dict,
    homography: Optional[HomographyAdapter] = None,
    show_labels: bool = True,
    radius: int = 8,
) -> np.ndarray:
    """在2D球场图上绘制关键点的模板位置（用于调试对应关系）

    Args:
        field_img: 2D球场图像
        keypoints: {keypoint_id: TrackedKeypoint} - 图像上的关键点
        homography: 单应矩阵（用于将关键点投影到球场）
        show_labels: 是否显示关键点标签
        radius: 绘制半径

    Returns:
        绘制了关键点的球场图
    """
    display = field_img.copy()

    # 绘制模板关键点位置（固定位置）
    from projection.homography import template_to_image_points
    template_pts = template_to_image_points()

    for keypoint_id, template_pt in template_pts.items():
        x, y = int(template_pt[0]), int(template_pt[1])

        # 确保在球场图范围内
        if x < 0 or x >= display.shape[1] or y < 0 or y >= display.shape[0]:
            continue

        color = KEYPOINT_COLORS.get(keypoint_id, (255, 255, 255))

        # 绘制圆圈（空心，表示模板位置）
        cv2.circle(display, (x, y), radius, color, 2)

        if show_labels:
            label = keypoint_id.replace("_", " ")
            cv2.putText(
                display,
                label,
                (x + radius + 4, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )

    # 如果有 homography，可以额外绘制投影后的关键点位置
    if homography is not None:
        for keypoint_id, kp in keypoints.items():
            if not kp.visible:
                continue

            try:
                field_x, field_y = homography.pixel_to_field_meters(kp.pt[0], kp.pt[1])
                map_x, map_y = field_meter_center_to_map_pixel(field_x, field_y)
                mx, my = int(map_x), int(map_y)

                if 0 <= mx < display.shape[1] and 0 <= my < display.shape[0]:
                    # 绘制实心圆，表示投影位置
                    color = KEYPOINT_COLORS.get(keypoint_id, (255, 255, 255))
                    cv2.circle(display, (mx, my), radius - 2, color, -1)
            except Exception:
                pass

    return display


def build_projected_objects(
    tracked_objects: list[ObjectTrack],
    homography: Optional[HomographyAdapter] = None,
) -> list[ProjectedObject]:
    if homography is None:
        homography = build_default_homography()

    tracklets = project_tracked_objects(tracked_objects, homography)
    return [
        ProjectedObject(
            track_id=int(t.track_id),
            class_id=int(t.class_id),
            xyxy=t.xyxy,
            confidence=float(t.confidence),
            map_x=float(t.map_x),
            map_y=float(t.map_y),
            team=t.team,
        )
        for t in tracklets
    ]


def render_projection_frame(
    tracked_objects: list[ObjectTrack],
    field_img: np.ndarray,
    homography: Optional[HomographyAdapter] = None,
) -> np.ndarray:
    if homography is None:
        homography = build_default_homography()

    display_map = field_img.copy()
    tracklets = build_projected_objects(tracked_objects, homography)
    draw_projected_tracklets(display_map, tracklets)
    return display_map


def run_projection_video_pipeline_from_tracks(
    tracked_objects_stream: Iterable[list[ObjectTrack]],
    output_path: str,
    fps: float,
    homography: Optional[HomographyAdapter] = None,
    field_map_path: str = "field_map.png",
    show_live: bool = True,
) -> None:
    if homography is None:
        homography = build_default_homography()

    field_img = load_field_map(field_map_path)
    map_h, map_w = field_img.shape[:2]

    writer = None
    for fourcc_name in ("mp4v", "XVID", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        w = cv2.VideoWriter(output_path, fourcc, fps, (map_w, map_h))
        if w.isOpened():
            writer = w
            break
    if writer is None or not writer.isOpened():
        print("错误：无法创建输出视频，请检查路径与编码器。", file=sys.stderr)
        return

    frame_idx = 0
    try:
        for tracked_objects in tracked_objects_stream:
            frame_idx += 1
            display_map = render_projection_frame(tracked_objects, field_img, homography)
            writer.write(display_map)
            if show_live:
                cv2.imshow("Projection 2D Map", display_map)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        writer.release()
        if show_live:
            cv2.destroyWindow("Projection 2D Map")

    print(f"已写入 {output_path}，共 {frame_idx} 帧。")


def run_projection_video_pipeline(
    video_path: str,
    output_path: str = "projection_2d.mp4",
    model_path: str = DEFAULT_MODEL_PATH,
    homography: Optional[HomographyAdapter] = None,
    field_map_path: str = "field_map.png",
    show_live: bool = True,
    frame_rate: int = 30,
    conf_thresh: float = 0.22,
    device: str = "auto",
    dynamic: bool = False,
    recalib_interval: int = 10,
    calibration: str = "",
    calib_backend: str = "nbjw",
    use_prev_homography: bool = True,
    debug: bool = False,
) -> None:
    if homography is None and not calibration:
        homography = build_default_homography()
    selected_device = _resolve_device(device)

    cap = open_video_capture(video_path)
    if not cap.isOpened():
        print("错误：无法打开视频文件，请检查路径与格式。", file=sys.stderr)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or frame_rate
    field_img = load_field_map(field_map_path)
    map_h, map_w = field_img.shape[:2]

    writer = None
    for fourcc_name in ("mp4v", "XVID", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        w = cv2.VideoWriter(output_path, fourcc, fps, (map_w, map_h))
        if w.isOpened():
            writer = w
            break
    if writer is None or not writer.isOpened():
        print("错误：无法创建输出视频，请检查路径与编码器。", file=sys.stderr)
        cap.release()
        return

    model, tracker = build_detector_and_tracker(
        model_path=model_path,
        frame_rate=int(fps),
        device=selected_device,
    )
    engine = create_projection_engine(
        calib_backend=calib_backend,
        dynamic=dynamic or calib_backend in {"nbjw", "pnl"},
        recalib_interval=recalib_interval,
        calibration_path=calibration,
        field_path=field_map_path,
        debug=debug,
        use_prev_homography=use_prev_homography,
    )

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            tracked_objects = run_detection_and_tracking(
                model, tracker, frame, conf_thresh=conf_thresh
            )
            H_adapter = homography if homography is not None else engine.update(frame)
            display_map = render_projection_frame(tracked_objects, field_img, H_adapter)
            writer.write(display_map)

            if show_live:
                cv2.imshow("Projection 2D Map", display_map)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        if show_live:
            cv2.destroyAllWindows()

    print(f"已写入 {output_path}，共 {frame_idx} 帧。")
