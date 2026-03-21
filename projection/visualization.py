# -*- coding: utf-8 -*-
"""
projection 可视化能力：
- 将 tracking 输出投影到 2D 球场平面
- 导出仅包含 2D 球场视角的视频（不含越位业务判定）
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import cv2
import numpy as np

from projection.homography import HomographyAdapter, build_default_homography
from projection.modeling import ProjectedTracklet, project_tracked_objects
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
    w, h = 800, 533
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (50, 150, 50)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (255, 255, 255), 2)
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
        mx, my = int(t.map_x), int(t.map_y)
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
) -> None:
    if homography is None:
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
            tracklets = project_tracked_objects(tracked_objects, homography)

            display_map = field_img.copy()
            draw_projected_tracklets(display_map, tracklets)
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
