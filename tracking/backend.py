# -*- coding: utf-8 -*-
"""
tracking 底座能力：
- YOLO 检测 + ByteTrack 追踪
- 输出像素空间的稳定 tracklet（不做投影）
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# 自定义足球检测模型类别约定：
# 0=球（ball），1=守门员（goalkeeper），2=球员（player），3=裁判（referee）
BALL_CLASS_ID = 0
HUMAN_CLASS_IDS = (1, 2, 3)


@dataclass
class TrackedObject:
    """单帧中一个被追踪目标（像素空间）。"""

    track_id: int
    class_id: int
    xyxy: Tuple[int, int, int, int]
    confidence: float
    team: str  # "RED" | "BLUE" | "BALL" | "WHITE"


def get_player_team(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> str:
    """根据 bbox 上半身裁剪的 HSV 色调简单区分红/蓝。"""
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return "WHITE"
    crop = frame[y1 : y1 + (y2 - y1) // 2, x1:x2]
    if crop.size == 0:
        return "WHITE"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    avg_h = np.mean(hsv[:, :, 0])
    return "RED" if avg_h < 60 else "BLUE"


def run_detection_and_tracking(
    model: YOLO,
    tracker: sv.ByteTrack,
    frame: np.ndarray,
    conf_thresh: float = 0.08,
    classes: Optional[List[int]] = None,
) -> List[TrackedObject]:
    """对单帧做 YOLO+ByteTrack，输出像素空间 tracklet。"""
    if classes is None:
        classes = [BALL_CLASS_ID, *HUMAN_CLASS_IDS]

    results = model.predict(
        frame,
        imgsz=1280,
        conf=conf_thresh,
        classes=classes,
        verbose=False,
    )[0]

    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    if detections.tracker_id is None:
        return []

    xyxy = detections.xyxy
    tracker_ids = detections.tracker_id
    class_ids = detections.class_id
    confidences = detections.confidence

    tracked: List[TrackedObject] = []
    for i in range(len(detections)):
        tid = int(tracker_ids[i])
        cls = int(class_ids[i])
        box = xyxy[i]
        conf = float(confidences[i]) if confidences is not None else 0.0
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        team = "BALL" if cls == BALL_CLASS_ID else get_player_team(frame, x1, y1, x2, y2)
        tracked.append(
            TrackedObject(
                track_id=tid,
                class_id=cls,
                xyxy=(x1, y1, x2, y2),
                confidence=conf,
                team=team,
            )
        )
    return tracked


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "tracking", "data", "football-player-detection.pt")


def build_detector_and_tracker(
    model_path: str = DEFAULT_MODEL_PATH,
    track_thresh: float = 0.12,
    track_buffer: int = 50,
    match_thresh: float = 0.8,
    frame_rate: int = 30,
    device: str = "cpu",
) -> Tuple[YOLO, sv.ByteTrack]:
    """构建 YOLO 检测器与 ByteTrack 追踪器。"""
    model = YOLO(model_path).to(device=device)
    tracker = sv.ByteTrack(
        track_activation_threshold=track_thresh,
        lost_track_buffer=track_buffer,
        minimum_matching_threshold=match_thresh,
        frame_rate=frame_rate,
    )
    return model, tracker
