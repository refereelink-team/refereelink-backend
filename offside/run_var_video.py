# -*- coding: utf-8 -*-
"""
足球越位分析 - 单帧可视化入口。
- 消费 tracking + projection 结果
- 在指定关键帧做越位/出界判定并输出单帧可视化
"""
import os
import sys
import json
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "tracking", "data", "football-player-detection.pt"
)

# 避免 OpenMP 重复库警告
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from core import AsyncPersistence, GameStateManager

from projection.homography import HomographyAdapter, build_default_homography
from projection.modeling import ProjectedTracklet, project_tracked_objects
from tracking.backend import (
    BALL_CLASS_ID,
    build_detector_and_tracker,
    run_detection_and_tracking,
)
from offside.judgement import (
    check_judgement,
    get_attacker_defender_direction,
    draw_offside_on_map,
    compute_offside_line,
)
from offside.offside_core_integration import process_frame_with_core


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
    """加载战术板底图；若不存在则生成简易绿底。"""
    img = cv2.imread(field_path)
    if img is not None:
        return img
    # 与 judgement 中 FIELD_RECT 比例接近的简易图
    w, h = 800, 533
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (50, 150, 50)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (255, 255, 255), 2)
    return img


def open_video_capture(source: str):
    """尽量兼容多种视频源：先尝试 FFMPEG 后端。"""
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(source)
    return cap


def tracklets_to_players_map(tracklets: list, ball_as_none: bool = True) -> list:
    """将 Tracklet 列表转为 judgement 使用的 players_map（仅人，不含球）。"""
    return [
        {"id": t.track_id, "team": t.team, "x": t.map_x, "y": t.map_y}
        for t in tracklets
        if t.team in ("RED", "BLUE")
    ]


def draw_tracklets_and_judgement(
    frame: np.ndarray,
    tracklets: list[ProjectedTracklet],
    draw_boxes: bool = True,
) -> None:
    """在单帧图像上绘制 bbox、track_id、出界/进球提示。原地修改 frame。"""
    for t in tracklets:
        x1, y1, x2, y2 = t.xyxy
        is_ball = t.class_id == BALL_CLASS_ID
        color = (0, 215, 255) if is_ball else ((0, 0, 255) if t.team == "RED" else (255, 0, 0))
        if draw_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if is_ball:
            # 足球：标得更明显一点
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 8, (0, 215, 255), -1)
            label = f"BALL#{t.track_id}"
        else:
            label = f"#{t.track_id}"
        cv2.putText(
            frame, label, (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        msg, alert_color = check_judgement(t.map_x, t.map_y, t.team, is_ball=is_ball)
        if msg:
            if msg == "OUT OF BOUNDS":
                cv2.putText(
                    frame, f"PLAYER {t.track_id} OUT", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                )
                cv2.circle(frame, ((x1 + x2) // 2, y2), 20, (0, 0, 255), 2)
            else:
                cv2.putText(
                    frame, msg, (frame.shape[1] // 2 - 150, frame.shape[0] // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, alert_color, 3,
                )


def draw_map_overlay(
    display_map: np.ndarray,
    tracklets: list[ProjectedTracklet],
    attacker_team: str,
    defender_team: str,
    direction: str,
) -> None:
    """在战术板上画球员/球点、出界圈、越位线与越位标记。"""
    players_map = tracklets_to_players_map(tracklets)
    for t in tracklets:
        mx, my = int(t.map_x), int(t.map_y)
        if t.team == "BALL":
            cv2.circle(display_map, (mx, my), 8, (0, 215, 255), -1)
            msg, _ = check_judgement(t.map_x, t.map_y, "BALL", is_ball=True)
            if msg:
                cv2.putText(display_map, msg, (mx + 10, my), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            color = (0, 0, 255) if t.team == "RED" else (255, 0, 0)
            cv2.circle(display_map, (mx, my), 6, color, -1)
            cv2.putText(display_map, str(t.track_id), (mx + 5, my), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            msg, _ = check_judgement(t.map_x, t.map_y, t.team, is_ball=False)
            if msg == "OUT OF BOUNDS":
                cv2.circle(display_map, (mx, my), 15, (0, 0, 255), 2)
    if attacker_team and defender_team and direction:
        draw_offside_on_map(display_map, players_map, attacker_team, defender_team, direction)


def save_keyframe_decision(
    frame_idx: int,
    timestamp_sec: float,
    video_path: str,
    frame_bgr: np.ndarray,
    display_map: np.ndarray,
    tracklets: list[ProjectedTracklet],
    attacker_team: str,
    defender_team: str,
    direction: str,
    output_dir: str = "keyframes",
) -> None:
    """
    保存“关键帧判罚”结果：
    - 单帧画面截图（带标注）
    - 战术板截图
    - 一份 JSON 元数据
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = f"keyframe_{frame_idx:06d}"

    frame_path = os.path.join(output_dir, base_name + "_frame.jpg")
    map_path = os.path.join(output_dir, base_name + "_map.jpg")
    meta_path = os.path.join(output_dir, base_name + ".json")

    cv2.imwrite(frame_path, frame_bgr)
    cv2.imwrite(map_path, display_map)

    players_map = tracklets_to_players_map(tracklets)
    offside_x = None
    offside_players = []

    if attacker_team and defender_team and direction:
        offside_x = compute_offside_line(players_map, defender_team, direction)
        if offside_x is not None:
            if direction == "LEFT":
                is_offside = lambda x: x < offside_x
            else:
                is_offside = lambda x: x > offside_x
            for t in tracklets:
                if t.team == attacker_team and is_offside(t.map_x):
                    offside_players.append(int(t.track_id))

    # 拆分球员与足球，方便前端使用
    players_payload = []
    ball_payload = None
    for t in tracklets:
        payload = {
            "track_id": int(t.track_id),
            "class_id": int(t.class_id),
            "team": t.team,
            "pixel_xyxy": list(map(int, t.xyxy)),
            "map_xy": [float(t.map_x), float(t.map_y)],
            "is_offside": attacker_team
            and offside_x is not None
            and t.team == attacker_team
            and (
                (direction == "LEFT" and t.map_x < offside_x)
                or (direction == "RIGHT" and t.map_x > offside_x)
            ),
        }
        if t.team == "BALL":
            ball_payload = payload
        else:
            players_payload.append(payload)

    meta = {
        "video": os.path.basename(video_path),
        "frame_index": int(frame_idx),
        "timestamp_sec": float(timestamp_sec) if timestamp_sec is not None else None,
        "frame_image": os.path.abspath(frame_path),
        "map_image": os.path.abspath(map_path),
        "attacker_team": attacker_team,
        "defender_team": defender_team,
        "direction": direction,
        "offside_line_x": float(offside_x) if offside_x is not None else None,
        "offside_players": offside_players,
        "players": players_payload,
        "ball": ball_payload,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"[关键帧] 已保存 frame={frame_idx}, t={timestamp_sec:.2f}s "
        f"-> {os.path.abspath(meta_path)}"
    )


def run_single_frame_pipeline(
    video_path: str,
    frame_index: int,
    output_dir: str = "offside_output",
    model_path: str = DEFAULT_MODEL_PATH,
    homography: Optional[HomographyAdapter] = None,
    field_map_path: str = "field_map.png",
    show_live: bool = True,
    frame_rate: int = 30,
    game_state: Optional[GameStateManager] = None,
    state_source: str = "OFFSIDE_MODULE",
    emit_offside_query: bool = True,
    state_output_path: str = "",
    state_flush_interval: float = 0.5,
    device: str = "cpu",
) -> None:
    """
    主流程：读取视频到目标帧 -> tracking -> projection -> offside 判定 -> 输出单帧结果。
    """
    if homography is None:
        homography = build_default_homography()
    selected_device = _resolve_device(device)

    cap = open_video_capture(video_path)
    if not cap.isOpened():
        print("错误：无法打开视频文件，请检查路径与格式。", file=sys.stderr)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or frame_rate

    model, tracker = build_detector_and_tracker(
        model_path=model_path,
        frame_rate=int(fps),
        device=selected_device,
    )
    field_img = load_field_map(field_map_path)

    # 可选：若未显式传入 game_state，但提供了 state_output_path，则按 tracking/main.py 的约定创建
    persistence: Optional[AsyncPersistence] = None
    owns_game_state = False
    if game_state is None and state_output_path:
        game_state = GameStateManager()
        persistence = AsyncPersistence(
            game_state=game_state,
            output_path=state_output_path,
            flush_interval=state_flush_interval,
        )
        persistence.start()
        owns_game_state = True

    print("--- 越位单帧可视化已启动 ---")
    print(f"输入: {video_path} | 目标帧: {frame_index}")

    frame_idx = 0
    target_frame: Optional[np.ndarray] = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx == frame_index:
                target_frame = frame.copy()
                break

        if target_frame is None:
            print(
                f"错误：目标帧 {frame_index} 超出视频总帧数（总计 {frame_idx}）。",
                file=sys.stderr,
            )
            return

        tracked_objects = run_detection_and_tracking(
            model, tracker, target_frame, conf_thresh=0.22
        )
        tracklets = project_tracked_objects(tracked_objects, homography)
        players_map = tracklets_to_players_map(tracklets)
        attacker_team, defender_team, direction = get_attacker_defender_direction(players_map)

        display_map = field_img.copy()
        draw_map_overlay(display_map, tracklets, attacker_team or "", defender_team or "", direction or "")
        draw_tracklets_and_judgement(target_frame, tracklets, draw_boxes=True)

        if game_state is not None:
            video_ts = frame_index / float(fps) if fps else None
            process_frame_with_core(
                frame_id=frame_index,
                video_ts=video_ts,
                source=state_source,
                tracklets=tracklets,
                game_state=game_state,
                emit_offside_query=emit_offside_query,
            )

        timestamp_sec = frame_index / float(fps) if fps else 0.0
        save_keyframe_decision(
            frame_idx=frame_index,
            timestamp_sec=timestamp_sec,
            video_path=video_path,
            frame_bgr=target_frame.copy(),
            display_map=display_map.copy(),
            tracklets=tracklets,
            attacker_team=attacker_team or "",
            defender_team=defender_team or "",
            direction=direction or "",
            output_dir=output_dir,
        )

        if show_live:
            cv2.imshow("Offside Frame", target_frame)
            cv2.imshow("Offside Tactical Map", display_map)
            cv2.waitKey(0)
    finally:
        cap.release()
        if show_live:
            cv2.destroyAllWindows()
        if persistence is not None and owns_game_state:
            persistence.stop()
            persistence.join()
    print(f"已完成单帧越位可视化，frame={frame_index}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="足球越位分析 - 单帧可视化")
    parser.add_argument("input", nargs="?", default="test.mp4", help="输入视频路径")
    parser.add_argument(
        "--frame_index",
        type=int,
        required=True,
        help="要分析的关键帧（从 1 开始）",
    )
    parser.add_argument(
        "--output_dir",
        default="offside_output",
        help="单帧结果输出目录（保存 frame/map/json）",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="足球检测模型权重路径（默认使用 tracking/data/football-player-detection.pt）",
    )
    parser.add_argument("--no-show", action="store_true", help="不显示可视化窗口")
    parser.add_argument(
        "--device",
        default="auto",
        help="推理设备：auto/cpu/cuda/mps（auto 按 cuda>mps>cpu）",
    )
    parser.add_argument("--field", default="field_map.png", help="战术板底图路径")
    parser.add_argument(
        "--state_output_path",
        default="",
        help="可选，启用 core 状态 CSV 导出（仅写关键帧）",
    )
    parser.add_argument(
        "--state_flush_interval",
        type=float,
        default=0.5,
        help="core 异步刷盘间隔（秒）",
    )
    args = parser.parse_args()
    run_single_frame_pipeline(
        args.input,
        frame_index=args.frame_index,
        output_dir=args.output_dir,
        model_path=args.model,
        field_map_path=args.field,
        show_live=not args.no_show,
        state_output_path=args.state_output_path,
        state_flush_interval=args.state_flush_interval,
        device=args.device,
    )
