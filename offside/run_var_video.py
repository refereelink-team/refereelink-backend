from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import cv2
import numpy as np

from core import AsyncPersistence, GameStateManager
from offside.judgement import (
    check_judgement,
    compute_offside_line,
    draw_offside_on_map,
    get_attacker_defender_direction,
)
from offside.offside_core_integration import process_frame_with_core
from projection.coords import (
    FIELD_MAP_HEIGHT,
    FIELD_MAP_WIDTH,
    PITCH_BOTTOM,
    PITCH_LEFT,
    PITCH_RIGHT,
    PITCH_TOP,
    field_meter_center_to_map_pixel,
)
from projection.homography import HomographyAdapter, build_default_meter_homography
from projection.modeling import ProjectedTracklet, project_tracked_objects
from tracking.backend import (
    BALL_CLASS_ID,
    build_detector_and_tracker,
    run_detection_and_tracking,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "tracking",
    "data",
    "football-player-detection.pt",
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


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
    img = np.zeros((FIELD_MAP_HEIGHT, FIELD_MAP_WIDTH, 3), dtype=np.uint8)
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


def tracklets_to_players_map(tracklets: list, ball_as_none: bool = True) -> list:
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
    for t in tracklets:
        x1, y1, x2, y2 = t.xyxy
        is_ball = t.class_id == BALL_CLASS_ID
        color = (0, 215, 255) if is_ball else ((0, 0, 255) if t.team == "RED" else (255, 0, 0))
        if draw_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if is_ball:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 8, (0, 215, 255), -1)
            label = f"BALL#{t.track_id}"
        else:
            label = f"#{t.track_id}"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        msg, alert_color = check_judgement(t.map_x, t.map_y, t.team, is_ball=is_ball)
        if not msg:
            continue
        if msg == "OUT OF BOUNDS":
            cv2.putText(
                frame,
                f"PLAYER {t.track_id} OUT",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.circle(frame, ((x1 + x2) // 2, y2), 20, (0, 0, 255), 2)
        else:
            cv2.putText(
                frame,
                msg,
                (frame.shape[1] // 2 - 150, frame.shape[0] // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                alert_color,
                3,
            )


def draw_map_overlay(
    display_map: np.ndarray,
    tracklets: list[ProjectedTracklet],
    attacker_team: str,
    defender_team: str,
    direction: str,
) -> None:
    players_map = tracklets_to_players_map(tracklets)
    for t in tracklets:
        mx_f, my_f = field_meter_center_to_map_pixel(t.map_x, t.map_y)
        mx, my = int(mx_f), int(my_f)
        if mx < 0 or mx >= display_map.shape[1] or my < 0 or my >= display_map.shape[0]:
            continue

        if t.team == "BALL":
            cv2.circle(display_map, (mx, my), 8, (0, 215, 255), -1)
            msg, _ = check_judgement(t.map_x, t.map_y, "BALL", is_ball=True)
            if msg:
                cv2.putText(
                    display_map,
                    msg,
                    (mx + 10, my),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            continue

        color = (0, 0, 255) if t.team == "RED" else (255, 0, 0)
        cv2.circle(display_map, (mx, my), 6, color, -1)
        cv2.putText(
            display_map,
            str(t.track_id),
            (mx + 5, my),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
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
        f"[keyframe] saved frame={frame_idx}, t={timestamp_sec:.2f}s -> {os.path.abspath(meta_path)}"
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
    if homography is None:
        homography = build_default_meter_homography()
    selected_device = _resolve_device(device)

    cap = open_video_capture(video_path)
    if not cap.isOpened():
        print("Error: failed to open input video.", file=sys.stderr)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or frame_rate
    model, tracker = build_detector_and_tracker(
        model_path=model_path,
        frame_rate=int(fps),
        device=selected_device,
    )
    field_img = load_field_map(field_map_path)

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

    print("--- Offside single-frame visualization started ---")
    print(f"Input: {video_path} | Target frame: {frame_index}")

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
                f"Error: target frame {frame_index} exceeds total frames ({frame_idx}).",
                file=sys.stderr,
            )
            return

        tracked_objects = run_detection_and_tracking(
            model,
            tracker,
            target_frame,
            conf_thresh=0.22,
        )
        tracklets = project_tracked_objects(tracked_objects, homography)
        players_map = tracklets_to_players_map(tracklets)
        attacker_team, defender_team, direction = get_attacker_defender_direction(players_map)

        display_map = field_img.copy()
        draw_map_overlay(
            display_map,
            tracklets,
            attacker_team or "",
            defender_team or "",
            direction or "",
        )
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

    print(f"Completed offside single-frame visualization: frame={frame_index}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Football offside analysis - single frame visualization",
    )
    parser.add_argument("input", nargs="?", default="test.mp4", help="Input video path")
    parser.add_argument(
        "--frame_index",
        type=int,
        required=True,
        help="Target frame index (1-based)",
    )
    parser.add_argument(
        "--output_dir",
        default="offside_output",
        help="Output directory for frame/map/json results",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Player detection model path",
    )
    parser.add_argument("--no-show", action="store_true", help="Disable visualization windows")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto/cpu/cuda/mps",
    )
    parser.add_argument("--field", default="field_map.png", help="Field map path")
    parser.add_argument(
        "--state_output_path",
        default="",
        help="Optional: output path for core state CSV",
    )
    parser.add_argument(
        "--state_flush_interval",
        type=float,
        default=0.5,
        help="Async persistence flush interval (seconds)",
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
