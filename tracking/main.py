import argparse
import math
import os
import sys
import time
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core import AsyncPersistence, FramePacket, GameStateManager, ObjectTrack, PlayerState, Team

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, "data/football-player-detection.pt")

GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

TEAM_0_COLOR_ID = 0
TEAM_1_COLOR_ID = 1
GOALKEEPER_COLOR_ID = 2
REFEREE_COLOR_ID = 3

CROP_SAMPLE_RATIO = 0.10
MIN_CROP_SAMPLES = 48
MAX_CROP_SAMPLES = 240
CROPS_COLLECTION_END = 240

COLORS = ["#FF1493", "#00BFFF", "#FF6347", "#FFD700"]
MODE_NAME = "PLAYER_TEAM_CLASSIFICATION"

FrameResult = Tuple[np.ndarray, Dict[int, PlayerState]]


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


def _resolve_player_id(detections, index: int) -> int:
    if detections.tracker_id is None:
        return index
    tracker_id = detections.tracker_id[index]
    if tracker_id is None:
        return index
    return int(tracker_id)


def _team_from_color(color_id: int, class_id: int) -> Team:
    if class_id == REFEREE_CLASS_ID:
        return Team.REFEREE
    if color_id == TEAM_0_COLOR_ID:
        return Team.HOME
    if color_id == TEAM_1_COLOR_ID:
        return Team.AWAY
    return Team.UNKNOWN


def _tracked_team_name(team: Team) -> str:
    if team == Team.HOME:
        return "RED"
    if team == Team.AWAY:
        return "BLUE"
    return "WHITE"


def _build_state_players(
    detections,
    team_overrides: Optional[Dict[int, Team]] = None,
) -> Dict[int, PlayerState]:
    import supervision as sv

    if len(detections) == 0:
        return {}

    players: Dict[int, PlayerState] = {}
    anchors = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    class_ids = (
        detections.class_id
        if detections.class_id is not None
        else np.full(len(detections), PLAYER_CLASS_ID, dtype=int)
    )

    for i in range(len(detections)):
        class_id = int(class_ids[i])
        if class_id not in {PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID, REFEREE_CLASS_ID}:
            continue

        player_id = _resolve_player_id(detections, i)
        confidence = (
            float(detections.confidence[i]) if detections.confidence is not None else 0.0
        )
        default_team = Team.REFEREE if class_id == REFEREE_CLASS_ID else Team.UNKNOWN
        team = (
            team_overrides.get(player_id, default_team)
            if team_overrides is not None
            else default_team
        )
        x, y = anchors[i]
        players[player_id] = PlayerState(
            player_id=player_id,
            team=team,
            pixel_x=float(x),
            pixel_y=float(y),
            confidence=confidence,
        )
    return players


def _build_team_overrides(detections, color_lookup: np.ndarray) -> Dict[int, Team]:
    team_overrides: Dict[int, Team] = {}
    if len(detections) == 0:
        return team_overrides

    class_ids = (
        detections.class_id
        if detections.class_id is not None
        else np.full(len(detections), PLAYER_CLASS_ID, dtype=int)
    )
    for i, color in enumerate(color_lookup.tolist()):
        class_id = int(class_ids[i])
        player_id = _resolve_player_id(detections, i)
        team_overrides[player_id] = _team_from_color(int(color), class_id)
    return team_overrides


def _build_tracked_objects(
    detections,
    team_overrides: Dict[int, Team],
) -> List[ObjectTrack]:
    if len(detections) == 0:
        return []

    class_ids = (
        detections.class_id
        if detections.class_id is not None
        else np.full(len(detections), PLAYER_CLASS_ID, dtype=int)
    )
    confidences = detections.confidence
    tracked_objects: List[ObjectTrack] = []
    for i in range(len(detections)):
        class_id = int(class_ids[i])
        if class_id not in {PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID, REFEREE_CLASS_ID}:
            continue

        player_id = _resolve_player_id(detections, i)
        team = team_overrides.get(
            player_id,
            Team.REFEREE if class_id == REFEREE_CLASS_ID else Team.UNKNOWN,
        )
        x1, y1, x2, y2 = detections.xyxy[i]
        confidence = float(confidences[i]) if confidences is not None else 0.0
        tracked_objects.append(
            ObjectTrack(
                track_id=player_id,
                class_id=class_id,
                xyxy=(int(x1), int(y1), int(x2), int(y2)),
                confidence=confidence,
                team=_tracked_team_name(team),
            )
        )
    return tracked_objects


def _resolve_sampling_plan(total_frames: int, end: Optional[int]) -> Tuple[int, Optional[int]]:
    if total_frames <= 0:
        return 1, end

    safe_end = end
    if safe_end is not None:
        safe_end = max(0, min(int(safe_end), total_frames))

    analyze_frames = int(safe_end if safe_end is not None else total_frames)
    analyze_frames = max(1, analyze_frames)

    if analyze_frames <= (MIN_CROP_SAMPLES * 2):
        return 1, safe_end

    target_samples = int(analyze_frames * CROP_SAMPLE_RATIO)
    target_samples = max(MIN_CROP_SAMPLES, target_samples)
    target_samples = min(MAX_CROP_SAMPLES, target_samples, analyze_frames)

    stride = max(1, math.ceil(analyze_frames / target_samples))
    return stride, safe_end


def collect_player_crops(source_video_path: str, player_detection_model, end: Optional[int] = None) -> List[np.ndarray]:
    import supervision as sv

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    total_frames = int(video_info.total_frames or 0)
    stride, safe_end = _resolve_sampling_plan(total_frames=total_frames, end=end)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=stride, end=safe_end
    )
    crops = []
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops += [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
    return crops


def resolve_goalkeepers_team_id(players, players_team_id: np.ndarray, goalkeepers) -> np.ndarray:
    import supervision as sv

    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def build_role_team_detections(frame: np.ndarray, detections, team_model, smoother):
    import supervision as sv

    players = detections[detections.class_id == PLAYER_CLASS_ID]
    player_crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
    if len(player_crops) > 0:
        raw_players_team_id = team_model.predict(player_crops)
        player_track_ids = [
            _resolve_player_id(players, idx)
            for idx in range(len(players))
        ]
        players_team_id = smoother.update(player_track_ids, raw_players_team_id)
    else:
        players_team_id = np.array([], dtype=int)

    goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
    referees = detections[detections.class_id == REFEREE_CLASS_ID]

    if (
        len(goalkeepers) > 0
        and len(players) > 0
        and np.any(players_team_id == 0)
        and np.any(players_team_id == 1)
    ):
        goalkeepers_team_id_for_state = resolve_goalkeepers_team_id(
            players=players, players_team_id=players_team_id, goalkeepers=goalkeepers
        )
    else:
        goalkeepers_team_id_for_state = np.full(
            len(goalkeepers), GOALKEEPER_COLOR_ID, dtype=int
        )

    merged_detections = sv.Detections.merge([players, goalkeepers, referees])
    draw_color_lookup = np.array(
        players_team_id.tolist()
        + [GOALKEEPER_COLOR_ID] * len(goalkeepers)
        + [REFEREE_COLOR_ID] * len(referees),
        dtype=int,
    )
    state_color_lookup = np.array(
        players_team_id.tolist()
        + goalkeepers_team_id_for_state.tolist()
        + [REFEREE_COLOR_ID] * len(referees),
        dtype=int,
    )
    return merged_detections, draw_color_lookup, state_color_lookup


def run_player_team_classification_packets(
    source_video_path: str, device: str
) -> Iterator[FramePacket]:
    import supervision as sv
    from ultralytics import YOLO
    from tracking.common.realtime_team import TeamAssignmentSmoother, TeamPrototypeModel

    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(COLORS),
        thickness=2,
    )
    ellipse_label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(COLORS),
        text_color=sv.Color.from_hex("#FFFFFF"),
        text_padding=5,
        text_thickness=1,
        text_position=sv.Position.BOTTOM_CENTER,
    )

    selected_device = _resolve_device(device)
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=selected_device)
    crops = collect_player_crops(
        source_video_path=source_video_path,
        player_detection_model=player_detection_model,
        end=CROPS_COLLECTION_END,
    )

    team_model = TeamPrototypeModel.fit(crops)
    smoother = TeamAssignmentSmoother()

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    frame_index = 0
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        detections, color_lookup, state_lookup = build_role_team_detections(
            frame=frame, detections=detections, team_model=team_model, smoother=smoother
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup
        )
        annotated_frame = ellipse_label_annotator.annotate(
            annotated_frame,
            detections,
            labels=labels,
            custom_color_lookup=color_lookup,
        )
        state_teams = _build_team_overrides(detections=detections, color_lookup=state_lookup)
        state_players = _build_state_players(detections=detections, team_overrides=state_teams)
        tracked_objects = _build_tracked_objects(detections=detections, team_overrides=state_teams)
        frame_index += 1
        yield FramePacket(
            frame_id=frame_index,
            timestamp=time.time(),
            video_ts=None,
            source_id=MODE_NAME,
            raw_frame=frame,
            annotated_frame=annotated_frame,
            tracked_objects=tracked_objects,
            players=state_players,
        )


def run_player_team_classification(
    source_video_path: str, device: str
) -> Iterator[FrameResult]:
    for packet in run_player_team_classification_packets(
        source_video_path=source_video_path,
        device=device,
    ):
        yield packet.annotated_frame, packet.players


def main(
    source_video_path: str,
    target_video_path: str,
    device: str,
    state_output_path: str = "",
    state_flush_interval: float = 0.5,
) -> None:
    import cv2
    import supervision as sv

    selected_device = _resolve_device(device)
    if device == "auto":
        print(f"[device] auto selected: {selected_device}")

    frame_generator = run_player_team_classification_packets(
        source_video_path=source_video_path, device=selected_device
    )

    game_state: Optional[GameStateManager] = None
    persistence: Optional[AsyncPersistence] = None
    if state_output_path:
        game_state = GameStateManager()
        persistence = AsyncPersistence(
            game_state=game_state,
            output_path=state_output_path,
            flush_interval=state_flush_interval,
        )
        persistence.start()

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    try:
        with sv.VideoSink(target_video_path, video_info) as sink:
            for packet in frame_generator:
                sink.write_frame(packet.annotated_frame)
                if game_state is not None:
                    game_state.update_packet(packet)

                cv2.imshow("frame", packet.annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cv2.destroyAllWindows()
        if persistence is not None:
            persistence.stop()
            persistence.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tracking pipeline (PLAYER_TEAM_CLASSIFICATION only)."
    )
    parser.add_argument("--source_video_path", type=str, required=True)
    parser.add_argument("--target_video_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--state_output_path", type=str, default="")
    parser.add_argument("--state_flush_interval", type=float, default=0.5)
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        state_output_path=args.state_output_path,
        state_flush_interval=args.state_flush_interval,
    )
