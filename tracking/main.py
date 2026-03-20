import argparse
import time
from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from core import AsyncPersistence, FrameState, GameStateManager, PlayerState, Team
from tracking.annotators.soccer import draw_pitch, draw_points_on_pitch
from tracking.common.ball import BallTracker, BallAnnotator
from tracking.common.team import TeamClassifier
from tracking.common.view import ViewTransformer
from tracking.configs.soccer import SoccerPitchConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

TEAM_0_COLOR_ID = 0
TEAM_1_COLOR_ID = 1
GOALKEEPER_COLOR_ID = 2
REFEREE_COLOR_ID = 3

STRIDE = 60
TARGET_CROP_SAMPLES = 80
CROPS_COLLECTION_END = 1500
CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    PLAYER_TEAM_CLASSIFICATION = 'PLAYER_TEAM_CLASSIFICATION'
    RADAR = 'RADAR'


FrameResult = Tuple[np.ndarray, Dict[int, PlayerState]]


def _resolve_player_id(detections: sv.Detections, index: int) -> int:
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


def _build_state_players(
    detections: sv.Detections,
    team_overrides: Optional[Dict[int, Team]] = None
) -> Dict[int, PlayerState]:
    if len(detections) == 0:
        return {}

    players: Dict[int, PlayerState] = {}
    anchors = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    if detections.class_id is None:
        class_ids = np.full(len(detections), PLAYER_CLASS_ID, dtype=int)
    else:
        class_ids = detections.class_id

    for i in range(len(detections)):
        class_id = int(class_ids[i])
        if class_id not in {PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID, REFEREE_CLASS_ID}:
            continue

        player_id = _resolve_player_id(detections, i)
        confidence = (
            float(detections.confidence[i])
            if detections.confidence is not None
            else 0.0
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
            confidence=confidence
        )
    return players


def _build_team_overrides(
    detections: sv.Detections,
    color_lookup: np.ndarray
) -> Dict[int, Team]:
    team_overrides: Dict[int, Team] = {}
    if len(detections) == 0:
        return team_overrides

    if detections.class_id is None:
        class_ids = np.full(len(detections), PLAYER_CLASS_ID, dtype=int)
    else:
        class_ids = detections.class_id

    for i, color in enumerate(color_lookup.tolist()):
        class_id = int(class_ids[i])
        player_id = _resolve_player_id(detections, i)
        team_overrides[player_id] = _team_from_color(int(color), class_id)
    return team_overrides


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def _calculate_crop_stride(video_info: sv.VideoInfo) -> int:
    """
    Dynamically derive a frame stride so short videos still yield enough crops.
    """
    total_frames = video_info.total_frames or 0
    if total_frames <= 0:
        return STRIDE
    stride = max(1, total_frames // TARGET_CROP_SAMPLES)
    return min(STRIDE, stride)


def collect_player_crops(
    source_video_path: str,
    player_detection_model: YOLO,
    end: Optional[int] = None
) -> List[np.ndarray]:
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    stride = _calculate_crop_stride(video_info)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE, end=end)
    crops = []
    for frame in tqdm(frame_generator, desc='collecting player crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops += get_crops(frame, players)
    return crops


def build_role_team_detections(
    frame: np.ndarray,
    detections: sv.Detections,
    team_classifier: TeamClassifier
) -> Tuple[sv.Detections, np.ndarray, np.ndarray]:
    players = detections[detections.class_id == PLAYER_CLASS_ID]
    player_crops = get_crops(frame, players)
    players_team_id = (
        team_classifier.predict(player_crops)
        if len(player_crops) > 0
        else np.array([], dtype=int)
    )

    goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
    referees = detections[detections.class_id == REFEREE_CLASS_ID]

    if (
        len(goalkeepers) > 0
        and len(players) > 0
        and np.any(players_team_id == 0)
        and np.any(players_team_id == 1)
    ):
        goalkeepers_team_id_for_state = resolve_goalkeepers_team_id(
            players=players,
            players_team_id=players_team_id,
            goalkeepers=goalkeepers
        )
    else:
        goalkeepers_team_id_for_state = np.full(
            len(goalkeepers), GOALKEEPER_COLOR_ID, dtype=int
        )

    merged_detections = sv.Detections.merge([players, goalkeepers, referees])
    draw_color_lookup = np.array(
        players_team_id.tolist() +
        [GOALKEEPER_COLOR_ID] * len(goalkeepers) +
        [REFEREE_COLOR_ID] * len(referees),
        dtype=int
    )
    state_color_lookup = np.array(
        players_team_id.tolist() +
        goalkeepers_team_id_for_state.tolist() +
        [REFEREE_COLOR_ID] * len(referees),
        dtype=int
    )
    return merged_detections, draw_color_lookup, state_color_lookup


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
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


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
    return radar


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[FrameResult]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame, {}


def run_player_detection(source_video_path: str, device: str) -> Iterator[FrameResult]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        state_players = _build_state_players(detections=detections)
        yield annotated_frame, state_players


def run_ball_detection(source_video_path: str, device: str) -> Iterator[FrameResult]:
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame, {}


def run_player_tracking(source_video_path: str, device: str) -> Iterator[FrameResult]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        state_players = _build_state_players(detections=detections)
        yield annotated_frame, state_players


def run_team_classification(source_video_path: str, device: str) -> Iterator[FrameResult]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    crops = collect_player_crops(
        source_video_path=source_video_path,
        player_detection_model=player_detection_model,
        end=CROPS_COLLECTION_END
    )
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        state_teams = _build_team_overrides(
            detections=detections, color_lookup=color_lookup
        )
        state_players = _build_state_players(
            detections=detections, team_overrides=state_teams
        )
        yield annotated_frame, state_players


def run_radar(source_video_path: str, device: str) -> Iterator[FrameResult]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    crops = collect_player_crops(
        source_video_path=source_video_path,
        player_detection_model=player_detection_model,
        end=CROPS_COLLECTION_END
    )
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels,
            custom_color_lookup=color_lookup)

        h, w, _ = frame.shape
        radar = render_radar(detections, keypoints, color_lookup)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        state_teams = _build_team_overrides(
            detections=detections, color_lookup=color_lookup
        )
        state_players = _build_state_players(
            detections=detections, team_overrides=state_teams
        )
        yield annotated_frame, state_players


def run_player_team_classification(
    source_video_path: str,
    device: str
) -> Iterator[FrameResult]:
    """
    Distinguish players, goalkeepers, and referees first, then apply team
    classification only to player detections.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    crops = collect_player_crops(
        source_video_path=source_video_path,
        player_detection_model=player_detection_model,
        end=CROPS_COLLECTION_END
    )

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        detections, color_lookup, state_lookup = build_role_team_detections(
            frame=frame,
            detections=detections,
            team_classifier=team_classifier
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame,
            detections,
            labels=labels,
            custom_color_lookup=color_lookup
        )
        state_teams = _build_team_overrides(
            detections=detections, color_lookup=state_lookup
        )
        state_players = _build_state_players(
            detections=detections, team_overrides=state_teams
        )
        yield annotated_frame, state_players


def main(
    source_video_path: str,
    target_video_path: str,
    device: str,
    mode: Mode,
    state_output_path: str = "",
    state_flush_interval: float = 0.5
) -> None:
    if mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_TEAM_CLASSIFICATION:
        frame_generator = run_player_team_classification(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.RADAR:
        frame_generator = run_radar(
            source_video_path=source_video_path, device=device)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    game_state: Optional[GameStateManager] = None
    persistence: Optional[AsyncPersistence] = None
    if state_output_path:
        game_state = GameStateManager()
        persistence = AsyncPersistence(
            game_state=game_state,
            output_path=state_output_path,
            flush_interval=state_flush_interval
        )
        persistence.start()

    frame_id = 0
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    try:
        with sv.VideoSink(target_video_path, video_info) as sink:
            for frame, state_players in frame_generator:
                sink.write_frame(frame)
                if game_state is not None:
                    game_state.update_frame(
                        FrameState(
                            frame_id=frame_id,
                            timestamp=time.time(),
                            video_ts=None,
                            players=state_players,
                            ball=None,
                            source=mode.value
                        )
                    )
                    frame_id += 1

                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cv2.destroyAllWindows()
        if persistence is not None:
            persistence.stop()
            persistence.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_DETECTION)
    parser.add_argument('--state_output_path', type=str, default='')
    parser.add_argument('--state_flush_interval', type=float, default=0.5)
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode,
        state_output_path=args.state_output_path,
        state_flush_interval=args.state_flush_interval
    )
