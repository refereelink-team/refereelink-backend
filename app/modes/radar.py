from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from app.classification.team import TeamClassifier
from app.constants.classes import (
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    REFEREE_CLASS_ID,
    STRIDE,
)
from app.constants.paths import PITCH_DETECTION_MODEL_PATH, PLAYER_DETECTION_MODEL_PATH
from app.geometry.pitch_projection import PitchProjectionEngine, PitchProjectionResult
from app.runtime import (
    CONFIG,
    ELLIPSE_ANNOTATOR,
    ELLIPSE_LABEL_ANNOTATOR,
    annotate_pitch_observations,
    get_crops,
    render_radar,
    resolve_goalkeepers_team_id,
)


RadarLogCallback = Optional[Callable[[str], None]]


@dataclass
class RadarFrameData:
    frame_index: int
    tracked_frame: np.ndarray
    radar_frame: np.ndarray
    detections_total: int
    player_count: int
    goalkeeper_count: int
    referee_count: int
    radar_available: bool
    homography_status: str


def emit_radar_log(log_callback: RadarLogCallback, message: str) -> None:
    if log_callback is not None:
        log_callback(message)


def format_radar_frame_summary(update: RadarFrameData) -> str:
    radar_status = update.homography_status if update.radar_available else 'fallback'
    return (
        f"frame={update.frame_index} total={update.detections_total} "
        f"players={update.player_count} goalkeepers={update.goalkeeper_count} "
        f"referees={update.referee_count} radar={radar_status}"
    )


def render_tracked_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    color_lookup: np.ndarray,
    projection: PitchProjectionResult,
) -> np.ndarray:
    labels = [str(tracker_id) for tracker_id in detections.tracker_id]

    tracked_frame = frame.copy()
    tracked_frame = ELLIPSE_ANNOTATOR.annotate(
        tracked_frame, detections, custom_color_lookup=color_lookup
    )
    tracked_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
        tracked_frame,
        detections,
        labels=labels,
        custom_color_lookup=color_lookup,
    )
    tracked_frame = annotate_pitch_observations(
        tracked_frame,
        observations=projection.tracking_observations,
    )
    return tracked_frame


def overlay_radar_on_frame(
    tracked_frame: np.ndarray,
    radar_frame: np.ndarray,
) -> np.ndarray:
    annotated_frame = tracked_frame.copy()
    frame_height, frame_width, _ = annotated_frame.shape
    radar = sv.resize_image(radar_frame, (frame_width // 2, frame_height // 2))
    radar_height, radar_width, _ = radar.shape
    rect = sv.Rect(
        x=frame_width // 2 - radar_width // 2,
        y=frame_height - radar_height,
        width=radar_width,
        height=radar_height,
    )
    return sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)


def iter_radar_analysis(
    source_video_path: str,
    device: str,
    log_callback: RadarLogCallback = None,
) -> Iterator[RadarFrameData]:
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    projection_engine = PitchProjectionEngine(config=CONFIG, fps=video_info.fps)
    emit_radar_log(log_callback, 'loading player detection model')
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    emit_radar_log(log_callback, 'loading pitch detection model')
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)

    emit_radar_log(log_callback, 'collecting player crops for team classifier')
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE
    )
    crops = []
    sampled_frames = 0
    for frame in tqdm(frame_generator, desc='collecting crops'):
        sampled_frames += 1
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])
    emit_radar_log(
        log_callback,
        f'collected {len(crops)} player crops from {sampled_frames} sampled frames',
    )

    emit_radar_log(log_callback, 'fitting team classifier')
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)
    emit_radar_log(log_callback, 'team classifier ready')

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame_index, frame in enumerate(frame_generator, start=1):
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
            players, players_team_id, goalkeepers
        )

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        merged_detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist()
            + goalkeepers_team_id.tolist()
            + [REFEREE_CLASS_ID] * len(referees)
        )
        projection = projection_engine.update(frame=frame, keypoints=keypoints)

        tracked_frame = render_tracked_frame(
            frame=frame,
            detections=merged_detections,
            color_lookup=color_lookup,
            projection=projection,
        )

        radar_frame = render_radar(
            detections=merged_detections,
            projection=projection,
            color_lookup=color_lookup,
        )
        radar_available = projection.available
        if projection.homography_status == 'unavailable':
            emit_radar_log(log_callback, f'frame={frame_index} radar projection unavailable')
        elif projection.homography_status == 'stale':
            emit_radar_log(log_callback, f'frame={frame_index} reusing stale homography')

        update = RadarFrameData(
            frame_index=frame_index,
            tracked_frame=tracked_frame,
            radar_frame=radar_frame,
            detections_total=len(merged_detections),
            player_count=len(players),
            goalkeeper_count=len(goalkeepers),
            referee_count=len(referees),
            radar_available=radar_available,
            homography_status=projection.homography_status,
        )
        emit_radar_log(log_callback, format_radar_frame_summary(update))
        yield update


def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    for update in iter_radar_analysis(source_video_path=source_video_path, device=device):
        yield overlay_radar_on_frame(
            tracked_frame=update.tracked_frame,
            radar_frame=update.radar_frame,
        )
