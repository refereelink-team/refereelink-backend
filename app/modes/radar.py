from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, List, Optional

import cv2
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
    # Optional foul prediction — populated when a MVFoul checkpoint is provided.
    # Typed as Any to avoid a hard import of fouls_far at module load time.
    foul_prediction: Optional[Any] = field(default=None)
    # World-coordinate (x, y) location of foul on the pitch, or None when no foul.
    foul_location: Optional[np.ndarray] = field(default=None)


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


def compute_motion_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute the centroid (center of mass) of a binary mask in image coordinates.

    Args:
        mask: Binary or soft mask (H, W) with values in [0, 1] or [0, 255].

    Returns:
        Centroid as (x, y) float64 in image space (x=col, y=row),
        or None if the mask is empty or invalid.
    """
    if mask is None:
        return None
    mask_f = mask.astype(np.float32)
    if mask.ndim != 2:
        return None
    m = cv2.moments(mask_f)
    if m['m00'] <= 0:
        return None
    cx = m['m10'] / m['m00']
    cy = m['m01'] / m['m00']
    return np.array([cx, cy], dtype=np.float64)


def project_point_to_world(
    image_point: np.ndarray,
    homography: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Transform a single point from image space to world/pitch coordinates.

    Args:
        image_point: Point as (x, y) in image space (shape (2,)).
        homography: 3x3 transformation matrix.

    Returns:
        World point as (x, y) in pitch coordinates (centimeters), or None on failure.
    """
    if image_point is None or homography is None:
        return None
    try:
        pt = np.array([[image_point]], dtype=np.float64)
        transformed = cv2.perspectiveTransform(pt.reshape(-1, 1, 2), homography)
        return transformed.reshape(-1).astype(np.float64)
    except cv2.error:
        return None


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
    foul_checkpoint_path: Optional[str] = None,
) -> Iterator[RadarFrameData]:
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    projection_engine = PitchProjectionEngine(config=CONFIG, fps=video_info.fps)
    emit_radar_log(log_callback, 'loading player detection model')
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    emit_radar_log(log_callback, 'loading pitch detection model')
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)

    foul_detector = None
    if foul_checkpoint_path is not None:
        from app.foul_detection.detector import FoulDetector
        emit_radar_log(log_callback, 'loading foul detection model')
        foul_detector = FoulDetector(checkpoint_path=foul_checkpoint_path, device=device)
        emit_radar_log(log_callback, 'foul detection model ready')

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

    # Circular buffer of 3 frames for motion mask computation (prev, curr, next).
    _frame_buffer: List[Optional[np.ndarray]] = [None, None, None]
    _buffer_head: int = 0

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

        foul_prediction = None
        if foul_detector is not None:
            foul_prediction = foul_detector.update(frame)

        # Compute foul location: derive from motion mask centroid projected through homography.
        foul_location: Optional[np.ndarray] = None
        if foul_prediction is not None:
            from offside.foul_overlay import _hud_show_prediction, motion_foul_region_mask
            if _hud_show_prediction(
                foul_prediction,
                min_offence_confidence=0.48,
                min_action_confidence=0.45,
                strict_hud_filter=True,
            ):
                prev_idx = (_buffer_head - 1) % 3
                next_idx = (_buffer_head + 1) % 3
                prev_frame = _frame_buffer[prev_idx]
                next_frame = _frame_buffer[next_idx]
                motion_mask = motion_foul_region_mask(frame, prev_frame, next_frame)
                if motion_mask is not None:
                    centroid = compute_motion_centroid(motion_mask)
                    if centroid is not None and projection.homography is not None:
                        world_point = project_point_to_world(centroid, projection.homography)
                        if world_point is not None:
                            wx, wy = world_point
                            if 0 <= wx <= CONFIG.length and 0 <= wy <= CONFIG.width:
                                foul_location = world_point
                                emit_radar_log(
                                    log_callback,
                                    f'frame={frame_index} foul_location=({wx:.0f},{wy:.0f})',
                                )

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
            foul_location=foul_location,
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
            foul_prediction=foul_prediction,
            foul_location=foul_location,
        )
        emit_radar_log(log_callback, format_radar_frame_summary(update))
        yield update

        # Update circular frame buffer for next iteration's motion mask.
        _frame_buffer[_buffer_head] = frame.copy()
        _buffer_head = (_buffer_head + 1) % 3


def run_radar(
    source_video_path: str,
    device: str,
    foul_checkpoint_path: Optional[str] = None,
) -> Iterator[np.ndarray]:
    for update in iter_radar_analysis(
        source_video_path=source_video_path,
        device=device,
        foul_checkpoint_path=foul_checkpoint_path,
    ):
        combined = overlay_radar_on_frame(
            tracked_frame=update.tracked_frame,
            radar_frame=update.radar_frame,
        )
        if update.foul_prediction is not None:
            from offside.foul_overlay import _hud_show_prediction, draw_foul_hud
            if _hud_show_prediction(
                update.foul_prediction,
                min_offence_confidence=0.48,
                min_action_confidence=0.45,
                strict_hud_filter=True,
            ):
                draw_foul_hud(combined, update.foul_prediction)  # in-place
        yield combined
