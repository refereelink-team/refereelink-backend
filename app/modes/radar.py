from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import cv2
import numpy as np
import supervision as sv

from app.constants.paths import (
    CAMERA_CALIBRATION_PATH,
    PITCH_DETECTION_MODEL_PATH,
    PLAYER_DETECTION_MODEL_PATH,
)
from app.geometry.pitch_projection import PitchProjectionResult
from app.runtime import (
    ELLIPSE_ANNOTATOR,
    ELLIPSE_LABEL_ANNOTATOR,
    annotate_pitch_observations,
    render_radar,
)
from app.vision.core import VisionCore


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
    # Optional foul prediction — populated when a foul checkpoint is provided.
    # Typed as Any to avoid a hard import of the predictor at module load time.
    foul_prediction: Optional[Any] = field(default=None)
    # World-coordinate (x, y) location of foul on the pitch, or None when no foul.
    # Currently always None: single-view foul localization is deferred until a
    # dedicated localization model is available.
    foul_location: Optional[np.ndarray] = field(default=None)


def emit_radar_log(log_callback: RadarLogCallback, message: str) -> None:
    if log_callback is not None:
        log_callback(message)


def format_radar_frame_summary(update: RadarFrameData) -> str:
    radar_status = update.homography_status if update.radar_available else "fallback"
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
    if m["m00"] <= 0:
        return None
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
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
    labels = (
        [str(tracker_id) for tracker_id in detections.tracker_id]
        if detections.tracker_id is not None
        else []
    )

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
    player_model_path: str = PLAYER_DETECTION_MODEL_PATH,
    pitch_model_path: str = PITCH_DETECTION_MODEL_PATH,
    camera_calibration_path: Optional[str] = CAMERA_CALIBRATION_PATH,
    enable_undistortion: bool = True,
    calibration_alpha: float = 0.0,
    pitch_detection_interval: int = 5,
    imgsz: int = 640,
) -> Iterator[RadarFrameData]:
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    emit_radar_log(log_callback, "loading shared vision core")
    vision_core = VisionCore(
        device=device,
        fps=video_info.fps,
        player_model_path=player_model_path,
        pitch_model_path=pitch_model_path,
        camera_calibration_path=camera_calibration_path,
        enable_undistortion=enable_undistortion,
        calibration_alpha=calibration_alpha,
        pitch_detection_interval=pitch_detection_interval,
        imgsz=imgsz,
    )
    vision_core.load_models()
    emit_radar_log(log_callback, "shared vision core ready")

    foul_detector = None
    if foul_checkpoint_path is not None:
        from app.foul_detection.detector import FoulDetector

        emit_radar_log(log_callback, "loading foul detection model")
        foul_detector = FoulDetector(checkpoint_path=foul_checkpoint_path, device=device)
        emit_radar_log(log_callback, "foul detection model ready")

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    for frame_index, frame in enumerate(frame_generator, start=1):
        vision_frame = vision_core.process(frame, frame_index)
        detections = vision_frame.tracked_detections
        projection = vision_frame.projection
        color_lookup = vision_frame.color_lookup
        players = detections
        goalkeepers = detections[:0]
        referees = detections[:0]

        foul_prediction = None
        if foul_detector is not None:
            foul_prediction = foul_detector.update(
                vision_frame.undistorted_frame, frame_index=frame_index
            )

        # Foul location projection on the radar is deferred until a dedicated
        # single-view localization model is available. The predictor still
        # emits candidate events; the radar marker stays absent for now.
        foul_location: Optional[np.ndarray] = None

        tracked_frame = render_tracked_frame(
            frame=vision_frame.undistorted_frame,
            detections=detections,
            color_lookup=color_lookup,
            projection=projection,
        )

        radar_frame = render_radar(
            detections=detections,
            projection=projection,
            color_lookup=color_lookup,
            foul_location=foul_location,
        )
        radar_available = projection.available
        if projection.homography_status == "unavailable":
            emit_radar_log(log_callback, f"frame={frame_index} radar projection unavailable")
        elif projection.homography_status == "stale":
            emit_radar_log(log_callback, f"frame={frame_index} reusing stale homography")
        elif projection.homography_status == "reused":
            emit_radar_log(log_callback, f"frame={frame_index} reusing homography")

        update = RadarFrameData(
            frame_index=frame_index,
            tracked_frame=tracked_frame,
            radar_frame=radar_frame,
            detections_total=len(detections),
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


def run_radar(
    source_video_path: str,
    device: str,
    foul_checkpoint_path: Optional[str] = None,
    player_model_path: str = PLAYER_DETECTION_MODEL_PATH,
    pitch_model_path: str = PITCH_DETECTION_MODEL_PATH,
    camera_calibration_path: Optional[str] = CAMERA_CALIBRATION_PATH,
    enable_undistortion: bool = True,
    calibration_alpha: float = 0.0,
    pitch_detection_interval: int = 5,
    imgsz: int = 640,
) -> Iterator[np.ndarray]:
    for update in iter_radar_analysis(
        source_video_path=source_video_path,
        device=device,
        foul_checkpoint_path=foul_checkpoint_path,
        player_model_path=player_model_path,
        pitch_model_path=pitch_model_path,
        camera_calibration_path=camera_calibration_path,
        enable_undistortion=enable_undistortion,
        calibration_alpha=calibration_alpha,
        pitch_detection_interval=pitch_detection_interval,
        imgsz=imgsz,
    ):
        combined = overlay_radar_on_frame(
            tracked_frame=update.tracked_frame,
            radar_frame=update.radar_frame,
        )
        yield combined
