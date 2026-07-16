from typing import Iterator, Optional

import numpy as np
import supervision as sv

from app.constants.paths import (
    CAMERA_CALIBRATION_PATH,
    PITCH_DETECTION_MODEL_PATH,
)
from app.geometry.pitch_projection import PitchProjectionEngine
from app.runtime import CONFIG, annotate_pitch_observations
from app.vision.core import VisionCore


def render_pitch_detection_frame(
    frame: np.ndarray,
    keypoints: sv.KeyPoints,
    projection_engine: PitchProjectionEngine,
) -> np.ndarray:
    projection = projection_engine.update(frame=frame, keypoints=keypoints)
    return annotate_pitch_observations(frame, projection.tracking_observations)


def run_pitch_detection(
    source_video_path: str,
    device: str,
    pitch_model_path: str = PITCH_DETECTION_MODEL_PATH,
    camera_calibration_path: Optional[str] = CAMERA_CALIBRATION_PATH,
    enable_undistortion: bool = True,
    calibration_alpha: float = 0.0,
    pitch_detection_interval: int = 5,
    imgsz: int = 640,
) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    vision_core = VisionCore(
        device=device,
        fps=video_info.fps,
        pitch_model_path=pitch_model_path,
        camera_calibration_path=camera_calibration_path,
        enable_undistortion=enable_undistortion,
        calibration_alpha=calibration_alpha,
        pitch_detection_interval=pitch_detection_interval,
        imgsz=imgsz,
        enable_player=False,
    )
    vision_core.load_models()
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame_index, frame in enumerate(frame_generator, start=1):
        vision_frame = vision_core.process(frame, frame_index)
        yield annotate_pitch_observations(
            vision_frame.undistorted_frame,
            vision_frame.projection.tracking_observations,
        )
