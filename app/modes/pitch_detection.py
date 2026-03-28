from typing import Iterator

import numpy as np
import supervision as sv
from ultralytics import YOLO

from app.constants.paths import PITCH_DETECTION_MODEL_PATH
from app.geometry.pitch_projection import PitchProjectionEngine
from app.runtime import CONFIG, annotate_pitch_observations


def render_pitch_detection_frame(
    frame: np.ndarray,
    keypoints: sv.KeyPoints,
    projection_engine: PitchProjectionEngine,
) -> np.ndarray:
    projection = projection_engine.update(frame=frame, keypoints=keypoints)
    return annotate_pitch_observations(frame, projection.tracking_observations)


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    projection_engine = PitchProjectionEngine(config=CONFIG, fps=video_info.fps)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        yield render_pitch_detection_frame(
            frame=frame,
            keypoints=keypoints,
            projection_engine=projection_engine,
        )
