from typing import Iterator, Optional

import numpy as np
import supervision as sv

from app.constants.paths import CAMERA_CALIBRATION_PATH, PLAYER_DETECTION_MODEL_PATH
from app.runtime import ELLIPSE_ANNOTATOR, ELLIPSE_LABEL_ANNOTATOR
from app.vision.core import VisionCore


def run_player_tracking(
    source_video_path: str,
    device: str,
    player_model_path: str = PLAYER_DETECTION_MODEL_PATH,
    camera_calibration_path: Optional[str] = CAMERA_CALIBRATION_PATH,
    enable_undistortion: bool = True,
    calibration_alpha: float = 0.0,
    imgsz: int = 640,
) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    vision_core = VisionCore(
        device=device,
        player_model_path=player_model_path,
        camera_calibration_path=camera_calibration_path,
        enable_undistortion=enable_undistortion,
        calibration_alpha=calibration_alpha,
        imgsz=imgsz,
        enable_pitch=False,
    )
    vision_core.load_models()
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame_index, frame in enumerate(frame_generator, start=1):
        vision_frame = vision_core.process(frame, frame_index)
        detections = vision_frame.tracked_detections

        labels = [
            str(tracker_id) for tracker_id in detections.tracker_id
        ] if detections.tracker_id is not None else []

        annotated_frame = vision_frame.undistorted_frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        yield annotated_frame
