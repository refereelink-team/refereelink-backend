import inspect
from typing import Iterator

import numpy as np
import supervision as sv
from ultralytics import YOLO

from app.constants.paths import BALL_DETECTION_MODEL_PATH
from app.tracking.ball import BallAnnotator, BallTracker


def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
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

    slicer_kwargs = {
        'callback': callback,
        'slice_wh': (640, 640),
    }
    slicer_signature = inspect.signature(sv.InferenceSlicer.__init__)
    if 'overlap_filter' in slicer_signature.parameters:
        slicer_kwargs['overlap_filter'] = sv.OverlapFilter.NONE
    else:
        slicer_kwargs['overlap_filter_strategy'] = sv.OverlapFilter.NONE

    slicer = sv.InferenceSlicer(**slicer_kwargs)

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame
