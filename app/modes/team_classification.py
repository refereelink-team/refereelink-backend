from typing import Iterator, Optional

import numpy as np
import supervision as sv

from app.constants.paths import CAMERA_CALIBRATION_PATH, PLAYER_DETECTION_MODEL_PATH
from app.runtime import ELLIPSE_ANNOTATOR, ELLIPSE_LABEL_ANNOTATOR
from app.vision.core import VisionCore


def run_team_classification(
    source_video_path: str,
    device: str,
    player_model_path: str = PLAYER_DETECTION_MODEL_PATH,
    camera_calibration_path: Optional[str] = CAMERA_CALIBRATION_PATH,
    enable_undistortion: bool = True,
    calibration_alpha: float = 0.0,
    imgsz: int = 640,
) -> Iterator[np.ndarray]:
    """Render person detections using the shared core.

    Official YOLOv11 COCO weights only provide the ``person`` class in this
    phase, so team and role classification remain intentionally unresolved.
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
        labels = (
            [str(track_id) for track_id in detections.tracker_id]
            if detections.tracker_id is not None
            else []
        )
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            vision_frame.undistorted_frame.copy(),
            detections,
            custom_color_lookup=vision_frame.color_lookup,
        )
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame,
            detections,
            labels=labels,
            custom_color_lookup=vision_frame.color_lookup,
        )
        yield annotated_frame
