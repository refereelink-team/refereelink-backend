"""FOUL_DETECTION mode — standalone foul detection pipeline.

Streams frames from a source video, maintains a rolling frame buffer via
:class:`~app.foul_detection.detector.FoulDetector`, and yields frames.
Candidate events are surfaced by the caller, not drawn on the frame.
"""

from typing import Iterator, Optional

import numpy as np
import supervision as sv

from app.constants.paths import CAMERA_CALIBRATION_PATH
from app.foul_detection.detector import FoulDetector
from app.geometry.camera import build_undistorter


def run_foul_detection(
    source_video_path: str,
    device: str,
    foul_checkpoint_path: str,
    window_size: int = 24,
    stride: int = 8,
    camera_calibration_path: Optional[str] = CAMERA_CALIBRATION_PATH,
    enable_undistortion: bool = True,
    calibration_alpha: float = 0.0,
) -> Iterator[np.ndarray]:
    """Yield BGR frames after running the rolling foul detector.

    Args:
        source_video_path: Path to the input video file.
        device: Torch device string (``'cpu'``, ``'cuda'``, ``'mps'``, …).
        foul_checkpoint_path: Path to the foul model checkpoint.
        window_size: Sliding window length in frames (default 24).
        stride: Inference interval in frames (default 8).

    Yields:
        BGR frames as ``uint8`` numpy arrays.
    """
    detector = FoulDetector(
        checkpoint_path=foul_checkpoint_path,
        device=device,
        window_size=window_size,
        stride=stride,
    )
    undistorter = build_undistorter(
        calibration_path=camera_calibration_path,
        enabled=enable_undistortion,
        alpha=calibration_alpha,
    )

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame_index, frame in enumerate(frame_generator, start=1):
        undistorted_frame = undistorter.apply(frame)
        detector.update(undistorted_frame, frame_index=frame_index)
        yield undistorted_frame
