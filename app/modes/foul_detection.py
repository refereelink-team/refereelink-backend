"""
FOUL_DETECTION mode — standalone foul detection pipeline.

Streams frames from a source video, maintains a rolling frame buffer via
:class:`~app.foul_detection.detector.FoulDetector`, and overlays the
fouls_far HUD on frames where a foul prediction passes the built-in
confidence filter.
"""

from typing import Iterator, Optional

import numpy as np
import supervision as sv

from app.constants.paths import CAMERA_CALIBRATION_PATH
from app.foul_detection.detector import FoulDetector
from app.geometry.camera import build_undistorter

# FoulDetector.__init__ has already injected the repo root onto sys.path,
# so the following imports from offside resolve without installing fouls_far.
from offside.foul_overlay import _hud_show_prediction, draw_foul_hud


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
    """
    Yield BGR frames annotated with a foul HUD overlay.

    The HUD is drawn only when the current rolling-window prediction passes
    the fouls_far ``_hud_show_prediction`` filter (strict mode, offence ≥ 0.48,
    action ≥ 0.45).  Frames during the warmup period and frames where no foul
    is detected are yielded unmodified.

    Args:
        source_video_path: Path to the input video file.
        device: Torch device string (``'cpu'``, ``'cuda'``, ``'mps'``, …).
        foul_checkpoint_path: Path to the MVFoul ``.pth.tar`` checkpoint.
        window_size: Sliding window length in frames (default 24).
        stride: Inference interval in frames (default 8).

    Yields:
        Annotated BGR frames as ``uint8`` numpy arrays.
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
    for frame in frame_generator:
        undistorted_frame = undistorter.apply(frame)
        prediction = detector.update(undistorted_frame)
        annotated = undistorted_frame.copy()
        if prediction is not None and _hud_show_prediction(
            prediction,
            min_offence_confidence=0.48,
            min_action_confidence=0.45,
            strict_hud_filter=True,
        ):
            draw_foul_hud(annotated, prediction)  # modifies annotated in-place
        yield annotated
