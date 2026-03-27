# -*- coding: utf-8 -*-
"""
calibration: SN-style field calibration module.

Provides:
- SoccerPitch: full 3D pitch geometry in center-origin meters
- Camera: camera pose estimation from homography + 3D→2D projection
- SegmentationNetwork: DeepLabV3 pitch segmentation
- SNCalibrationEngine: per-frame calibration with optical flow tracking
"""

from calibration.camera import Camera
from calibration.detect import (
    SegmentationNetwork,
    generate_class_synthesis,
    get_line_extremities,
)
from calibration.engine import SNCalibrationEngine
from calibration.pitch import SoccerPitch

__all__ = [
    "SoccerPitch",
    "Camera",
    "SegmentationNetwork",
    "generate_class_synthesis",
    "get_line_extremities",
    "SNCalibrationEngine",
]
