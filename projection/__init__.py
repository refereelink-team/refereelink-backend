"""
projection: 球场投影建模模块。

负责 3D/像素到 2D 球场平面的投影建模能力，供 offside 等上层业务模块复用。
"""

from .homography import (
    DEFAULT_DST_PTS,
    DEFAULT_SRC_PTS,
    HomographyAdapter,
    build_default_homography,
)
from .modeling import ProjectedTracklet, project_tracked_objects
from .visualization import run_projection_video_pipeline

__all__ = [
    "HomographyAdapter",
    "DEFAULT_SRC_PTS",
    "DEFAULT_DST_PTS",
    "build_default_homography",
    "ProjectedTracklet",
    "project_tracked_objects",
    "run_projection_video_pipeline",
]
