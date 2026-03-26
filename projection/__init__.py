"""
projection: 球场投影建模模块。

负责 3D/像素到 2D 球场平面的投影建模能力，供 offside 等上层业务模块复用。

增强功能：
- HomographyState: 带质量指标的单应矩阵状态
- HomographyEstimator: 带 RANSAC 质量评估的估计器
- HomographyQualityGate: 质量门控
- KeypointManager: 语义关键点池管理
- ShotChangeDetector: 镜头切换检测
- TrackSmoother: 球员轨迹平滑
"""

from .calibrator import FieldCalibrator, FieldLineDetector, LineFallbackDetector
from .dynamic_projector import DynamicProjector, create_dynamic_projector
from .homography import (
    DEFAULT_DST_PTS,
    DEFAULT_SRC_PTS,
    HomographyAdapter,
    HomographyEstimator,
    HomographyQuality,
    HomographyQualityGate,
    HomographySmoother,
    HomographyState,
    build_default_homography,
    template_to_image_points,
)
from .keypoint_manager import KeypointManager, PITCH_KEYPOINT_IDS, TrackedKeypoint
from .modeling import (
    ProjectedTracklet,
    TrackSmoother,
    create_track_smoother,
    extract_footpoint,
    project_tracked_objects,
)
from .shot_change_detector import ShotChangeDetector
from .visualization import (
    build_projected_objects,
    render_projection_frame,
    run_projection_video_pipeline,
    run_projection_video_pipeline_from_tracks,
    draw_keypoints_on_frame,
    draw_keypoints_on_field,
    KEYPOINT_COLORS,
)

__all__ = [
    # Homography
    "HomographyAdapter",
    "HomographySmoother",
    "HomographyState",
    "HomographyQuality",
    "HomographyEstimator",
    "HomographyQualityGate",
    "template_to_image_points",
    "DEFAULT_SRC_PTS",
    "DEFAULT_DST_PTS",
    "build_default_homography",
    # Keypoint Manager
    "KeypointManager",
    "TrackedKeypoint",
    "PITCH_KEYPOINT_IDS",
    # Shot Change
    "ShotChangeDetector",
    # Modeling
    "ProjectedTracklet",
    "TrackSmoother",
    "create_track_smoother",
    "extract_footpoint",
    "project_tracked_objects",
    # Visualization
    "build_projected_objects",
    "render_projection_frame",
    "run_projection_video_pipeline",
    "run_projection_video_pipeline_from_tracks",
    # Calibrator
    "FieldCalibrator",
    "FieldLineDetector",
    "LineFallbackDetector",
    # Dynamic Projector
    "DynamicProjector",
    "create_dynamic_projector",
]