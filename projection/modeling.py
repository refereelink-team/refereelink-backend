# -*- coding: utf-8 -*-
"""
projection 建模能力：
- 将 tracking 底座输出的像素 tracklet 映射为 2D 球场平面模型
"""

from dataclasses import dataclass
from typing import List, Tuple

from projection.homography import HomographyAdapter
from tracking.backend import TrackedObject


@dataclass
class ProjectedTracklet:
    track_id: int
    class_id: int
    xyxy: Tuple[int, int, int, int]
    confidence: float
    map_x: float
    map_y: float
    team: str


def project_tracked_objects(
    tracked_objects: List[TrackedObject], homography: HomographyAdapter
) -> List[ProjectedTracklet]:
    """把 tracking 底座输出映射为 2D 平面 tracklet。"""
    projected: List[ProjectedTracklet] = []
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.xyxy
        foot_u = (x1 + x2) / 2.0
        foot_v = float(y2)
        map_x, map_y = homography.pixel_to_map(foot_u, foot_v)
        projected.append(
            ProjectedTracklet(
                track_id=obj.track_id,
                class_id=obj.class_id,
                xyxy=obj.xyxy,
                confidence=obj.confidence,
                map_x=map_x,
                map_y=map_y,
                team=obj.team,
            )
        )
    return projected
