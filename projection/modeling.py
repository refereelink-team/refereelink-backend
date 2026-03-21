# -*- coding: utf-8 -*-
"""
projection 建模能力：
- 将统一 packet 体系中的像素跟踪结果映射为 2D 球场平面模型
"""

from typing import Iterable, List

from core import ObjectTrack, ProjectedObject

from projection.homography import HomographyAdapter

ProjectedTracklet = ProjectedObject


def project_tracked_objects(
    tracked_objects: Iterable[ObjectTrack], homography: HomographyAdapter
) -> List[ProjectedTracklet]:
    """把像素空间跟踪结果映射为 2D 平面投影对象。"""
    projected: List[ProjectedTracklet] = []
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.xyxy
        foot_u = (x1 + x2) / 2.0
        foot_v = float(y2)
        map_x, map_y = homography.pixel_to_map(foot_u, foot_v)
        projected.append(
            ProjectedObject(
                track_id=int(obj.track_id),
                class_id=int(obj.class_id),
                xyxy=tuple(int(v) for v in obj.xyxy),
                confidence=float(obj.confidence),
                map_x=float(map_x),
                map_y=float(map_y),
                team=obj.team,
            )
        )
    return projected
