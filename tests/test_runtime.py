from typing import Tuple

import numpy as np

from app.geometry.pitch_projection import (
    PitchKeypointObservation,
    ProjectedPitchKeypoint,
    PitchProjectionResult,
    build_pitch_point_references,
)
import supervision as sv
from app.runtime import (
    CONFIG,
    PITCH_DRAW_PADDING,
    PITCH_DRAW_SCALE,
    draw_detected_pitch_keypoints_on_pitch,
    draw_reference_pitch_keypoints,
    render_empty_radar,
    render_radar,
)


REFERENCES = build_pitch_point_references(CONFIG)


def _pitch_canvas_point(point: Tuple[float, float]) -> Tuple[int, int]:
    return (
        int(point[0] * PITCH_DRAW_SCALE) + PITCH_DRAW_PADDING,
        int(point[1] * PITCH_DRAW_SCALE) + PITCH_DRAW_PADDING,
    )


def _build_observation(index: int) -> PitchKeypointObservation:
    reference = REFERENCES[index]
    return PitchKeypointObservation(
        reference=reference,
        image_xy=(100.0 + index, 120.0 + index),
        confidence=0.9,
        source='model',
    )


def test_draw_reference_pitch_keypoints_draws_ring_around_reference_vertex() -> None:
    pitch = np.zeros((900, 1400, 3), dtype=np.uint8)

    annotated_pitch = draw_reference_pitch_keypoints(pitch, labels=CONFIG.labels)

    center_x, center_y = _pitch_canvas_point(CONFIG.vertices[0])
    assert np.any(
        annotated_pitch[center_y - 1:center_y + 2, center_x + 10:center_x + 13] != 0
    )


def test_draw_detected_pitch_keypoints_draws_filled_marker() -> None:
    pitch = np.zeros((900, 1400, 3), dtype=np.uint8)

    annotated_pitch = draw_detected_pitch_keypoints_on_pitch(
        pitch,
        observations=[
            ProjectedPitchKeypoint(
                reference=REFERENCES[14],
                projected_world_xy=REFERENCES[14].world_xy,
                source='model',
            )
        ],
    )

    center_x, center_y = _pitch_canvas_point(CONFIG.vertices[14])
    assert np.any(annotated_pitch[center_y - 2:center_y + 3, center_x - 2:center_x + 3] != 0)


def test_render_radar_draws_reference_and_detected_pitch_keypoints() -> None:
    projection = PitchProjectionResult(
        tracking_observations=[],
        projected_keypoints=[
            ProjectedPitchKeypoint(
                reference=REFERENCES[0],
                projected_world_xy=REFERENCES[0].world_xy,
                source='model',
            )
        ],
        homography=np.eye(3, dtype=np.float32),
        homography_status='fresh',
        reprojection_error=0.0,
    )

    radar = render_radar(
        detections=sv.Detections(xyxy=np.empty((0, 4), dtype=np.float32)),
        projection=projection,
        color_lookup=np.array([]),
    )

    center_x, center_y = _pitch_canvas_point(CONFIG.vertices[0])
    assert tuple(radar[center_y, center_x]) != (34, 139, 34)
    assert tuple(radar[center_y, center_x + 12]) != (34, 139, 34)


def test_render_empty_radar_keeps_reference_pitch_keypoints() -> None:
    radar = render_empty_radar()

    center_x, center_y = _pitch_canvas_point(CONFIG.vertices[0])
    assert tuple(radar[center_y, center_x + 12]) != (34, 139, 34)
