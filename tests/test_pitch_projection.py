from typing import Dict, Iterable, Tuple

import numpy as np
import supervision as sv

from app.geometry.pitch_projection import PitchProjectionEngine, build_pitch_point_references
from app.runtime import CONFIG


REFERENCES = build_pitch_point_references(CONFIG)


def _make_frame_with_points(points: Iterable[Tuple[float, float]]) -> np.ndarray:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (0, 96, 0)
    for x_coord, y_coord in points:
        x_int = int(round(x_coord))
        y_int = int(round(y_coord))
        frame[max(0, y_int - 2):y_int + 3, max(0, x_int - 2):x_int + 3] = (255, 255, 255)
    return frame


def _make_keypoints(
    assignments: Dict[int, Tuple[Tuple[float, float], float]],
) -> sv.KeyPoints:
    xy = np.zeros((1, len(REFERENCES), 2), dtype=np.float32)
    confidence = np.zeros((1, len(REFERENCES)), dtype=np.float32)
    for index, (point, score) in assignments.items():
        xy[0, index] = np.array(point, dtype=np.float32)
        confidence[0, index] = score
    return sv.KeyPoints(xy=xy, confidence=confidence)


def _image_point_from_world(point: Tuple[float, float]) -> Tuple[float, float]:
    return point[0] * 0.05 + 100.0, point[1] * 0.05 + 60.0


def test_pitch_projection_engine_filters_border_and_low_confidence_points() -> None:
    engine = PitchProjectionEngine(config=CONFIG, fps=25.0)
    assignments = {
        0: ((_image_point_from_world(REFERENCES[0].world_xy)), 0.95),
        5: ((_image_point_from_world(REFERENCES[5].world_xy)), 0.95),
        13: ((_image_point_from_world(REFERENCES[13].world_xy)), 0.95),
        24: ((_image_point_from_world(REFERENCES[24].world_xy)), 0.95),
        29: ((_image_point_from_world(REFERENCES[29].world_xy)), 0.20),
        30: ((3.0, 300.0), 0.95),
    }
    visible_points = [point for point, _ in assignments.values()]
    frame = _make_frame_with_points(visible_points)

    projection = engine.update(frame=frame, keypoints=_make_keypoints(assignments))
    labels = {observation.reference.label for observation in projection.tracking_observations}

    assert REFERENCES[29].label not in labels
    assert REFERENCES[30].label not in labels
    assert REFERENCES[0].label in labels
    assert REFERENCES[24].label in labels


def test_pitch_projection_engine_rejects_outlier_and_builds_fresh_homography() -> None:
    engine = PitchProjectionEngine(config=CONFIG, fps=25.0)
    inlier_indices = [0, 5, 13, 16, 24, 29]
    assignments = {
        index: (_image_point_from_world(REFERENCES[index].world_xy), 0.95)
        for index in inlier_indices
    }
    assignments[30] = ((1100.0, 600.0), 0.95)
    frame = _make_frame_with_points([point for point, _ in assignments.values()])

    projection = engine.update(frame=frame, keypoints=_make_keypoints(assignments))
    labels = {observation.reference.label for observation in projection.tracking_observations}

    assert projection.homography_status == 'fresh'
    assert projection.available
    assert REFERENCES[30].label not in labels


def test_pitch_projection_engine_reuses_stale_homography_without_showing_missing_points() -> None:
    engine = PitchProjectionEngine(config=CONFIG, fps=25.0)
    inlier_indices = [0, 5, 13, 16, 24, 29]
    first_assignments = {
        index: (_image_point_from_world(REFERENCES[index].world_xy), 0.95)
        for index in inlier_indices
    }
    first_frame = _make_frame_with_points([point for point, _ in first_assignments.values()])
    first_projection = engine.update(
        frame=first_frame,
        keypoints=_make_keypoints(first_assignments),
    )

    second_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    second_frame[:] = (0, 96, 0)
    second_projection = engine.update(
        frame=second_frame,
        keypoints=_make_keypoints({}),
    )

    assert first_projection.homography_status == 'fresh'
    assert second_projection.homography_status == 'stale'
    assert second_projection.available
    assert second_projection.tracking_observations == []


def test_pitch_projection_engine_reuse_expires_after_half_second() -> None:
    engine = PitchProjectionEngine(config=CONFIG, fps=25.0)
    inlier_indices = [0, 5, 13, 16, 24, 29]
    assignments = {
        index: (_image_point_from_world(REFERENCES[index].world_xy), 0.95)
        for index in inlier_indices
    }
    frame = _make_frame_with_points([point for point, _ in assignments.values()])
    engine.update(frame=frame, keypoints=_make_keypoints(assignments))

    reused = [engine.reuse(frame) for _ in range(engine.max_stale_frames)]
    expired = engine.reuse(frame)

    assert all(result.homography_status == "reused" for result in reused)
    assert expired.homography_status == "unavailable"
    assert not expired.available
