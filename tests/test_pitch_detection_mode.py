import numpy as np
import supervision as sv

from app.geometry.pitch_projection import PitchProjectionEngine, build_pitch_point_references
from app.modes.pitch_detection import render_pitch_detection_frame
from app.runtime import CONFIG


REFERENCES = build_pitch_point_references(CONFIG)


def _make_keypoints(assignments: dict[int, tuple[tuple[float, float], float]]) -> sv.KeyPoints:
    xy = np.zeros((1, len(REFERENCES), 2), dtype=np.float32)
    confidence = np.zeros((1, len(REFERENCES)), dtype=np.float32)
    for index, (point, score) in assignments.items():
        xy[0, index] = np.array(point, dtype=np.float32)
        confidence[0, index] = score
    return sv.KeyPoints(xy=xy, confidence=confidence)


def _make_frame(points: list[tuple[float, float]]) -> np.ndarray:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (0, 96, 0)
    for x_coord, y_coord in points:
        x_int = int(round(x_coord))
        y_int = int(round(y_coord))
        frame[max(0, y_int - 2):y_int + 3, max(0, x_int - 2):x_int + 3] = (255, 255, 255)
    return frame


def _image_point_from_world(point: tuple[float, float]) -> tuple[float, float]:
    return point[0] * 0.05 + 100.0, point[1] * 0.05 + 60.0


def test_render_pitch_detection_frame_hides_invalid_keypoints() -> None:
    engine = PitchProjectionEngine(config=CONFIG, fps=25.0)
    assignments = {
        0: (_image_point_from_world(REFERENCES[0].world_xy), 0.95),
        5: (_image_point_from_world(REFERENCES[5].world_xy), 0.95),
        13: (_image_point_from_world(REFERENCES[13].world_xy), 0.95),
        24: (_image_point_from_world(REFERENCES[24].world_xy), 0.95),
        29: (_image_point_from_world(REFERENCES[29].world_xy), 0.20),
        30: ((3.0, 300.0), 0.95),
    }
    frame = _make_frame([point for point, _ in assignments.values()])

    annotated = render_pitch_detection_frame(
        frame=frame,
        keypoints=_make_keypoints(assignments),
        projection_engine=engine,
    )

    valid_x, valid_y = [int(round(v)) for v in assignments[0][0]]
    invalid_x, invalid_y = [int(round(v)) for v in assignments[30][0]]

    assert np.any(annotated[valid_y - 2:valid_y + 3, valid_x - 2:valid_x + 3] != frame[valid_y - 2:valid_y + 3, valid_x - 2:valid_x + 3])
    assert np.array_equal(
        annotated[invalid_y - 2:invalid_y + 3, invalid_x - 2:invalid_x + 3],
        frame[invalid_y - 2:invalid_y + 3, invalid_x - 2:invalid_x + 3],
    )
