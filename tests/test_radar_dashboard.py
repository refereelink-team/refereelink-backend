import numpy as np

from app.main import Mode
from app.modes.radar_dashboard import TRACKING_PANEL_RATIO, compose_dashboard_frame
from app.runtime import annotate_pitch_keypoints


def test_mode_enum_accepts_radar_dashboard() -> None:
    assert Mode('RADAR_DASHBOARD') == Mode.RADAR_DASHBOARD


def test_compose_dashboard_frame_builds_three_region_layout() -> None:
    tracked_frame = np.full((720, 1280, 3), (0, 0, 255), dtype=np.uint8)
    radar_frame = np.full((400, 600, 3), (0, 255, 0), dtype=np.uint8)

    dashboard_frame = compose_dashboard_frame(
        tracked_frame=tracked_frame,
        radar_frame=radar_frame,
        log_lines=['loading models', 'frame=1 total=5 players=4 goalkeepers=1'],
    )

    assert dashboard_frame.shape == tracked_frame.shape

    top_height = dashboard_frame.shape[0] - min(
        dashboard_frame.shape[0] - 1,
        max(180, int(dashboard_frame.shape[0] * 0.28)),
    )
    tracking_width = int(round(dashboard_frame.shape[1] * TRACKING_PANEL_RATIO))
    left_panel = dashboard_frame[:top_height, :tracking_width]
    right_panel = dashboard_frame[:top_height, tracking_width:]
    bottom_panel = dashboard_frame[top_height:, :]

    assert left_panel.shape[1] > right_panel.shape[1]
    assert left_panel[:, :, 2].mean() > left_panel[:, :, 1].mean()
    assert right_panel[:, :, 1].mean() > right_panel[:, :, 2].mean()
    assert np.any(bottom_panel != bottom_panel[0, 0])


def test_annotate_pitch_keypoints_draws_on_valid_points() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    xy = np.array([[20, 30], [80, 70], [0, 0]], dtype=np.float32)

    annotated_frame = annotate_pitch_keypoints(frame, xy, labels=['01', '02', '03'])

    assert np.count_nonzero(annotated_frame) > 0
    assert np.any(annotated_frame[26:34, 16:24] != 0)
