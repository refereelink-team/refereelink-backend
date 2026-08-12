import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import supervision as sv

from app.annotators.pitch import draw_pitch, draw_points_on_pitch
from app.config.pitch import SoccerPitchConfiguration
from app.constants.colors import COLORS
from app.geometry.pitch_projection import (
    PitchKeypointObservation,
    ProjectedPitchKeypoint,
    PitchProjectionResult,
)


CONFIG = SoccerPitchConfiguration()
PITCH_DRAW_PADDING = 50
PITCH_DRAW_SCALE = 0.1

VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


def normalize_proxy_env() -> None:
    """
    Normalize proxy environment variables for libraries expecting socks5 scheme.

    Some tools export SOCKS proxies as "socks://...", while httpx/huggingface
    expects "socks5://...".
    """
    proxy_env_keys = [
        'http_proxy', 'https_proxy', 'all_proxy',
        'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
    ]
    for key in proxy_env_keys:
        value = os.environ.get(key)
        if value and value.startswith('socks://'):
            os.environ[key] = value.replace('socks://', 'socks5://', 1)


def annotate_pitch_keypoints(
    frame: np.ndarray,
    xy: np.ndarray,
    labels: Sequence[str],
) -> np.ndarray:
    annotated_frame = frame.copy()
    for index, point in enumerate(xy):
        _draw_video_keypoint_marker(
            annotated_frame,
            point=point,
            label=labels[index] if index < len(labels) else str(index + 1),
            color=sv.Color.from_hex(CONFIG.colors[index % len(CONFIG.colors)]).as_bgr(),
        )

    return annotated_frame


def annotate_pitch_observations(
    frame: np.ndarray,
    observations: Sequence[PitchKeypointObservation],
) -> np.ndarray:
    annotated_frame = frame.copy()
    for observation in observations:
        _draw_video_keypoint_marker(
            annotated_frame,
            point=np.array(observation.image_xy, dtype=np.float32),
            label=observation.reference.label,
            color=sv.Color.from_hex(observation.reference.color).as_bgr(),
        )
    return annotated_frame


def _draw_video_keypoint_marker(
    frame: np.ndarray,
    point: np.ndarray,
    label: str,
    color: Tuple[int, int, int],
) -> None:
    if point[0] <= 1 or point[1] <= 1:
        return

    center = (int(point[0]), int(point[1]))
    cv2.circle(
        frame,
        center,
        5,
        color,
        thickness=-1,
    )
    cv2.circle(
        frame,
        center,
        8,
        (20, 20, 20),
        thickness=1,
    )
    cv2.putText(
        frame,
        label,
        (center[0] + 6, center[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )


def _get_valid_pitch_keypoint_mask(xy: np.ndarray) -> np.ndarray:
    return (xy[:, 0] > 1) & (xy[:, 1] > 1)


def _project_pitch_point_to_canvas(
    point: Sequence[float],
    padding: int = PITCH_DRAW_PADDING,
    scale: float = PITCH_DRAW_SCALE,
) -> Tuple[int, int]:
    return (
        int(point[0] * scale) + padding,
        int(point[1] * scale) + padding,
    )


def draw_reference_pitch_keypoints(
    pitch: np.ndarray,
    labels: Sequence[str],
    padding: int = PITCH_DRAW_PADDING,
    scale: float = PITCH_DRAW_SCALE,
) -> np.ndarray:
    annotated_pitch = pitch.copy()
    for index, point in enumerate(CONFIG.vertices):
        center = _project_pitch_point_to_canvas(point, padding=padding, scale=scale)
        label = labels[index] if index < len(labels) else str(index + 1)
        cv2.circle(
            annotated_pitch,
            center,
            12,
            (245, 245, 245),
            thickness=2,
        )
        cv2.putText(
            annotated_pitch,
            label,
            (center[0] + 10, center[1] + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return annotated_pitch


def draw_detected_pitch_keypoints_on_pitch(
    pitch: np.ndarray,
    observations: Sequence[ProjectedPitchKeypoint],
    padding: int = PITCH_DRAW_PADDING,
    scale: float = PITCH_DRAW_SCALE,
) -> np.ndarray:
    annotated_pitch = pitch.copy()
    for observation in observations:
        center = _project_pitch_point_to_canvas(
            observation.projected_world_xy,
            padding=padding,
            scale=scale,
        )
        color = sv.Color.from_hex(observation.reference.color).as_bgr()
        cv2.circle(
            annotated_pitch,
            center,
            7,
            color,
            thickness=-1,
        )
        cv2.circle(
            annotated_pitch,
            center,
            9,
            (20, 20, 20),
            thickness=1,
        )
    return annotated_pitch


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar(
    detections: sv.Detections,
    projection: PitchProjectionResult,
    color_lookup: np.ndarray,
    foul_location: Optional[np.ndarray] = None,
    config: Optional[SoccerPitchConfiguration] = None,
) -> np.ndarray:
    pitch_config = config or CONFIG
    radar = draw_pitch(config=pitch_config)
    radar = draw_reference_pitch_keypoints(radar, labels=pitch_config.labels)

    if projection.homography_status == 'fresh':
        radar = draw_detected_pitch_keypoints_on_pitch(
            radar,
            observations=projection.projected_keypoints,
        )
    elif projection.homography_status in {'reused', 'stale'}:
        cv2.putText(
            radar,
            'REUSED HOMOGRAPHY' if projection.homography_status == 'reused' else 'STALE HOMOGRAPHY',
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            sv.Color.WHITE.as_bgr(),
            2,
            cv2.LINE_AA,
        )

    if projection.homography is None:
        cv2.putText(
            radar,
            'RADAR UNAVAILABLE',
            (40, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            sv.Color.WHITE.as_bgr(),
            2,
            cv2.LINE_AA,
        )
        return radar

    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER).astype(np.float32)
    if xy.size == 0:
        transformed_xy = xy
    else:
        transformed_xy = cv2.perspectiveTransform(
            xy.reshape(-1, 1, 2),
            projection.homography,
        ).reshape(-1, 2)
        pitch_mask = (
            (transformed_xy[:, 0] >= 0)
            & (transformed_xy[:, 0] <= pitch_config.length)
            & (transformed_xy[:, 1] >= 0)
            & (transformed_xy[:, 1] <= pitch_config.width)
        )
        transformed_xy = transformed_xy[pitch_mask]
        color_lookup = color_lookup[pitch_mask]

    radar = draw_points_on_pitch(
        config=pitch_config, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=pitch_config, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=pitch_config, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=pitch_config, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=pitch_config, xy=transformed_xy[color_lookup == 4],
        face_color=sv.Color.from_hex(COLORS[4]), radius=20, pitch=radar)

    # Draw foul location marker on the radar.
    if foul_location is not None:
        foul_color = sv.Color.from_hex('#FF4500')
        screen_x = int(foul_location[0] * PITCH_DRAW_SCALE) + PITCH_DRAW_PADDING
        screen_y = int(foul_location[1] * PITCH_DRAW_SCALE) + PITCH_DRAW_PADDING
        cv2.circle(radar, (screen_x, screen_y), 18, foul_color.as_bgr(), -1)
        cv2.circle(radar, (screen_x, screen_y), 22, (30, 30, 30), 2)
        cv2.putText(
            radar,
            'FOUL',
            (screen_x - 18, screen_y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

    return radar


def render_empty_radar(message: str = 'RADAR UNAVAILABLE') -> np.ndarray:
    radar = draw_pitch(config=CONFIG)
    radar = draw_reference_pitch_keypoints(radar, labels=CONFIG.labels)
    cv2.putText(
        radar,
        message,
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        sv.Color.WHITE.as_bgr(),
        2,
        cv2.LINE_AA,
    )
    return radar
