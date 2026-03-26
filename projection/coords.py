from __future__ import annotations

from typing import Tuple

# field_map.png geometry
FIELD_MAP_WIDTH = 1200
FIELD_MAP_HEIGHT = 800
PITCH_LEFT = 75.0
PITCH_TOP = 60.0
PITCH_RIGHT = 1125.0
PITCH_BOTTOM = 740.0
PITCH_WIDTH_PX = PITCH_RIGHT - PITCH_LEFT
PITCH_HEIGHT_PX = PITCH_BOTTOM - PITCH_TOP

# FIFA 11v11 pitch geometry (meters)
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
HALF_PITCH_LENGTH_M = PITCH_LENGTH_M / 2.0
HALF_PITCH_WIDTH_M = PITCH_WIDTH_M / 2.0


def field_meter_center_to_top_left(x_m: float, y_m: float) -> Tuple[float, float]:
    """Convert center-origin meters to top-left-origin meters."""
    return x_m + HALF_PITCH_LENGTH_M, y_m + HALF_PITCH_WIDTH_M


def field_meter_top_left_to_center(x_m: float, y_m: float) -> Tuple[float, float]:
    """Convert top-left-origin meters to center-origin meters."""
    return x_m - HALF_PITCH_LENGTH_M, y_m - HALF_PITCH_WIDTH_M


def field_meter_center_to_map_pixel(x_m: float, y_m: float) -> Tuple[float, float]:
    """Convert center-origin meters to field_map pixel coordinates."""
    x_tl, y_tl = field_meter_center_to_top_left(x_m, y_m)
    x_px = PITCH_LEFT + (x_tl / PITCH_LENGTH_M) * PITCH_WIDTH_PX
    y_px = PITCH_TOP + (y_tl / PITCH_WIDTH_M) * PITCH_HEIGHT_PX
    return x_px, y_px


def map_pixel_to_field_meter_center(x_px: float, y_px: float) -> Tuple[float, float]:
    """Convert field_map pixel coordinates to center-origin meters."""
    x_tl = ((x_px - PITCH_LEFT) / PITCH_WIDTH_PX) * PITCH_LENGTH_M
    y_tl = ((y_px - PITCH_TOP) / PITCH_HEIGHT_PX) * PITCH_WIDTH_M
    return field_meter_top_left_to_center(x_tl, y_tl)


def clamp_field_meter_center(x_m: float, y_m: float) -> Tuple[float, float]:
    """Clamp a metric point to legal pitch bounds."""
    x = min(HALF_PITCH_LENGTH_M, max(-HALF_PITCH_LENGTH_M, x_m))
    y = min(HALF_PITCH_WIDTH_M, max(-HALF_PITCH_WIDTH_M, y_m))
    return x, y

