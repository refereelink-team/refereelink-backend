"""Metric pitch geometry and per-venue profiles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class PitchDimensions:
    """Pitch dimensions in metres.

    Defaults describe the common international 105 x 68 m field rather than
    the legacy visualization canvas.  Production venues should persist their
    measured dimensions in a :class:`VenueProfile`.
    """

    length_m: float = 105.0
    width_m: float = 68.0
    penalty_area_depth_m: float = 16.5
    penalty_area_width_m: float = 40.32
    goal_area_depth_m: float = 5.5
    goal_area_width_m: float = 18.32
    centre_circle_radius_m: float = 9.15
    penalty_spot_distance_m: float = 11.0

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not np.isfinite(value) or value <= 0 for value in values.values()):
            raise ValueError("all pitch dimensions must be finite and positive")
        if self.penalty_area_width_m >= self.width_m:
            raise ValueError("penalty area must be narrower than the pitch")
        if self.goal_area_width_m >= self.penalty_area_width_m:
            raise ValueError("goal area must be narrower than the penalty area")
        if self.penalty_area_depth_m >= self.length_m / 2:
            raise ValueError("penalty area depth is invalid for pitch length")


@dataclass(frozen=True)
class VenueProfile:
    venue_id: str
    pitch: PitchDimensions
    coordinate_system: str = "left_goal_line_origin_x_length_y_width_z_up"
    version: int = 1

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VenueProfile":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("version", 0)) != 1:
            raise ValueError("unsupported venue profile version")
        pitch_payload = payload.get("pitch")
        if not isinstance(pitch_payload, dict):
            raise ValueError("venue profile is missing pitch dimensions")
        return cls(
            venue_id=str(payload["venue_id"]),
            pitch=PitchDimensions(**pitch_payload),
            coordinate_system=str(payload.get("coordinate_system", "")),
            version=1,
        )


class PitchModel:
    """Named landmarks and line segments on the z=0 pitch plane."""

    def __init__(self, dimensions: PitchDimensions | None = None) -> None:
        self.dimensions = dimensions or PitchDimensions()

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (0.0, self.dimensions.length_m, 0.0, self.dimensions.width_m)

    @property
    def landmarks(self) -> Dict[str, Tuple[float, float]]:
        """Return the canonical 32-point annotation vocabulary.

        The ordering mirrors the legacy pitch detector, but names and values
        are expressed in the metric V2 coordinate system.  Keeping this
        vocabulary in the geometry model prevents annotation, training, and
        runtime code from maintaining separate point tables.
        """

        d = self.dimensions
        penalty_y0 = (d.width_m - d.penalty_area_width_m) / 2.0
        penalty_y1 = (d.width_m + d.penalty_area_width_m) / 2.0
        goal_y0 = (d.width_m - d.goal_area_width_m) / 2.0
        goal_y1 = (d.width_m + d.goal_area_width_m) / 2.0
        centre_x = d.length_m / 2.0
        return {
            "left_top_corner": (0.0, 0.0),
            "left_goal_line_penalty_top": (0.0, penalty_y0),
            "left_goal_line_goal_area_top": (0.0, goal_y0),
            "left_goal_line_goal_area_bottom": (0.0, goal_y1),
            "left_goal_line_penalty_bottom": (0.0, penalty_y1),
            "left_bottom_corner": (0.0, d.width_m),
            "left_goal_area_front_top": (d.goal_area_depth_m, goal_y0),
            "left_goal_area_front_bottom": (d.goal_area_depth_m, goal_y1),
            "left_penalty_spot": (d.penalty_spot_distance_m, d.width_m / 2.0),
            "left_penalty_area_front_top": (d.penalty_area_depth_m, penalty_y0),
            "left_penalty_area_front_goal_top": (d.penalty_area_depth_m, goal_y0),
            "left_penalty_area_front_goal_bottom": (d.penalty_area_depth_m, goal_y1),
            "left_penalty_area_front_bottom": (d.penalty_area_depth_m, penalty_y1),
            "centre_top": (centre_x, 0.0),
            "centre_circle_top": (centre_x, d.width_m / 2.0 - d.centre_circle_radius_m),
            "centre_circle_bottom": (
                centre_x,
                d.width_m / 2.0 + d.centre_circle_radius_m,
            ),
            "centre_bottom": (centre_x, d.width_m),
            "right_penalty_area_front_top": (
                d.length_m - d.penalty_area_depth_m,
                penalty_y0,
            ),
            "right_penalty_area_front_goal_top": (
                d.length_m - d.penalty_area_depth_m,
                goal_y0,
            ),
            "right_penalty_area_front_goal_bottom": (
                d.length_m - d.penalty_area_depth_m,
                goal_y1,
            ),
            "right_penalty_area_front_bottom": (
                d.length_m - d.penalty_area_depth_m,
                penalty_y1,
            ),
            "right_penalty_spot": (
                d.length_m - d.penalty_spot_distance_m,
                d.width_m / 2.0,
            ),
            "right_goal_area_front_top": (d.length_m - d.goal_area_depth_m, goal_y0),
            "right_goal_area_front_bottom": (
                d.length_m - d.goal_area_depth_m,
                goal_y1,
            ),
            "right_top_corner": (d.length_m, 0.0),
            "right_goal_line_penalty_top": (d.length_m, penalty_y0),
            "right_goal_line_goal_area_top": (d.length_m, goal_y0),
            "right_goal_line_goal_area_bottom": (d.length_m, goal_y1),
            "right_goal_line_penalty_bottom": (d.length_m, penalty_y1),
            "right_bottom_corner": (d.length_m, d.width_m),
            "centre_circle_left": (
                centre_x - d.centre_circle_radius_m,
                d.width_m / 2.0,
            ),
            "centre_circle_right": (
                centre_x + d.centre_circle_radius_m,
                d.width_m / 2.0,
            ),
        }

    @property
    def point_landmarks(self) -> Dict[str, Tuple[float, float]]:
        """Return isolated painted points that are not part of the 32-point grid."""

        d = self.dimensions
        return {
            "centre_spot": (d.length_m / 2.0, d.width_m / 2.0),
            "left_penalty_spot": (d.penalty_spot_distance_m, d.width_m / 2.0),
            "right_penalty_spot": (
                d.length_m - d.penalty_spot_distance_m,
                d.width_m / 2.0,
            ),
        }

    def semantic_elements(self, curve_samples: int = 96) -> Dict[str, np.ndarray]:
        """Return metric polylines for every semantic painted pitch element."""

        if curve_samples < 8:
            raise ValueError("curve_samples must be at least 8")
        d = self.dimensions
        penalty_y0 = (d.width_m - d.penalty_area_width_m) / 2.0
        penalty_y1 = (d.width_m + d.penalty_area_width_m) / 2.0
        goal_y0 = (d.width_m - d.goal_area_width_m) / 2.0
        goal_y1 = (d.width_m + d.goal_area_width_m) / 2.0
        centre_x = d.length_m / 2.0

        def segment(start: Tuple[float, float], end: Tuple[float, float]) -> np.ndarray:
            return np.asarray((start, end), dtype=np.float64)

        theta = np.linspace(0.0, 2.0 * np.pi, curve_samples, endpoint=False)
        centre_circle = np.column_stack(
            (
                centre_x + d.centre_circle_radius_m * np.cos(theta),
                d.width_m / 2.0 + d.centre_circle_radius_m * np.sin(theta),
            )
        )
        arc_limit = np.arccos(
            (d.penalty_area_depth_m - d.penalty_spot_distance_m)
            / d.centre_circle_radius_m
        )
        left_theta = np.linspace(-arc_limit, arc_limit, curve_samples // 2)
        right_theta = np.linspace(np.pi - arc_limit, np.pi + arc_limit, curve_samples // 2)

        return {
            "top_touchline": segment((0.0, 0.0), (d.length_m, 0.0)),
            "bottom_touchline": segment((0.0, d.width_m), (d.length_m, d.width_m)),
            "left_goal_line": segment((0.0, 0.0), (0.0, d.width_m)),
            "right_goal_line": segment((d.length_m, 0.0), (d.length_m, d.width_m)),
            "halfway_line": segment((centre_x, 0.0), (centre_x, d.width_m)),
            "left_penalty_area_top": segment((0.0, penalty_y0), (d.penalty_area_depth_m, penalty_y0)),
            "left_penalty_area_front": segment((d.penalty_area_depth_m, penalty_y0), (d.penalty_area_depth_m, penalty_y1)),
            "left_penalty_area_bottom": segment((d.penalty_area_depth_m, penalty_y1), (0.0, penalty_y1)),
            "right_penalty_area_top": segment((d.length_m - d.penalty_area_depth_m, penalty_y0), (d.length_m, penalty_y0)),
            "right_penalty_area_front": segment((d.length_m - d.penalty_area_depth_m, penalty_y0), (d.length_m - d.penalty_area_depth_m, penalty_y1)),
            "right_penalty_area_bottom": segment((d.length_m, penalty_y1), (d.length_m - d.penalty_area_depth_m, penalty_y1)),
            "left_goal_area_top": segment((0.0, goal_y0), (d.goal_area_depth_m, goal_y0)),
            "left_goal_area_front": segment((d.goal_area_depth_m, goal_y0), (d.goal_area_depth_m, goal_y1)),
            "left_goal_area_bottom": segment((d.goal_area_depth_m, goal_y1), (0.0, goal_y1)),
            "right_goal_area_top": segment((d.length_m - d.goal_area_depth_m, goal_y0), (d.length_m, goal_y0)),
            "right_goal_area_front": segment((d.length_m - d.goal_area_depth_m, goal_y0), (d.length_m - d.goal_area_depth_m, goal_y1)),
            "right_goal_area_bottom": segment((d.length_m, goal_y1), (d.length_m - d.goal_area_depth_m, goal_y1)),
            "centre_circle": centre_circle,
            "left_penalty_arc": np.column_stack(
                (
                    d.penalty_spot_distance_m
                    + d.centre_circle_radius_m * np.cos(left_theta),
                    d.width_m / 2.0
                    + d.centre_circle_radius_m * np.sin(left_theta),
                )
            ),
            "right_penalty_arc": np.column_stack(
                (
                    d.length_m
                    - d.penalty_spot_distance_m
                    + d.centre_circle_radius_m * np.cos(right_theta),
                    d.width_m / 2.0
                    + d.centre_circle_radius_m * np.sin(right_theta),
                )
            ),
        }

    def contains(self, xy_m: Tuple[float, float], margin_m: float = 0.0) -> bool:
        x_coord, y_coord = xy_m
        return (
            -margin_m <= x_coord <= self.dimensions.length_m + margin_m
            and -margin_m <= y_coord <= self.dimensions.width_m + margin_m
        )

    def grid(self, x_steps: int = 22, y_steps: int = 15) -> np.ndarray:
        if x_steps < 2 or y_steps < 2:
            raise ValueError("pitch grid needs at least two samples per axis")
        x_values = np.linspace(0.0, self.dimensions.length_m, x_steps)
        y_values = np.linspace(0.0, self.dimensions.width_m, y_steps)
        return np.array(np.meshgrid(x_values, y_values)).reshape(2, -1).T
