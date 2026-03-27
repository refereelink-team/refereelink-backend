# -*- coding: utf-8 -*-
"""
Soccer pitch geometry (3D world model).

Adapted from SoccerNet-Calibration (sn-calibration-main/src/soccerpitch.py).
Provides full 3D pitch geometry in center-origin meters (x ∈ [-52.5, 52.5], y ∈ [-34, 34]).
"""

import numpy as np


class SoccerPitch:
    """Static class variables specified by FIFA rules."""
    GOAL_LINE_TO_PENALTY_MARK = 11.0
    PENALTY_AREA_WIDTH = 40.32
    PENALTY_AREA_LENGTH = 16.5
    GOAL_AREA_WIDTH = 18.32
    GOAL_AREA_LENGTH = 5.5
    CENTER_CIRCLE_RADIUS = 9.15
    GOAL_HEIGHT = 2.44
    GOAL_LENGTH = 7.32

    # 28 line classes used by the segmentation model
    lines_classes = [
        'Big rect. left bottom',
        'Big rect. left main',
        'Big rect. left top',
        'Big rect. right bottom',
        'Big rect. right main',
        'Big rect. right top',
        'Circle central',
        'Circle left',
        'Circle right',
        'Goal left crossbar',
        'Goal left post left',
        'Goal left post right',
        'Goal right crossbar',
        'Goal right post left',
        'Goal right post right',
        'Goal unknown',
        'Line unknown',
        'Middle line',
        'Side line bottom',
        'Side line left',
        'Side line right',
        'Side line top',
        'Small rect. left bottom',
        'Small rect. left main',
        'Small rect. left top',
        'Small rect. right bottom',
        'Small rect. right main',
        'Small rect. right top',
    ]

    # Palette maps class name → BGR color for visualization
    palette = {
        'Big rect. left bottom': (127, 0, 0),
        'Big rect. left main': (102, 102, 102),
        'Big rect. left top': (0, 0, 127),
        'Big rect. right bottom': (86, 32, 39),
        'Big rect. right main': (48, 77, 0),
        'Big rect. right top': (14, 97, 100),
        'Circle central': (0, 0, 255),
        'Circle left': (255, 127, 0),
        'Circle right': (0, 255, 255),
        'Goal left crossbar': (255, 255, 200),
        'Goal left post left': (165, 255, 0),
        'Goal left post right': (155, 119, 45),
        'Goal right crossbar': (86, 32, 139),
        'Goal right post left': (196, 120, 153),
        'Goal right post right': (166, 36, 52),
        'Goal unknown': (0, 0, 0),
        'Line unknown': (0, 0, 0),
        'Middle line': (255, 255, 0),
        'Side line bottom': (255, 0, 255),
        'Side line left': (0, 255, 150),
        'Side line right': (0, 230, 0),
        'Side line top': (230, 0, 0),
        'Small rect. left bottom': (0, 150, 255),
        'Small rect. left main': (254, 173, 225),
        'Small rect. left top': (87, 72, 39),
        'Small rect. right bottom': (122, 0, 255),
        'Small rect. right main': (255, 255, 255),
        'Small rect. right top': (153, 23, 153),
    }

    def __init__(self, pitch_length=105.0, pitch_width=68.0):
        """
        Initialize 3D coordinates of all pitch elements.

        Args:
            pitch_length: pitch length in meters (FIFA rules: 90-120m)
            pitch_width: pitch width in meters (FIFA rules: 45-90m)
        """
        self.PITCH_LENGTH = pitch_length
        self.PITCH_WIDTH = pitch_width

        half_l = pitch_length / 2.0
        half_w = pitch_width / 2.0

        # Key 3D points (center-origin)
        self.center_mark = np.array([0.0, 0.0, 0.0], dtype='float')
        self.bottom_right_corner = np.array([half_l, half_w, 0.0], dtype='float')
        self.bottom_left_corner = np.array([-half_l, half_w, 0.0], dtype='float')
        self.top_right_corner = np.array([half_l, -half_w, 0.0], dtype='float')
        self.top_left_corner = np.array([-half_l, -half_w, 0.0], dtype='float')

        # Halfway line marks
        self.halfway_and_bottom_touch_line_mark = np.array([0.0, half_w, 0.0], dtype='float')
        self.halfway_and_top_touch_line_mark = np.array([0.0, -half_w, 0.0], dtype='float')

        # Penalty marks
        self.left_penalty_mark = np.array([-half_l + self.GOAL_LINE_TO_PENALTY_MARK, 0.0, 0.0], dtype='float')
        self.right_penalty_mark = np.array([half_l - self.GOAL_LINE_TO_PENALTY_MARK, 0.0, 0.0], dtype='float')

        # Penalty area corners
        self.left_penalty_area_top_right = np.array(
            [-half_l + self.PENALTY_AREA_LENGTH, -self.PENALTY_AREA_WIDTH / 2.0, 0.0], dtype='float')
        self.left_penalty_area_top_left = np.array(
            [-half_l, -self.PENALTY_AREA_WIDTH / 2.0, 0.0], dtype='float')
        self.left_penalty_area_bottom_right = np.array(
            [-half_l + self.PENALTY_AREA_LENGTH, self.PENALTY_AREA_WIDTH / 2.0, 0.0], dtype='float')
        self.left_penalty_area_bottom_left = np.array(
            [-half_l, self.PENALTY_AREA_WIDTH / 2.0, 0.0], dtype='float')

        self.right_penalty_area_top_right = np.array(
            [half_l, -self.PENALTY_AREA_WIDTH / 2.0, 0.0], dtype='float')
        self.right_penalty_area_top_left = np.array(
            [half_l - self.PENALTY_AREA_LENGTH, -self.PENALTY_AREA_WIDTH / 2.0, 0.0], dtype='float')
        self.right_penalty_area_bottom_right = np.array(
            [half_l, self.PENALTY_AREA_WIDTH / 2.0, 0.0], dtype='float')
        self.right_penalty_area_bottom_left = np.array(
            [half_l - self.PENALTY_AREA_LENGTH, self.PENALTY_AREA_WIDTH / 2.0, 0.0], dtype='float')

        # Goal area corners
        self.left_goal_area_top_right = np.array(
            [-half_l + self.GOAL_AREA_LENGTH, -self.GOAL_AREA_WIDTH / 2.0, 0.0], dtype='float')
        self.left_goal_area_top_left = np.array(
            [-half_l, -self.GOAL_AREA_WIDTH / 2.0, 0.0], dtype='float')
        self.left_goal_area_bottom_right = np.array(
            [-half_l + self.GOAL_AREA_LENGTH, self.GOAL_AREA_WIDTH / 2.0, 0.0], dtype='float')
        self.left_goal_area_bottom_left = np.array(
            [-half_l, self.GOAL_AREA_WIDTH / 2.0, 0.0], dtype='float')

        self.right_goal_area_top_right = np.array(
            [half_l, -self.GOAL_AREA_WIDTH / 2.0, 0.0], dtype='float')
        self.right_goal_area_top_left = np.array(
            [half_l - self.GOAL_AREA_LENGTH, -self.GOAL_AREA_WIDTH / 2.0, 0.0], dtype='float')
        self.right_goal_area_bottom_right = np.array(
            [half_l, self.GOAL_AREA_WIDTH / 2.0, 0.0], dtype='float')
        self.right_goal_area_bottom_left = np.array(
            [half_l - self.GOAL_AREA_LENGTH, self.GOAL_AREA_WIDTH / 2.0, 0.0], dtype='float')

        # Center circle arc intersections
        x_16m = -half_l + self.PENALTY_AREA_LENGTH
        dx = self.PENALTY_AREA_LENGTH - self.GOAL_LINE_TO_PENALTY_MARK
        y_arc = -np.sqrt(self.CENTER_CIRCLE_RADIUS ** 2 - dx ** 2)
        self.top_left_16M_arc = np.array([x_16m, y_arc, 0.0], dtype='float')
        x_16m_r = half_l - self.PENALTY_AREA_LENGTH
        y_arc_r = -np.sqrt(self.CENTER_CIRCLE_RADIUS ** 2 - dx ** 2)
        self.top_right_16M_arc = np.array([x_16m_r, y_arc_r, 0.0], dtype='float')
        y_arc_bl = np.sqrt(self.CENTER_CIRCLE_RADIUS ** 2 - dx ** 2)
        self.bottom_left_16M_arc = np.array([x_16m, y_arc_bl, 0.0], dtype='float')
        y_arc_br = np.sqrt(self.CENTER_CIRCLE_RADIUS ** 2 - dx ** 2)
        self.bottom_right_16M_arc = np.array([x_16m_r, y_arc_br, 0.0], dtype='float')

        # Goal posts
        self.left_goal_bottom_left_post = np.array([-half_l, self.GOAL_LENGTH / 2.0, 0.0], dtype='float')
        self.left_goal_top_left_post = np.array([-half_l, self.GOAL_LENGTH / 2.0, -self.GOAL_HEIGHT], dtype='float')
        self.left_goal_bottom_right_post = np.array([-half_l, -self.GOAL_LENGTH / 2.0, 0.0], dtype='float')
        self.left_goal_top_right_post = np.array([-half_l, -self.GOAL_LENGTH / 2.0, -self.GOAL_HEIGHT], dtype='float')

        self.right_goal_bottom_left_post = np.array([half_l, -self.GOAL_LENGTH / 2.0, 0.0], dtype='float')
        self.right_goal_top_left_post = np.array([half_l, -self.GOAL_LENGTH / 2.0, -self.GOAL_HEIGHT], dtype='float')
        self.right_goal_bottom_right_post = np.array([half_l, self.GOAL_LENGTH / 2.0, 0.0], dtype='float')
        self.right_goal_top_right_post = np.array([half_l, self.GOAL_LENGTH / 2.0, -self.GOAL_HEIGHT], dtype='float')

        # Point dictionary for lookups
        self.point_dict = {
            "CENTER_MARK": self.center_mark,
            "L_PENALTY_MARK": self.left_penalty_mark,
            "R_PENALTY_MARK": self.right_penalty_mark,
            "TL_PITCH_CORNER": self.top_left_corner,
            "BL_PITCH_CORNER": self.bottom_left_corner,
            "TR_PITCH_CORNER": self.top_right_corner,
            "BR_PITCH_CORNER": self.bottom_right_corner,
            "L_PENALTY_AREA_TL": self.left_penalty_area_top_left,
            "L_PENALTY_AREA_TR": self.left_penalty_area_top_right,
            "L_PENALTY_AREA_BL": self.left_penalty_area_bottom_left,
            "L_PENALTY_AREA_BR": self.left_penalty_area_bottom_right,
            "R_PENALTY_AREA_TL": self.right_penalty_area_top_left,
            "R_PENALTY_AREA_TR": self.right_penalty_area_top_right,
            "R_PENALTY_AREA_BL": self.right_penalty_area_bottom_left,
            "R_PENALTY_AREA_BR": self.right_penalty_area_bottom_right,
            "L_GOAL_AREA_TL": self.left_goal_area_top_left,
            "L_GOAL_AREA_TR": self.left_goal_area_top_right,
            "L_GOAL_AREA_BL": self.left_goal_area_bottom_left,
            "L_GOAL_AREA_BR": self.left_goal_area_bottom_right,
            "R_GOAL_AREA_TL": self.right_goal_area_top_left,
            "R_GOAL_AREA_TR": self.right_goal_area_top_right,
            "R_GOAL_AREA_BL": self.right_goal_area_bottom_left,
            "R_GOAL_AREA_BR": self.right_goal_area_bottom_right,
            "L_GOAL_TL": self.left_goal_top_left_post,
            "L_GOAL_TR": self.left_goal_top_right_post,
            "L_GOAL_BL": self.left_goal_bottom_left_post,
            "L_GOAL_BR": self.left_goal_bottom_right_post,
            "R_GOAL_TL": self.right_goal_top_left_post,
            "R_GOAL_TR": self.right_goal_top_right_post,
            "R_GOAL_BL": self.right_goal_bottom_left_post,
            "R_GOAL_BR": self.right_goal_bottom_right_post,
            "T_TOUCH_AND_HALFWAY": self.halfway_and_top_touch_line_mark,
            "B_TOUCH_AND_HALFWAY": self.halfway_and_bottom_touch_line_mark,
            "T_HALFWAY_AND_CIRCLE": np.array([0.0, -self.CENTER_CIRCLE_RADIUS, 0.0], dtype='float'),
            "B_HALFWAY_AND_CIRCLE": np.array([0.0, self.CENTER_CIRCLE_RADIUS, 0.0], dtype='float'),
            "TL_16M_ARC": self.top_left_16M_arc,
            "BL_16M_ARC": self.bottom_left_16M_arc,
            "TR_16M_ARC": self.top_right_16M_arc,
            "BR_16M_ARC": self.bottom_right_16M_arc,
        }

        # Line extremities: maps class name → (start_3D, end_3D)
        self.line_extremities = {
            'Big rect. left bottom': (self.left_penalty_area_bottom_left, self.left_penalty_area_bottom_right),
            'Big rect. left top': (self.left_penalty_area_top_left, self.left_penalty_area_top_right),
            'Big rect. left main': (self.left_penalty_area_top_right, self.left_penalty_area_bottom_right),
            'Big rect. right bottom': (self.right_penalty_area_bottom_left, self.right_penalty_area_bottom_right),
            'Big rect. right top': (self.right_penalty_area_top_left, self.right_penalty_area_top_right),
            'Big rect. right main': (self.right_penalty_area_top_right, self.right_penalty_area_bottom_right),
            'Small rect. left bottom': (self.left_goal_area_bottom_left, self.left_goal_area_bottom_right),
            'Small rect. left top': (self.left_goal_area_top_left, self.left_goal_area_top_right),
            'Small rect. left main': (self.left_goal_area_top_right, self.left_goal_area_bottom_right),
            'Small rect. right bottom': (self.right_goal_area_bottom_left, self.right_goal_area_bottom_right),
            'Small rect. right top': (self.right_goal_area_top_left, self.right_goal_area_top_right),
            'Small rect. right main': (self.right_goal_area_top_right, self.right_goal_area_bottom_right),
            'Side line top': (self.top_left_corner, self.top_right_corner),
            'Side line bottom': (self.bottom_left_corner, self.bottom_right_corner),
            'Side line left': (self.top_left_corner, self.bottom_left_corner),
            'Side line right': (self.top_right_corner, self.bottom_right_corner),
            'Middle line': (self.halfway_and_top_touch_line_mark, self.halfway_and_bottom_touch_line_mark),
            'Goal left crossbar': (self.left_goal_top_left_post, self.left_goal_top_right_post),
            'Goal left post left': (self.left_goal_top_left_post, self.left_goal_bottom_left_post),
            'Goal left post right': (self.left_goal_top_right_post, self.left_goal_bottom_right_post),
            'Goal right crossbar': (self.right_goal_top_left_post, self.right_goal_top_right_post),
            'Goal right post left': (self.right_goal_top_left_post, self.right_goal_bottom_left_post),
            'Goal right post right': (self.right_goal_top_right_post, self.right_goal_bottom_right_post),
            'Circle right': (self.top_right_16M_arc, self.bottom_right_16M_arc),
            'Circle left': (self.top_left_16M_arc, self.bottom_left_16M_arc),
        }

    def sample_field_points(self, dist=0.1, dist_circles=0.2):
        """
        Sample pitch lines every `dist` meters.

        Args:
            dist: distance in meters between sample points on straight lines
            dist_circles: distance in meters between sample points on arcs

        Returns:
            dict: {class_name: [np.array([x, y, z]), ...]}
        """
        polylines = {}

        # Center circle
        center = self.center_mark
        from_angle = 0.0
        to_angle = 2 * np.pi
        x1 = center[0] + np.cos(from_angle) * self.CENTER_CIRCLE_RADIUS
        y1 = center[1] + np.sin(from_angle) * self.CENTER_CIRCLE_RADIUS
        polyline = [np.array([x1, y1, 0.0])]
        length = self.CENTER_CIRCLE_RADIUS * (to_angle - from_angle)
        nb_pts = int(length / dist_circles)
        dangle = dist_circles / self.CENTER_CIRCLE_RADIUS
        for i in range(1, nb_pts + 1):
            angle = from_angle + i * dangle
            x = center[0] + np.cos(angle) * self.CENTER_CIRCLE_RADIUS
            y = center[1] + np.sin(angle) * self.CENTER_CIRCLE_RADIUS
            polyline.append(np.array([x, y, 0.0]))
        polyline.append(np.array([x1, y1, 0.0]))  # close circle
        polylines['Circle central'] = polyline

        # All other lines
        for key, (start, end) in self.line_extremities.items():
            if 'Circle' in key:
                continue
            polyline = [start]
            total_dist = np.linalg.norm(end - start)
            nb_pts = max(2, int(total_dist / dist) - 1)
            v = (end - start) / np.linalg.norm(end - start)
            prev_pt = start
            for i in range(nb_pts):
                pt = prev_pt + dist * v
                polyline.append(pt)
                prev_pt = pt
            polyline.append(end)
            polylines[key] = polyline

        return polylines
