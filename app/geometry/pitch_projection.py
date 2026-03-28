from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import supervision as sv

from app.config.pitch import SoccerPitchConfiguration


MIN_KEYPOINT_CONFIDENCE = 0.35
MIN_KEYPOINTS_FOR_HOMOGRAPHY = 4
MIN_INLIERS_FOR_HOMOGRAPHY = 6
MAX_REPROJECTION_ERROR_PX = 12.0
FRAME_EDGE_MARGIN_PX = 8
CALIBRATION_OFFSET_PX = 3
MIN_LINE_BRIGHTNESS = 130
MAX_HUE_DELTA = 25.0
MAX_FLOW_OUTLIER_Z_SCORE = 2.0
MAX_STALE_SECONDS = 0.5
SYNTHESIS_MARGIN_RATIO = 0.2


@dataclass(frozen=True)
class PitchPointReference:
    index: int
    label: str
    world_xy: Tuple[float, float]
    color: str


@dataclass(frozen=True)
class PitchKeypointObservation:
    reference: PitchPointReference
    image_xy: Tuple[float, float]
    confidence: float
    source: str


@dataclass(frozen=True)
class ProjectedPitchKeypoint:
    reference: PitchPointReference
    projected_world_xy: Tuple[float, float]
    source: str


@dataclass
class PitchProjectionResult:
    tracking_observations: List[PitchKeypointObservation]
    projected_keypoints: List[ProjectedPitchKeypoint]
    homography: Optional[np.ndarray]
    homography_status: str
    reprojection_error: Optional[float]

    @property
    def available(self) -> bool:
        return self.homography is not None


def build_pitch_point_references(
    config: SoccerPitchConfiguration,
) -> List[PitchPointReference]:
    return [
        PitchPointReference(
            index=index,
            label=config.labels[index] if index < len(config.labels) else str(index + 1),
            world_xy=(
                float(config.vertices[index][0]),
                float(config.vertices[index][1]),
            ),
            color=config.colors[index % len(config.colors)],
        )
        for index in range(len(config.vertices))
    ]


class PitchProjectionEngine:
    def __init__(
        self,
        config: SoccerPitchConfiguration,
        fps: float,
        min_keypoint_confidence: float = MIN_KEYPOINT_CONFIDENCE,
        max_reprojection_error_px: float = MAX_REPROJECTION_ERROR_PX,
    ) -> None:
        self.config = config
        self.references = build_pitch_point_references(config)
        self.reference_by_label = {
            reference.label: reference for reference in self.references
        }
        self.references_by_index = {
            reference.index: reference for reference in self.references
        }
        self.min_keypoint_confidence = min_keypoint_confidence
        self.max_reprojection_error_px = max_reprojection_error_px
        self.max_stale_frames = max(1, int(round(max(fps, 1.0) * MAX_STALE_SECONDS)))
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_tracking_observations: Dict[str, PitchKeypointObservation] = {}
        self.prev_valid_homography: Optional[np.ndarray] = None
        self.stale_frames = 0

        self.vertical_groups = self._group_reference_labels(axis=0)
        self.horizontal_groups = self._group_reference_labels(axis=1)

    def update(
        self,
        frame: np.ndarray,
        keypoints: sv.KeyPoints,
    ) -> PitchProjectionResult:
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        model_observations = self._extract_model_observations(frame, keypoints)
        flow_observations = self._track_previous_observations(frame, current_gray)

        merged_tracking = dict(flow_observations)
        merged_tracking.update(model_observations)

        homography_input = self._build_homography_observations(frame, merged_tracking)
        homography, inlier_labels, reprojection_error = self._estimate_homography(
            homography_input
        )

        homography_status = 'unavailable'
        projected_keypoints: List[ProjectedPitchKeypoint] = []
        display_tracking = list(merged_tracking.values())
        if homography is not None:
            homography_status = 'fresh'
            self.prev_valid_homography = homography
            self.stale_frames = 0
            inlier_set = set(inlier_labels)
            display_tracking = [
                observation
                for observation in merged_tracking.values()
                if observation.reference.label in inlier_set
            ]
            projected_keypoints = self._project_keypoints(
                homography,
                [
                    homography_input[label]
                    for label in inlier_labels
                    if homography_input[label].source != 'synthetic'
                ],
            )
        elif self.prev_valid_homography is not None and self.stale_frames < self.max_stale_frames:
            homography = self.prev_valid_homography
            self.stale_frames += 1
            homography_status = 'stale'
            reprojection_error = None
            projected_keypoints = []
        else:
            self.prev_valid_homography = None
            self.stale_frames = 0

        self.prev_tracking_observations = {
            observation.reference.label: observation
            for observation in display_tracking
        }
        self.prev_gray = current_gray
        self.prev_frame = frame.copy()

        return PitchProjectionResult(
            tracking_observations=display_tracking,
            projected_keypoints=projected_keypoints,
            homography=homography,
            homography_status=homography_status,
            reprojection_error=reprojection_error,
        )

    def _extract_model_observations(
        self,
        frame: np.ndarray,
        keypoints: sv.KeyPoints,
    ) -> Dict[str, PitchKeypointObservation]:
        if keypoints.xy.size == 0:
            return {}

        xy = keypoints.xy[0]
        if keypoints.confidence is None:
            confidence = np.ones(xy.shape[0], dtype=np.float32)
        else:
            confidence = keypoints.confidence[0]

        observations: Dict[str, PitchKeypointObservation] = {}
        for index, point in enumerate(xy):
            reference = self.references_by_index.get(index)
            if reference is None:
                continue
            x_value = float(point[0])
            y_value = float(point[1])
            if x_value <= 1 or y_value <= 1:
                continue
            if confidence[index] < self.min_keypoint_confidence:
                continue
            if not self._is_inside_visible_frame(
                (x_value, y_value),
                frame.shape,
                margin=FRAME_EDGE_MARGIN_PX,
            ):
                continue
            calibrated = self._calibrate_to_pitch_line(frame, (x_value, y_value))
            if calibrated is None:
                continue

            observations[reference.label] = PitchKeypointObservation(
                reference=reference,
                image_xy=calibrated,
                confidence=float(confidence[index]),
                source='model',
            )
        return observations

    def _track_previous_observations(
        self,
        frame: np.ndarray,
        current_gray: np.ndarray,
    ) -> Dict[str, PitchKeypointObservation]:
        if (
            self.prev_gray is None
            or self.prev_frame is None
            or len(self.prev_tracking_observations) == 0
        ):
            return {}

        labels = list(self.prev_tracking_observations.keys())
        previous_points = np.array(
            [self.prev_tracking_observations[label].image_xy for label in labels],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        if previous_points.size == 0:
            return {}

        try:
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                current_gray,
                previous_points,
                None,
                **self.lk_params,
            )
        except cv2.error:
            return {}

        valid_indices = np.where(status[:, 0] == 1)[0]
        if len(valid_indices) == 0:
            return {}

        moved_previous = previous_points[valid_indices, 0, :]
        moved_current = new_points[valid_indices, 0, :]
        move_amounts = np.linalg.norm(moved_current - moved_previous, axis=1)
        mean_move = float(np.mean(move_amounts))
        std_move = float(np.std(move_amounts)) + 1e-6

        tracked: Dict[str, PitchKeypointObservation] = {}
        for local_index, point_index in enumerate(valid_indices):
            previous_point = moved_previous[local_index]
            current_point = moved_current[local_index]
            label = labels[point_index]
            z_score = (move_amounts[local_index] - mean_move) / std_move
            if z_score > MAX_FLOW_OUTLIER_Z_SCORE:
                continue
            if not self._is_inside_visible_frame(
                (float(current_point[0]), float(current_point[1])),
                frame.shape,
                margin=FRAME_EDGE_MARGIN_PX,
            ):
                continue
            if self._measure_hue_delta(
                self.prev_frame,
                frame,
                (float(previous_point[0]), float(previous_point[1])),
                (float(current_point[0]), float(current_point[1])),
            ) > MAX_HUE_DELTA:
                continue

            calibrated = self._calibrate_to_pitch_line(
                frame,
                (float(current_point[0]), float(current_point[1])),
            )
            if calibrated is None:
                continue

            previous_observation = self.prev_tracking_observations[label]
            tracked[label] = PitchKeypointObservation(
                reference=previous_observation.reference,
                image_xy=calibrated,
                confidence=previous_observation.confidence,
                source='flow',
            )
        return tracked

    def _build_homography_observations(
        self,
        frame: np.ndarray,
        observations: Dict[str, PitchKeypointObservation],
    ) -> Dict[str, PitchKeypointObservation]:
        merged = dict(observations)
        synthesized = self._synthesize_observations(frame.shape, merged)
        for label, observation in synthesized.items():
            if label not in merged:
                merged[label] = observation
        return merged

    def _synthesize_observations(
        self,
        frame_shape: Sequence[int],
        observations: Dict[str, PitchKeypointObservation],
    ) -> Dict[str, PitchKeypointObservation]:
        x_lines = self._fit_group_lines(observations, self.vertical_groups, axis=0)
        y_lines = self._fit_group_lines(observations, self.horizontal_groups, axis=1)

        synthesized: Dict[str, PitchKeypointObservation] = {}
        frame_height, frame_width = frame_shape[:2]
        margin = int(round(max(frame_width, frame_height) * SYNTHESIS_MARGIN_RATIO))
        for reference in self.references:
            if reference.label in observations:
                continue
            x_value = float(reference.world_xy[0])
            y_value = float(reference.world_xy[1])
            if x_value not in x_lines or y_value not in y_lines:
                continue
            intersection = self._intersect_lines(x_lines[x_value], y_lines[y_value])
            if intersection is None:
                continue
            x_coord, y_coord = intersection
            if not self._is_inside_extended_frame(
                (x_coord, y_coord),
                frame_shape,
                margin=margin,
            ):
                continue
            synthesized[reference.label] = PitchKeypointObservation(
                reference=reference,
                image_xy=(float(x_coord), float(y_coord)),
                confidence=0.0,
                source='synthetic',
            )
        return synthesized

    def _fit_group_lines(
        self,
        observations: Dict[str, PitchKeypointObservation],
        groups: Dict[float, List[str]],
        axis: int,
    ) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        lines: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
        for group_value, labels in groups.items():
            points = [
                observations[label].image_xy
                for label in labels
                if label in observations and observations[label].source != 'synthetic'
            ]
            if len(points) < 2:
                continue
            points_array = np.array(points, dtype=np.float32)
            vx, vy, x0, y0 = cv2.fitLine(
                points_array,
                cv2.DIST_L2,
                0,
                0.01,
                0.01,
            )
            lines[group_value] = (
                np.array([float(x0[0]), float(y0[0])], dtype=np.float32),
                np.array([float(vx[0]), float(vy[0])], dtype=np.float32),
            )
        return lines

    def _estimate_homography(
        self,
        observations: Dict[str, PitchKeypointObservation],
    ) -> Tuple[Optional[np.ndarray], List[str], Optional[float]]:
        if len(observations) < MIN_KEYPOINTS_FOR_HOMOGRAPHY:
            return None, [], None

        ordered = list(observations.values())
        image_points = np.array(
            [observation.image_xy for observation in ordered],
            dtype=np.float32,
        )
        world_points = np.array(
            [observation.reference.world_xy for observation in ordered],
            dtype=np.float32,
        )
        minimum_inliers = (
            MIN_INLIERS_FOR_HOMOGRAPHY
            if len(ordered) >= MIN_INLIERS_FOR_HOMOGRAPHY
            else len(ordered)
        )

        for method in (cv2.RANSAC, cv2.RHO, cv2.LMEDS):
            try:
                homography, mask = cv2.findHomography(
                    image_points,
                    world_points,
                    method,
                    5.0,
                )
            except cv2.error:
                homography = None
                mask = None

            if homography is None:
                continue

            inlier_mask = (
                np.ones(len(ordered), dtype=bool)
                if mask is None
                else mask.flatten().astype(bool)
            )
            if int(np.count_nonzero(inlier_mask)) < max(
                MIN_KEYPOINTS_FOR_HOMOGRAPHY,
                minimum_inliers,
            ):
                continue

            try:
                inverse_homography = np.linalg.inv(homography)
            except np.linalg.LinAlgError:
                continue

            reprojected = cv2.perspectiveTransform(
                world_points.reshape(-1, 1, 2),
                inverse_homography,
            ).reshape(-1, 2)
            reprojection_error = np.linalg.norm(reprojected - image_points, axis=1)
            mean_error = float(np.mean(reprojection_error[inlier_mask]))
            if mean_error > self.max_reprojection_error_px:
                continue

            inlier_labels = [
                ordered[index].reference.label
                for index, is_inlier in enumerate(inlier_mask)
                if is_inlier
            ]
            return homography, inlier_labels, mean_error

        return None, [], None

    def _project_keypoints(
        self,
        homography: np.ndarray,
        observations: Sequence[PitchKeypointObservation],
    ) -> List[ProjectedPitchKeypoint]:
        if len(observations) == 0:
            return []
        image_points = np.array(
            [observation.image_xy for observation in observations],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        projected_points = cv2.perspectiveTransform(image_points, homography).reshape(-1, 2)
        return [
            ProjectedPitchKeypoint(
                reference=observation.reference,
                projected_world_xy=(float(projected_points[index][0]), float(projected_points[index][1])),
                source=observation.source,
            )
            for index, observation in enumerate(observations)
        ]

    def _group_reference_labels(self, axis: int) -> Dict[float, List[str]]:
        grouped: Dict[float, List[str]] = {}
        for reference in self.references:
            value = float(reference.world_xy[axis])
            grouped.setdefault(value, []).append(reference.label)
        return grouped

    def _measure_hue_delta(
        self,
        previous_frame: np.ndarray,
        current_frame: np.ndarray,
        previous_point: Tuple[float, float],
        current_point: Tuple[float, float],
    ) -> float:
        previous_hue = self._mean_patch_hue(previous_frame, previous_point)
        current_hue = self._mean_patch_hue(current_frame, current_point)
        return abs(previous_hue - current_hue)

    def _mean_patch_hue(
        self,
        frame: np.ndarray,
        point: Tuple[float, float],
    ) -> float:
        x_coord = int(round(point[0]))
        y_coord = int(round(point[1]))
        x_min = max(0, x_coord - 1)
        x_max = min(frame.shape[1], x_coord + 2)
        y_min = max(0, y_coord - 1)
        y_max = min(frame.shape[0], y_coord + 2)
        patch = frame[y_min:y_max, x_min:x_max]
        if patch.size == 0:
            return 0.0
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        return float(np.mean(patch_hsv[:, :, 0]))

    def _calibrate_to_pitch_line(
        self,
        frame: np.ndarray,
        point: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        x_coord = int(round(point[0]))
        y_coord = int(round(point[1]))
        if not self._is_inside_visible_frame(
            (x_coord, y_coord),
            frame.shape,
            margin=FRAME_EDGE_MARGIN_PX,
        ):
            return None

        x_min = max(0, x_coord - CALIBRATION_OFFSET_PX)
        x_max = min(frame.shape[1], x_coord + CALIBRATION_OFFSET_PX + 1)
        y_min = max(0, y_coord - CALIBRATION_OFFSET_PX)
        y_max = min(frame.shape[0], y_coord + CALIBRATION_OFFSET_PX + 1)
        patch = frame[y_min:y_max, x_min:x_max]
        if patch.size == 0:
            return None

        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        value = patch_hsv[:, :, 2]
        saturation = patch_hsv[:, :, 1]
        bright_index = np.unravel_index(np.argmax(value), value.shape)
        max_brightness = int(value[bright_index])
        saturation_at_max = int(saturation[bright_index])
        if max_brightness < MIN_LINE_BRIGHTNESS or saturation_at_max > 150:
            return None

        adjusted_x = np.clip(x_min + bright_index[1], 0, frame.shape[1] - 1)
        adjusted_y = np.clip(y_min + bright_index[0], 0, frame.shape[0] - 1)
        if not self._is_inside_visible_frame(
            (float(adjusted_x), float(adjusted_y)),
            frame.shape,
            margin=FRAME_EDGE_MARGIN_PX,
        ):
            return None

        return float(adjusted_x), float(adjusted_y)

    def _intersect_lines(
        self,
        first_line: Tuple[np.ndarray, np.ndarray],
        second_line: Tuple[np.ndarray, np.ndarray],
    ) -> Optional[Tuple[float, float]]:
        first_point, first_direction = first_line
        second_point, second_direction = second_line
        system = np.array(
            [
                [float(first_direction[0]), -float(second_direction[0])],
                [float(first_direction[1]), -float(second_direction[1])],
            ],
            dtype=np.float32,
        )
        determinant = float(np.linalg.det(system))
        if abs(determinant) < 1e-5:
            return None
        rhs = second_point - first_point
        try:
            solution = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            return None
        intersection = first_point + solution[0] * first_direction
        return float(intersection[0]), float(intersection[1])

    def _is_inside_visible_frame(
        self,
        point: Tuple[float, float],
        frame_shape: Sequence[int],
        margin: int,
    ) -> bool:
        height = int(frame_shape[0])
        width = int(frame_shape[1])
        return (
            margin <= point[0] < width - margin
            and margin <= point[1] < height - margin
        )

    def _is_inside_extended_frame(
        self,
        point: Tuple[float, float],
        frame_shape: Sequence[int],
        margin: int,
    ) -> bool:
        height = int(frame_shape[0])
        width = int(frame_shape[1])
        return (
            -margin <= point[0] < width + margin
            and -margin <= point[1] < height + margin
        )
