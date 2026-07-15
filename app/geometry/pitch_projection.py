from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import supervision as sv

from app.config.pitch import SoccerPitchConfiguration


MIN_KEYPOINT_CONFIDENCE = 0.35
MIN_KEYPOINTS_FOR_HOMOGRAPHY = 4
MIN_INLIERS_FOR_HOMOGRAPHY = 4
MAX_REPROJECTION_ERROR_PX = 12.0
FRAME_EDGE_MARGIN_PX = 8
MAX_STALE_SECONDS = 0.5


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

        self.prev_valid_homography: Optional[np.ndarray] = None
        self.stale_frames = 0

    def update(
        self,
        frame: np.ndarray,
        keypoints: sv.KeyPoints,
    ) -> PitchProjectionResult:
        observations = self._extract_model_observations(frame, keypoints)

        homography, inlier_labels, reprojection_error = self._estimate_homography(
            observations
        )

        homography_status = 'unavailable'
        projected_keypoints: List[ProjectedPitchKeypoint] = []
        display_tracking = list(observations.values())

        if homography is not None:
            homography_status = 'fresh'
            self.prev_valid_homography = homography
            self.stale_frames = 0
            inlier_set = set(inlier_labels)
            display_tracking = [
                observation
                for observation in observations.values()
                if observation.reference.label in inlier_set
            ]
            projected_keypoints = self._project_keypoints(
                homography,
                [
                    observations[label]
                    for label in inlier_labels
                ],
            )
        elif (
            self.prev_valid_homography is not None
            and self.stale_frames < self.max_stale_frames
        ):
            homography = self.prev_valid_homography
            self.stale_frames += 1
            homography_status = 'stale'
            reprojection_error = None
            projected_keypoints = []
        else:
            self.prev_valid_homography = None
            self.stale_frames = 0

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

            observations[reference.label] = PitchKeypointObservation(
                reference=reference,
                image_xy=(x_value, y_value),
                confidence=float(confidence[index]),
                source='model',
            )
        return observations

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
