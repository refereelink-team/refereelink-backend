# -*- coding: utf-8 -*-
"""
Camera model: handles camera pose estimation from homography and 3D→2D projection.

Adapted from SoccerNet-Calibration (sn-calibration-main/src/camera.py).
Based on Multiple View Geometry (Hartley & Zisserman).
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from calibration.pitch import SoccerPitch


def pan_tilt_roll_to_orientation(pan: float, tilt: float, roll: float) -> np.ndarray:
    """Convert pan/tilt/roll euler angles to orientation rotation matrix."""
    Rpan = np.array([
        [np.cos(pan), -np.sin(pan), 0.0],
        [np.sin(pan), np.cos(pan), 0.0],
        [0.0, 0.0, 1.0]])
    Rroll = np.array([
        [np.cos(roll), -np.sin(roll), 0.0],
        [np.sin(roll), np.cos(roll), 0.0],
        [0.0, 0.0, 1.0]])
    Rtilt = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(tilt), -np.sin(tilt)],
        [0.0, np.sin(tilt), np.cos(tilt)]])
    return Rpan @ (Rtilt @ Rroll)


def rotation_matrix_to_pan_tilt_roll(rotation: np.ndarray) -> tuple:
    """Decompose rotation matrix into pan, tilt, roll (radians)."""
    orientation = np.transpose(rotation)
    first_tilt = np.arccos(np.clip(orientation[2, 2], -1.0, 1.0))
    second_tilt = -first_tilt

    sign_first = 1.0 if np.sin(first_tilt) > 0 else -1.0
    sign_second = 1.0 if np.sin(second_tilt) > 0 else -1.0

    first_pan = np.arctan2(sign_first * orientation[0, 2], sign_first * -orientation[1, 2])
    second_pan = np.arctan2(sign_second * orientation[0, 2], sign_second * -orientation[1, 2])
    first_roll = np.arctan2(sign_first * orientation[2, 0], sign_first * orientation[2, 1])
    second_roll = np.arctan2(sign_second * orientation[2, 0], sign_second * orientation[2, 1])

    if np.fabs(first_roll) < np.fabs(second_roll):
        return first_pan, first_tilt, first_roll
    return second_pan, second_tilt, second_roll


class Camera:
    """
    Camera model with intrinsic calibration, rotation, and translation.

    The camera pose is estimated from a homography that maps the 3D pitch plane
    (Z=0) to the image plane. This is then used to project any 3D world point to 2D image.
    """

    def __init__(self, iwidth: int = 960, iheight: int = 540):
        self.position = np.zeros(3)
        self.rotation = np.eye(3)
        self.calibration = np.eye(3)
        self.radial_distortion = np.zeros(6)
        self.thin_prism_disto = np.zeros(4)
        self.tangential_disto = np.zeros(2)
        self.image_width = iwidth
        self.image_height = iheight
        self.xfocal_length = 1.0
        self.yfocal_length = 1.0
        self.principal_point = (self.image_width / 2.0, self.image_height / 2.0)

    def solve_pnp(self, point_matches: list) -> None:
        """
        Recover rotation and translation from 3D-2D point correspondences.

        Args:
            point_matches: list of ((X, Y, Z), (x, y)) pairs
        """
        target_pts = np.array([pt[0] for pt in point_matches], dtype=np.float64)
        src_pts = np.array([pt[1] for pt in point_matches], dtype=np.float64)
        _, rvec, t, inliers = cv2.solvePnPRansac(
            target_pts, src_pts, self.calibration, None)
        self.rotation, _ = cv2.Rodrigues(rvec)
        self.position = -np.transpose(self.rotation) @ t.flatten()

    def refine_camera(self, point_matches: list) -> None:
        """
        Non-linear refinement of camera pose.

        Args:
            point_matches: list of ((X, Y, Z), (x, y)) pairs
        """
        rvec, _ = cv2.Rodrigues(self.rotation)
        target_pts = np.array([pt[0] for pt in point_matches], dtype=np.float64)
        src_pts = np.array([pt[1] for pt in point_matches], dtype=np.float64)
        rvec, t = cv2.solvePnPRefineLM(
            target_pts, src_pts, self.calibration, None,
            rvec, -self.rotation @ self.position,
            (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 20000, 0.00001))
        self.rotation, _ = cv2.Rodrigues(rvec)
        self.position = -np.transpose(self.rotation) @ t

    def from_homography(self, homography: np.ndarray) -> bool:
        """
        Initialize camera parameters from homography between 3D pitch plane and image.

        Based on Algorithm 8.2 + Example 8.1 from "Multiple View Geometry" (Hartley & Zisserman).

        Args:
            homography: 3x3 homography matrix (image = H @ pitch_3D_hom)

        Returns:
            True if successful, False if the calibration matrix could not be estimated
        """
        success, K = self.estimate_calibration_matrix_from_plane_homography(homography)
        if not success:
            return False

        hprim = np.linalg.inv(self.calibration) @ homography
        lambda1 = 1.0 / np.linalg.norm(hprim[:, 0])
        lambda2 = 1.0 / np.linalg.norm(hprim[:, 1])
        lambda3 = np.sqrt(lambda1 * lambda2)

        r0 = hprim[:, 0] * lambda1
        r1 = hprim[:, 1] * lambda2
        r2 = np.cross(r0, r1)

        R = np.column_stack((r0, r1, r2))
        u, s, vh = np.linalg.svd(R)
        R = u @ vh
        if np.linalg.det(R) < 0:
            u[:, 2] *= -1
            R = u @ vh
        self.rotation = R
        t = hprim[:, 2] * lambda3
        self.position = -np.transpose(self.rotation) @ t
        return True

    def estimate_calibration_matrix_from_plane_homography(self, homography: np.ndarray) -> tuple:
        """
        Extract calibration matrix K from homography using V View Geometry constraints.

        Returns:
            (success, K)
        """
        H = homography.reshape(9)
        A = np.zeros((5, 6))
        A[0, 1] = 1.0
        A[1, 0] = 1.0
        A[1, 2] = -1.0
        A[2, 3] = self.principal_point[1] / self.principal_point[0]
        A[2, 4] = -1.0
        A[3, 0] = H[0] * H[1]
        A[3, 1] = H[0] * H[4] + H[1] * H[3]
        A[3, 2] = H[3] * H[4]
        A[3, 3] = H[0] * H[7] + H[1] * H[6]
        A[3, 4] = H[3] * H[7] + H[4] * H[6]
        A[3, 5] = H[6] * H[7]
        A[4, 0] = H[0] * H[0] - H[1] * H[1]
        A[4, 1] = 2 * H[0] * H[3] - 2 * H[1] * H[4]
        A[4, 2] = H[3] * H[3] - H[4] * H[4]
        A[4, 3] = 2 * H[0] * H[6] - 2 * H[1] * H[7]
        A[4, 4] = 2 * H[3] * H[6] - 2 * H[4] * H[7]
        A[4, 5] = H[6] * H[6] - H[7] * H[7]

        _, _, vh = np.linalg.svd(A)
        w = vh[-1]
        W = np.zeros((3, 3))
        W[0, 0] = w[0] / w[5]
        W[0, 1] = w[1] / w[5]
        W[0, 2] = w[3] / w[5]
        W[1, 0] = w[1] / w[5]
        W[1, 1] = w[2] / w[5]
        W[1, 2] = w[4] / w[5]
        W[2, 0] = w[3] / w[5]
        W[2, 1] = w[4] / w[5]
        W[2, 2] = w[5] / w[5]

        try:
            Ktinv = np.linalg.cholesky(W)
        except np.linalg.LinAlgError:
            K = np.eye(3)
            return False, K

        K = np.linalg.inv(np.transpose(Ktinv))
        K /= K[2, 2]
        self.xfocal_length = K[0, 0]
        self.yfocal_length = K[1, 1]
        self.principal_point = (self.image_width / 2.0, self.image_height / 2.0)
        self.calibration = np.array([
            [self.xfocal_length, 0, self.principal_point[0]],
            [0, self.yfocal_length, self.principal_point[1]],
            [0, 0, 1]
        ], dtype='float')
        return True, K

    def distort(self, point: np.ndarray) -> np.ndarray:
        """
        Apply lens distortion to a normalized image point.

        Args:
            point: 2D normalized point (x, y)

        Returns:
            distorted normalized point
        """
        x, y = point[0], point[1]
        radius_sq = x * x + y * y
        numerator = 1.0
        denominator = 1.0
        for i in range(3):
            k = self.radial_distortion[i]
            numerator += k * radius_sq ** (i + 1)
            k2n = self.radial_distortion[i + 3]
            denominator += k2n * radius_sq ** (i + 1)
        radial_factor = numerator / denominator
        xpp = (x * radial_factor +
               2 * self.tangential_disto[0] * x * y +
               self.tangential_disto[1] * (radius_sq + 2 * x * x) +
               self.thin_prism_disto[0] * radius_sq +
               self.thin_prism_disto[1] * radius_sq ** 2)
        ypp = (y * radial_factor +
               2 * self.tangential_disto[1] * x * y +
               self.tangential_disto[0] * (radius_sq + 2 * y * y) +
               self.thin_prism_disto[2] * radius_sq +
               self.thin_prism_disto[3] * radius_sq ** 2)
        return np.array([xpp, ypp], dtype=np.float32)

    def project_point(self, point3D: np.ndarray, distort: bool = True) -> np.ndarray:
        """
        Project a 3D world point to 2D image coordinates.

        Args:
            point3D: 3D world point (X, Y, Z)
            distort: whether to apply lens distortion

        Returns:
            2D homogeneous image point (x, y, 1) or zeros if behind camera
        """
        point3D = np.asarray(point3D).flatten()
        point = point3D - self.position
        rotated = self.rotation @ np.transpose(point)
        if rotated[2] <= 1e-3:
            return np.zeros(3)
        rotated = rotated / rotated[2]
        if distort:
            distorted = self.distort(rotated[:2])
        else:
            distorted = rotated[:2]
        x = distorted[0] * self.xfocal_length + self.principal_point[0]
        y = distorted[1] * self.yfocal_length + self.principal_point[1]
        return np.array([x, y, 1.0])

    def scale_resolution(self, factor: float) -> None:
        """Adapt internal parameters for image resolution scaling."""
        self.xfocal_length *= factor
        self.yfocal_length *= factor
        self.image_width *= factor
        self.image_height *= factor
        self.principal_point = (self.image_width / 2.0, self.image_height / 2.0)
        self.calibration = np.array([
            [self.xfocal_length, 0, self.principal_point[0]],
            [0, self.yfocal_length, self.principal_point[1]],
            [0, 0, 1]
        ], dtype='float')

    def draw_corners(self, image: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
        """Draw pitch corner points on image."""
        field = SoccerPitch()
        for pt3D in field.point_dict.values():
            projected = self.project_point(pt3D)
            if projected[2] == 0.0:
                continue
            projected = projected / projected[2]
            if 0 < projected[0] < self.image_width and 0 < projected[1] < self.image_height:
                cv2.circle(image, (int(projected[0]), int(projected[1])), 3, color, 2)
        return image

    def draw_pitch(self, image: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
        """Draw full pitch lines on image."""
        field = SoccerPitch()
        polylines = field.sample_field_points()
        for line in polylines.values():
            prev_point = self.project_point(line[0])
            for point in line[1:]:
                projected = self.project_point(point)
                if projected[2] == 0.0:
                    continue
                projected = projected / projected[2]
                if 0 < projected[0] < self.image_width and 0 < projected[1] < self.image_height:
                    cv2.line(
                        image,
                        (int(prev_point[0]), int(prev_point[1])),
                        (int(projected[0]), int(projected[1])),
                        color, 1)
                prev_point = projected
        return image

    def draw_colorful_pitch(self, image: np.ndarray) -> np.ndarray:
        """Draw pitch lines colored by class."""
        field = SoccerPitch()
        polylines = field.sample_field_points()
        for key, line in polylines.items():
            if key not in field.palette:
                continue
            bgr = field.palette[key]
            color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
            prev_point = self.project_point(line[0])
            for point in line[1:]:
                projected = self.project_point(point)
                if projected[2] == 0.0:
                    continue
                projected = projected / projected[2]
                if 0 < projected[0] < self.image_width and 0 < projected[1] < self.image_height:
                    cv2.line(
                        image,
                        (int(prev_point[0]), int(prev_point[1])),
                        (int(projected[0]), int(projected[1])),
                        color, 1)
                prev_point = projected
        return image

    def to_json_parameters(self) -> dict:
        """Serialize camera parameters to JSON-compatible dict."""
        pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(self.rotation)
        return {
            "pan_degrees": pan * 180.0 / np.pi,
            "tilt_degrees": tilt * 180.0 / np.pi,
            "roll_degrees": roll * 180.0 / np.pi,
            "position_meters": self.position.tolist(),
            "x_focal_length": float(self.xfocal_length),
            "y_focal_length": float(self.yfocal_length),
            "principal_point": list(self.principal_point),
            "radial_distortion": self.radial_distortion.tolist(),
            "tangential_distortion": self.tangential_disto.tolist(),
            "thin_prism_distortion": self.thin_prism_disto.tolist(),
        }

    def from_json_parameters(self, calib_dict: dict) -> None:
        """Load camera parameters from JSON-compatible dict."""
        self.principal_point = tuple(calib_dict["principal_point"])
        self.image_width = 2 * self.principal_point[0]
        self.image_height = 2 * self.principal_point[1]
        self.xfocal_length = calib_dict["x_focal_length"]
        self.yfocal_length = calib_dict["y_focal_length"]
        self.calibration = np.array([
            [self.xfocal_length, 0, self.principal_point[0]],
            [0, self.yfocal_length, self.principal_point[1]],
            [0, 0, 1]
        ], dtype='float')
        pan = calib_dict['pan_degrees'] * np.pi / 180.0
        tilt = calib_dict['tilt_degrees'] * np.pi / 180.0
        roll = calib_dict['roll_degrees'] * np.pi / 180.0
        self.rotation = pan_tilt_roll_to_orientation(pan, tilt, roll)
        self.rotation = np.transpose(self.rotation)
        self.position = np.array(calib_dict['position_meters'], dtype='float')
        self.radial_distortion = np.array(calib_dict['radial_distortion'], dtype='float')
        self.tangential_disto = np.array(calib_dict['tangential_distortion'], dtype='float')
        self.thin_prism_disto = np.array(calib_dict['thin_prism_distortion'], dtype='float')

    def save(self, path: str) -> None:
        """Save camera to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_json_parameters(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Camera":
        """Load camera from JSON file."""
        with open(path) as f:
            data = json.load(f)
        cam = cls()
        cam.from_json_parameters(data)
        return cam
