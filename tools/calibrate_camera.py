#!/usr/bin/env python3
"""Calibrate and validate pinhole/fisheye models for a fixed wide-angle camera."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from app.field_registration.lens import LensCalibration, LensModel


@dataclass(frozen=True)
class ChessboardObservations:
    object_points: tuple[np.ndarray, ...]
    image_points: tuple[np.ndarray, ...]
    image_size: tuple[int, int]
    image_paths: tuple[Path, ...]


@dataclass(frozen=True)
class CalibrationFit:
    lens_model: LensModel
    camera_matrix: np.ndarray
    distortion: np.ndarray
    rms_px: float


@dataclass(frozen=True)
class ModelValidation:
    lens_model: LensModel
    median_error_px: float
    p95_error_px: float
    maximum_error_px: float
    fold_count: int
    corner_count: int

    @property
    def selection_score(self) -> float:
        return self.median_error_px + 0.25 * self.p95_error_px


def _image_paths(directory: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in extensions)


def collect_chessboard_observations(
    image_paths: Sequence[Path],
    pattern_size: tuple[int, int],
    square_size: float,
) -> ChessboardObservations:
    if not image_paths:
        raise ValueError("No calibration images were found")
    if min(pattern_size) < 2 or square_size <= 0.0:
        raise ValueError("pattern dimensions and square_size must be positive")
    object_template = np.zeros(
        (pattern_size[0] * pattern_size[1], 3), dtype=np.float32
    )
    object_template[:, :2] = np.mgrid[
        0 : pattern_size[0], 0 : pattern_size[1]
    ].T.reshape(-1, 2)
    object_template *= float(square_size)

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    valid_paths: list[Path] = []
    image_size: tuple[int, int] | None = None
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = current_size
        elif image_size != current_size:
            raise ValueError(
                "All calibration images must have the same size; "
                f"expected {image_size}, got {current_size} in {path}"
            )
        if hasattr(cv2, "findChessboardCornersSB"):
            found, corners = cv2.findChessboardCornersSB(gray, pattern_size)
        else:
            found, corners = cv2.findChessboardCorners(
                gray,
                pattern_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if found:
                corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30,
                        0.001,
                    ),
                )
        if found:
            object_points.append(object_template.copy())
            image_points.append(corners.astype(np.float32).reshape(-1, 1, 2))
            valid_paths.append(path)
    if image_size is None or len(object_points) < 3:
        raise ValueError(
            f"At least 3 valid chessboard images are required; found {len(object_points)}"
        )
    return ChessboardObservations(
        object_points=tuple(object_points),
        image_points=tuple(image_points),
        image_size=image_size,
        image_paths=tuple(valid_paths),
    )


def fit_camera_model(
    observations: ChessboardObservations,
    lens_model: LensModel,
    indices: Sequence[int] | None = None,
) -> CalibrationFit:
    selected = list(range(len(observations.object_points))) if indices is None else list(indices)
    if len(selected) < 3:
        raise ValueError("at least three views are required to fit a camera model")
    objects = [observations.object_points[index] for index in selected]
    images = [observations.image_points[index] for index in selected]
    if lens_model is LensModel.FISHEYE:
        fisheye_objects = [points.astype(np.float64).reshape(-1, 1, 3) for points in objects]
        fisheye_images = [points.astype(np.float64).reshape(-1, 1, 2) for points in images]
        camera_matrix = np.eye(3, dtype=np.float64)
        distortion = np.zeros((4, 1), dtype=np.float64)
        rms, camera_matrix, distortion, _, _ = cv2.fisheye.calibrate(
            fisheye_objects,
            fisheye_images,
            observations.image_size,
            camera_matrix,
            distortion,
            flags=(
                cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                | cv2.fisheye.CALIB_CHECK_COND
                | cv2.fisheye.CALIB_FIX_SKEW
            ),
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                100,
                1e-7,
            ),
        )
    else:
        rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
            objects,
            images,
            observations.image_size,
            None,
            None,
        )
    return CalibrationFit(
        lens_model=lens_model,
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        distortion=np.asarray(distortion, dtype=np.float64).reshape(-1),
        rms_px=float(rms),
    )


def _held_out_errors(
    fit: CalibrationFit,
    object_points: np.ndarray,
    image_points: np.ndarray,
) -> np.ndarray:
    objects = object_points.astype(np.float64).reshape(-1, 1, 3)
    observed = image_points.astype(np.float64).reshape(-1, 1, 2)
    if fit.lens_model is LensModel.FISHEYE:
        pose_points = cv2.fisheye.undistortPoints(
            observed,
            fit.camera_matrix,
            fit.distortion.reshape(4, 1),
            P=fit.camera_matrix,
        )
        success, rotation, translation = cv2.solvePnP(
            objects,
            pose_points,
            fit.camera_matrix,
            np.zeros(5),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            raise ValueError("held-out fisheye pose estimation failed")
        projected, _ = cv2.fisheye.projectPoints(
            objects,
            rotation,
            translation,
            fit.camera_matrix,
            fit.distortion.reshape(4, 1),
        )
    else:
        success, rotation, translation = cv2.solvePnP(
            objects,
            observed,
            fit.camera_matrix,
            fit.distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            raise ValueError("held-out pinhole pose estimation failed")
        projected, _ = cv2.projectPoints(
            objects,
            rotation,
            translation,
            fit.camera_matrix,
            fit.distortion,
        )
    return np.linalg.norm(projected.reshape(-1, 2) - observed.reshape(-1, 2), axis=1)


def cross_validate_camera_model(
    observations: ChessboardObservations,
    lens_model: LensModel,
    maximum_folds: int = 8,
) -> ModelValidation:
    view_count = len(observations.object_points)
    if view_count < 6:
        raise ValueError("hold-out model validation requires at least six valid views")
    fold_count = min(maximum_folds, view_count)
    if fold_count < 2:
        raise ValueError("maximum_folds must be at least two")
    holdouts = np.unique(
        np.rint(np.linspace(0, view_count - 1, fold_count)).astype(np.int64)
    )
    errors: list[float] = []
    successful_folds = 0
    for holdout in holdouts.tolist():
        train_indices = [index for index in range(view_count) if index != holdout]
        try:
            fit = fit_camera_model(observations, lens_model, train_indices)
            fold_errors = _held_out_errors(
                fit,
                observations.object_points[holdout],
                observations.image_points[holdout],
            )
        except (cv2.error, ValueError):
            continue
        if np.all(np.isfinite(fold_errors)):
            errors.extend(fold_errors.tolist())
            successful_folds += 1
    minimum_successful = max(2, int(np.ceil(len(holdouts) * 0.75)))
    if successful_folds < minimum_successful or not errors:
        raise ValueError(
            f"only {successful_folds}/{len(holdouts)} validation folds succeeded"
        )
    values = np.asarray(errors, dtype=np.float64)
    return ModelValidation(
        lens_model=lens_model,
        median_error_px=float(np.median(values)),
        p95_error_px=float(np.percentile(values, 95)),
        maximum_error_px=float(np.max(values)),
        fold_count=successful_folds,
        corner_count=int(values.size),
    )


def calibrate(
    image_paths: list[Path],
    pattern_size: tuple[int, int],
    square_size: float,
    lens_model: str = "pinhole",
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], float, int]:
    """Compatibility wrapper used by existing scripts and external callers."""

    observations = collect_chessboard_observations(
        image_paths, pattern_size, square_size
    )
    fit = fit_camera_model(observations, LensModel(lens_model))
    return (
        fit.camera_matrix,
        fit.distortion,
        observations.image_size,
        fit.rms_px,
        len(observations.object_points),
    )


def _select_candidate(
    validations: Sequence[ModelValidation],
    near_tie_px: float,
) -> ModelValidation:
    ordered = sorted(validations, key=lambda item: item.selection_score)
    best = ordered[0]
    pinhole = next(
        (item for item in ordered if item.lens_model is LensModel.PINHOLE), None
    )
    if pinhole is not None and pinhole.selection_score <= best.selection_score + near_tie_px:
        return pinhole
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("assets/calibration/camera.npz"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--pattern-cols", type=int, default=9)
    parser.add_argument("--pattern-rows", type=int, default=6)
    parser.add_argument("--square-size", type=float, default=1.0)
    parser.add_argument(
        "--lens-model",
        choices=("pinhole", "fisheye", "auto"),
        default="auto",
        help="auto compares held-out reprojection error; it never uses training RMS alone",
    )
    parser.add_argument("--validation-folds", type=int, default=8)
    parser.add_argument(
        "--near-tie-px",
        type=float,
        default=0.05,
        help="prefer the simpler pinhole model when validation scores are this close",
    )
    arguments = parser.parse_args()

    paths = _image_paths(arguments.images)
    observations = collect_chessboard_observations(
        paths,
        pattern_size=(arguments.pattern_cols, arguments.pattern_rows),
        square_size=arguments.square_size,
    )
    requested_models = (
        (LensModel.PINHOLE, LensModel.FISHEYE)
        if arguments.lens_model == "auto"
        else (LensModel(arguments.lens_model),)
    )
    if arguments.lens_model == "auto" and len(observations.object_points) < 6:
        raise ValueError(
            "--lens-model auto requires at least six valid chessboard views; "
            "capture more views or select a model explicitly"
        )
    fits: dict[LensModel, CalibrationFit] = {}
    validations: list[ModelValidation] = []
    failures: dict[str, str] = {}
    for model in requested_models:
        try:
            fit = fit_camera_model(observations, model)
            fits[model] = fit
            if len(observations.object_points) >= 6:
                validation = cross_validate_camera_model(
                    observations, model, arguments.validation_folds
                )
                validations.append(validation)
                print(
                    f"{model.value}: train RMS={fit.rms_px:.4f}px, "
                    f"holdout median={validation.median_error_px:.4f}px, "
                    f"P95={validation.p95_error_px:.4f}px"
                )
            else:
                print(
                    f"{model.value}: train RMS={fit.rms_px:.4f}px; "
                    "holdout validation unavailable"
                )
        except (cv2.error, ValueError) as error:
            failures[model.value] = str(error)
            print(f"{model.value} calibration failed: {error}")
    if not fits:
        raise RuntimeError("no camera model could be calibrated")
    selected_validation = (
        _select_candidate(validations, arguments.near_tie_px)
        if validations
        else None
    )
    selected_model = (
        selected_validation.lens_model
        if selected_validation is not None
        else next(iter(fits))
    )
    fit = fits[selected_model]
    calibration = LensCalibration(
        lens_model=selected_model,
        camera_matrix=fit.camera_matrix,
        distortion_coefficients=fit.distortion,
        image_size=observations.image_size,
        reprojection_error_px=fit.rms_px,
        validation_median_error_px=(
            selected_validation.median_error_px
            if selected_validation is not None
            else None
        ),
        validation_p95_error_px=(
            selected_validation.p95_error_px
            if selected_validation is not None
            else None
        ),
        valid_image_count=len(observations.object_points),
    )
    calibration.save(arguments.output)
    report = {
        "version": 1,
        "selected_model": selected_model.value,
        "selection_rule": "median_holdout_px + 0.25 * p95_holdout_px",
        "near_tie_policy": f"prefer pinhole within {arguments.near_tie_px}px",
        "image_size": list(observations.image_size),
        "source_image_count": len(paths),
        "valid_image_count": len(observations.object_points),
        "candidates": [],
        "failures": failures,
        "not_assessed": [
            "straight-line residual requires a separate line-validation set",
            "effective field of view must be checked on the deployment camera",
        ],
    }
    validations_by_model = {item.lens_model: item for item in validations}
    for model, candidate_fit in fits.items():
        validation = validations_by_model.get(model)
        report["candidates"].append(
            {
                "lens_model": model.value,
                "training_rms_px": candidate_fit.rms_px,
                "validation_median_px": (
                    validation.median_error_px if validation is not None else None
                ),
                "validation_p95_px": (
                    validation.p95_error_px if validation is not None else None
                ),
                "validation_maximum_px": (
                    validation.maximum_error_px if validation is not None else None
                ),
                "successful_folds": (
                    validation.fold_count if validation is not None else 0
                ),
                "heldout_corner_count": (
                    validation.corner_count if validation is not None else 0
                ),
                "selection_score": (
                    validation.selection_score if validation is not None else None
                ),
            }
        )
    report_path = arguments.report or arguments.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved {arguments.output}")
    print(f"Saved {report_path}")
    print(f"Selected lens model: {selected_model.value}")


if __name__ == "__main__":
    main()
