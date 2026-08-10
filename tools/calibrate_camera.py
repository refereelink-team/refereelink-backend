#!/usr/bin/env python3
"""Calibrate a fixed wide-angle camera using chessboard images.

Example:
    python tools/calibrate_camera.py --images calibration_images
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _image_paths(directory: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in extensions)


def calibrate(
    image_paths: list[Path],
    pattern_size: tuple[int, int],
    square_size: float,
    lens_model: str = "pinhole",
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], float, int]:
    if not image_paths:
        raise ValueError("No calibration images were found")

    object_points = np.zeros(
        (pattern_size[0] * pattern_size[1], 3), dtype=np.float32
    )
    object_points[:, :2] = np.mgrid[
        0 : pattern_size[0], 0 : pattern_size[1]
    ].T.reshape(-1, 2)
    object_points *= float(square_size)

    object_points_all: list[np.ndarray] = []
    image_points_all: list[np.ndarray] = []
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
                f"All calibration images must have the same size; "
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
            object_points_all.append(object_points.copy())
            image_points_all.append(corners.astype(np.float32))

    if image_size is None or len(object_points_all) < 3:
        raise ValueError(
            f"At least 3 valid chessboard images are required; found {len(object_points_all)}"
        )

    if lens_model == "fisheye":
        fisheye_objects = [
            points.astype(np.float64).reshape(-1, 1, 3) for points in object_points_all
        ]
        fisheye_images = [
            points.astype(np.float64).reshape(-1, 1, 2) for points in image_points_all
        ]
        camera_matrix = np.eye(3, dtype=np.float64)
        distortion = np.zeros((4, 1), dtype=np.float64)
        rms, camera_matrix, distortion, _, _ = cv2.fisheye.calibrate(
            fisheye_objects,
            fisheye_images,
            image_size,
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
    elif lens_model == "pinhole":
        rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
            object_points_all,
            image_points_all,
            image_size,
            None,
            None,
        )
    else:
        raise ValueError("lens_model must be 'pinhole' or 'fisheye'")
    return camera_matrix, distortion, image_size, float(rms), len(object_points_all)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a fixed wide-angle camera")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/calibration/camera.npz"),
    )
    parser.add_argument("--pattern-cols", type=int, default=9)
    parser.add_argument("--pattern-rows", type=int, default=6)
    parser.add_argument("--square-size", type=float, default=1.0)
    parser.add_argument(
        "--lens-model",
        choices=("pinhole", "fisheye", "auto"),
        default="auto",
        help="Fit one model or compare both by chessboard reprojection RMS",
    )
    args = parser.parse_args()

    paths = _image_paths(args.images)
    models = ("pinhole", "fisheye") if args.lens_model == "auto" else (args.lens_model,)
    candidates = []
    for lens_model in models:
        try:
            result = calibrate(
                paths,
                pattern_size=(args.pattern_cols, args.pattern_rows),
                square_size=args.square_size,
                lens_model=lens_model,
            )
            candidates.append((float(result[3]), lens_model, result))
            print(f"{lens_model} RMS: {result[3]:.4f}")
        except (cv2.error, ValueError) as exc:
            print(f"{lens_model} calibration failed: {exc}")
    if not candidates:
        raise RuntimeError("neither camera model could be calibrated")
    _, selected_model, selected = min(candidates, key=lambda item: item[0])
    matrix, distortion, image_size, rms, valid_count = selected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        camera_matrix=matrix,
        distortion_coefficients=distortion,
        image_width=np.array(image_size[0], dtype=np.int32),
        image_height=np.array(image_size[1], dtype=np.int32),
        reprojection_error=np.array(rms, dtype=np.float64),
        lens_model=np.array(selected_model),
    )
    print(f"Saved {args.output}")
    print(f"Valid images: {valid_count}/{len(paths)}")
    print(f"Image size: {image_size[0]}x{image_size[1]}")
    print(f"RMS reprojection error: {rms:.4f}")
    print(f"Selected lens model: {selected_model}")


if __name__ == "__main__":
    main()
