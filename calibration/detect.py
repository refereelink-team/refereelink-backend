# -*- coding: utf-8 -*-
"""
Segmentation-based pitch line detection.

Adapted from SoccerNet-Calibration (sn-calibration-main/src/detect_extremities.py).

Uses DeepLabV3 segmentation to classify each pixel into one of 28 pitch line classes,
then extracts line extremities (endpoints) from the detected blobs.
"""

from __future__ import annotations

import copy
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


def init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    """Initialize conv and BN layers."""
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class SegmentationNetwork:
    """
    DeepLabV3-based pitch segmentation network.

    Classifies each pixel into one of 29 classes (0=background, 1..28=pitch line classes).
    """

    def __init__(
        self,
        model_file: str,
        mean_file: str,
        std_file: str,
        num_classes: int = 29,
        width: int = 640,
        height: int = 360,
    ):
        """
        Args:
            model_file: path to .pth checkpoint
            mean_file: path to mean.npy
            std_file: path to std.npy
            num_classes: number of output classes (default 29 = 1 background + 28 line classes)
            width: input image width for model
            height: input image height for model
        """
        self.model_file = Path(model_file)
        self.mean_file = Path(mean_file)
        self.std_file = Path(std_file)
        self.num_classes = num_classes
        self.width = width
        self.height = height

        model = nn.DataParallel(deeplabv3_resnet50(pretrained=False, num_classes=num_classes))
        init_weight(
            model, nn.init.kaiming_normal_,
            nn.BatchNorm2d, 1e-3, 0.1, mode='fan_in')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(str(self.model_file), map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        self.model = model.to(self.device)

        self.mean = np.load(str(self.mean_file))
        self.std = np.load(str(self.std_file))

    def analyse_image(self, image: np.ndarray) -> np.ndarray:
        """
        Run segmentation on a BGR image.

        Args:
            image: BGR uint8 image

        Returns:
            semantic_mask: HxW uint8 array, values 0=background, 1..28=line classes
        """
        h, w = image.shape[:2]
        img = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        img = np.asarray(img, np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).to(self.device).unsqueeze(0)

        with torch.no_grad():
            output = self.model.forward(img.float())
        output = output['out'].data[0].cpu().numpy()
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        # Resize back to original resolution
        if output.shape[:2] != (h, w):
            output = cv2.resize(output, (w, h), interpolation=cv2.INTER_NEAREST)
        return output


def synthesize_mask(semantic_mask: np.ndarray, disk_radius: int = 6) -> List[np.ndarray]:
    """
    Fit circles on semantic mask blobs and return centers that have enough support.

    A circle center is kept only if the proportion of True pixels under the circle
    exceeds a threshold.

    Args:
        semantic_mask: boolean mask
        disk_radius: radius of circles to fit

    Returns:
        list of disk centers as np.ndarray([row, col])
    """
    mask = semantic_mask.copy().astype(np.uint8)
    points = np.transpose(np.nonzero(mask))
    disks = []

    while len(points) > 0:
        start = random.choice(points)
        dist = 10.0
        success = True
        while dist > 1.0:
            enough_support, center = _get_support_center(mask, start, disk_radius)
            if not enough_support:
                bad_point = np.round(center).astype(np.int32)
                cv2.circle(mask, (bad_point[1], bad_point[0]), disk_radius, 0, -1)
                success = False
            dist = np.sqrt(np.sum(np.square(center - start)))
            start = center
        if success:
            disks.append(np.round(start).astype(np.int32))
            cv2.circle(mask, (disks[-1][1], disks[-1][0]), disk_radius, 0, -1)
        points = np.transpose(np.nonzero(mask))

    return disks


def _get_support_center(
    mask: np.ndarray,
    start: np.ndarray,
    disk_radius: int,
    min_support: float = 0.1,
) -> Tuple[bool, np.ndarray]:
    """
    Return barycenter of True pixels under a circle, and whether support is sufficient.

    Args:
        mask: boolean mask
        start: (row, col) starting point
        disk_radius: circle radius
        min_support: minimum proportion of circle area that must be True

    Returns:
        (enough_support, barycenter)
    """
    x, y = int(start[0]), int(start[1])
    support_pixels = 1
    result = np.array([x, y], dtype=np.float64)
    xstart = max(0, x - disk_radius)
    xend = min(mask.shape[0] - 1, x + disk_radius)
    ystart = max(0, y - disk_radius)
    yend = min(mask.shape[1] - 1, y + disk_radius)

    for i in range(xstart, xend + 1):
        for j in range(ystart, yend + 1):
            dist = np.sqrt(np.square(x - i) + np.square(y - j))
            if dist < disk_radius and mask[i, j] > 0:
                support_pixels += 1
                result[0] += i
                result[1] += j

    support = support_pixels >= min_support * np.square(disk_radius) * np.pi
    result = result / max(support_pixels, 1)
    return support, result


def generate_class_synthesis(semantic_mask: np.ndarray, radius: int = 6) -> Dict[str, List[np.ndarray]]:
    """
    For each line class present in the semantic mask, synthesize circles over blobs.

    Args:
        semantic_mask: HxW uint8 array (0=background, 1..28=classes)
        radius: circle radius for synthesis

    Returns:
        buckets: {class_name: [disk_centers]} for each detected class
    """
    from calibration.pitch import SoccerPitch

    buckets: Dict[str, List[np.ndarray]] = dict()
    kernel = np.ones((5, 5), np.uint8)
    semantic_mask = cv2.erode(semantic_mask, kernel, iterations=1)

    for k, class_name in enumerate(SoccerPitch.lines_classes):
        mask = semantic_mask == k + 1
        if mask.sum() > 0:
            disk_list = synthesize_mask(mask.astype(np.uint8), radius)
            if len(disk_list):
                buckets[class_name] = disk_list

    return buckets


def join_points(point_list: List[np.ndarray], maxdist: float) -> List[List[np.ndarray]]:
    """
    Link close points into polylines.

    Points within `maxdist` of each other are linked sequentially to form polylines.

    Args:
        point_list: list of 2D points as np.ndarray
        maxdist: maximum distance between adjacent points in a polyline

    Returns:
        list of polylines, each polyline is a list of np.ndarray points
    """
    polylines = []
    if not len(point_list):
        return polylines

    head = point_list[0]
    tail = point_list[0]
    polyline = deque()
    polyline.append(point_list[0])
    remaining = copy.deepcopy(point_list[1:])

    while len(remaining) > 0:
        min_dist_tail = 1000.0
        min_dist_head = 1000.0
        best_head = -1
        best_tail = -1

        for j, point in enumerate(remaining):
            dist_tail = np.sqrt(np.sum(np.square(point - tail)))
            dist_head = np.sqrt(np.sum(np.square(point - head)))
            if dist_tail < min_dist_tail:
                min_dist_tail = dist_tail
                best_tail = j
            if dist_head < min_dist_head:
                min_dist_head = dist_head
                best_head = j

        if min_dist_head <= min_dist_tail and min_dist_head < maxdist:
            polyline.appendleft(remaining[best_head])
            head = polyline[0]
            remaining.pop(best_head)
        elif min_dist_tail < min_dist_head and min_dist_tail < maxdist:
            polyline.append(remaining[best_tail])
            tail = polyline[-1]
            remaining.pop(best_tail)
        else:
            polylines.append(list(polyline.copy()))
            head = remaining[0]
            tail = remaining[0]
            polyline = deque()
            polyline.append(head)
            remaining.pop(0)

    polylines.append(list(polyline))
    return polylines


def get_line_extremities(
    buckets: Dict[str, List[np.ndarray]],
    maxdist: float,
    width: int,
    height: int,
) -> Dict[str, List[Dict[str, float]]]:
    """
    Extract the longest polyline's endpoints for each line class.

    Args:
        buckets: {class_name: [disk_centers]} from generate_class_synthesis
        maxdist: max distance for join_points
        width: image width (for normalization)
        height: image height (for normalization)

    Returns:
        extremities: {class_name: [{'x': norm_x, 'y': norm_y}, {...}]} with normalized coords 0..1
    """
    extremities: Dict[str, List[Dict[str, float]]] = {}
    for class_name, disks_list in buckets.items():
        polyline_list = join_points(disks_list, maxdist)
        max_len = 0
        longest_polyline = []
        for polyline in polyline_list:
            if len(polyline) > max_len:
                max_len = len(polyline)
                longest_polyline = polyline

        if len(longest_polyline) >= 2:
            extremities[class_name] = [
                {'x': longest_polyline[0][1] / width, 'y': longest_polyline[0][0] / height},
                {'x': longest_polyline[-1][1] / width, 'y': longest_polyline[-1][0] / height},
            ]
    return extremities
