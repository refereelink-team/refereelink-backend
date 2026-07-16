#!/usr/bin/env python3
"""Benchmark the deployment-side frame encoding path.

The vision-model benchmark remains in ``tools/benchmark_phase1.py`` because
it requires real checkpoints. This script measures the deterministic shared
JPEG cache used by all MJPEG clients and can run on any machine.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from app.services.frame_encoder import LatestJpegFrame


def run(frames: int = 300, width: int = 1280, height: int = 720, quality: int = 75) -> dict:
    encoder = LatestJpegFrame(quality=quality)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    start = time.perf_counter()
    for index in range(frames):
        frame[0, 0, 0] = index % 255
        encoder.update(frame)
    elapsed = time.perf_counter() - start
    return {
        "frames": frames,
        "resolution": [width, height],
        "quality": quality,
        "elapsed_sec": round(elapsed, 4),
        "encode_fps": round(frames / max(elapsed, 1e-9), 2),
        "frames_encoded": encoder.frames_encoded,
        "average_encode_latency_ms": round(encoder.average_encode_time_ms, 3),
        "latest_jpeg_bytes": len(encoder.latest or b""),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--quality", type=int, default=75)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.frames, args.width, args.height, args.quality)
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
