# Demo Showcase (Web): Offside Review + Multiview Dashboard

This PR adds two **browser (Web)** demo UIs used for presentation:

| UI | Port (default) | Entry |
|---|---|---|
| Offside human review + 2D pitch / offside line | `6006` | `python main.py demo-offside review ...` |
| Multiview dashboard + fixed-H 2D projection + foul assist tip | `6008` | `python main.py demo-dashboard ...` |

These are FastAPI + HTML pages (not PySide). Open them in a browser after starting the servers.

## Prerequisites

- Python env with: `fastapi`, `uvicorn`, `opencv-python`, `numpy`, `pyyaml`, `torch`, `ultralytics`, `supervision` (dashboard preprocess / offside discover)
- `ffmpeg` on `PATH` (or set `FFMPEG_BIN`)
- YOLO weights under `assets/weights/` (at least `yolo11s.pt`)
- Showcase clips + track caches are committed under `demo_data/` (~35MB) so a fresh pull can open the same Web demos.

Optional env overrides:

```bash
export DEMO_DASHBOARD_DIR="$PWD/demo_data/dashboard"
export DEMO_OFFSIDE_REVIEW_DIR="$PWD/demo_data/offside/review"
export FFMPEG_BIN="$(command -v ffmpeg)"
```

## Included media (`demo_data/`)

```text
demo_data/offside/offside.mp4
demo_data/offside/review/frame_index.jsonl
demo_data/dashboard/main_foul_tracking.mp4
demo_data/dashboard/main_frames.jsonl
demo_data/dashboard/manual_homography.json
demo_data/dashboard/foul_roi.json
demo_data/multiview/videos/side1_tracking.mp4
demo_data/multiview/videos/side2_tracking.mp4
demo_data/multiview/metadata/side{1,2}_frames.jsonl
```

## 1) Multiview dashboard (6008)

```bash
python main.py demo-dashboard \
  --config configs/demo_dashboard.yaml \
  --host 0.0.0.0 --port 6008
```

Open `http://127.0.0.1:6008/`

Features: 3-camera sync clock, virtual camera wheel (赛3–赛8 placeholders), quiet 4+4 main-camera calib → 2D pitch, foul assist tip at ~4.5s with detection-box red flash.

## 2) Offside review (6006)

```bash
export DEMO_OFFSIDE_REVIEW_DIR="$PWD/demo_data/offside/review"
python main.py demo-offside review \
  --input demo_data/offside/offside.mp4 \
  --discovery-dir demo_data/offside/discovery \
  --config configs/offside_demo.yaml \
  --host 0.0.0.0 --port 6006 --device cuda
```

Open `http://127.0.0.1:6006/`

Features: frame scrub, quiet 4+4 calib, 2D pitch, role confirm, second-last defender offside line + OFFSIDE overlay.

## Notes for reviewers

- Showcase UIs are **preprocess / cache driven** for the dashboard (no live YOLO while scrubbing).
- Offside formal judge still requires human confirmation fields; review UI provides geometric assist.
- `*.mp4` is gitignored globally; showcase clips are force-added under `demo_data/`.
