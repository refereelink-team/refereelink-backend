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
- Preprocessed demo media under `demo_data/` (videos + jsonl track caches). Large binaries are **not** committed.

Optional env overrides:

```bash
export DEMO_DASHBOARD_DIR="$PWD/demo_outputs/dashboard"
export DEMO_OFFSIDE_REVIEW_DIR="$PWD/demo_outputs/offside/review"
export FFMPEG_BIN="$(command -v ffmpeg)"
```

## 1) Multiview dashboard (6008)

1. Prepare config:

```bash
cp configs/demo_dashboard.example.yaml configs/demo_dashboard.yaml
# edit video/metadata paths to your demo_data
```

2. Expected media (example layout):

```text
demo_data/dashboard/main_foul_tracking.mp4
demo_data/dashboard/main_frames.jsonl          # players[].xyxy + track_id
demo_data/multiview/videos/side1_tracking.mp4
demo_data/multiview/metadata/side1_frames.jsonl
demo_data/multiview/videos/side2_tracking.mp4
demo_data/multiview/metadata/side2_frames.jsonl
```

3. Start:

```bash
python main.py demo-dashboard \
  --config configs/demo_dashboard.yaml \
  --host 0.0.0.0 --port 6008
```

4. Open `http://127.0.0.1:6008/`

Features: 3-camera sync clock, virtual camera wheel (赛3–赛8 placeholders), quiet 4+4 main-camera calib → 2D pitch, foul assist tip at ~4.5s with detection-box red flash.

## 2) Offside review (6006)

1. Prepare config + discovery/review cache (from a prior discover run, or copy existing `frame_index.jsonl`):

```bash
cp configs/offside_demo.example.yaml configs/offside_demo.yaml
# set video:
```

2. Start:

```bash
python main.py demo-offside review \
  --input demo_data/offside/offside.mp4 \
  --discovery-dir demo_outputs/offside/discovery \
  --config configs/offside_demo.yaml \
  --host 0.0.0.0 --port 6006 --device cuda
```

3. Open `http://127.0.0.1:6006/`

Features: frame scrub, quiet 4+4 calib, 2D pitch, role confirm, second-last defender offside line + OFFSIDE overlay.

## Notes for reviewers

- Showcase UIs are **preprocess / cache driven** for the dashboard (no live YOLO while scrubbing).
- Offside formal judge still requires human confirmation fields; review UI provides geometric assist.
- Do not commit secrets, raw AutoDL absolute paths, or large `demo_outputs/` videos into git.
