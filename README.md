# Soccer Analysis Web Dashboard

Real-time computer-vision dashboard for football (soccer) matches. The
Python pipeline runs YOLO player/ball/pitch detection, ByteTrack tracking,
team classification, and homography-based pitch projection, and pushes
structured state to a React web dashboard over WebSocket. Live video is
delivered as a low-latency MJPEG stream from a dedicated FastAPI endpoint.

> The original PySide6/QML dashboard is preserved under `--mode
> RADAR_DASHBOARD_LEGACY`. The web dashboard is the recommended and only
> actively maintained front-end.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    VideoSource Layer                         │
│  LocalFileSource │ RTSPSource (GStreamer + FFmpeg fallback)  │
│  - non-blocking constructor  - auto-reconnect on drop        │
│  - status reporting via StateStore                            │
└──────────────────────┬───────────────────────────────────────┘
                       │ BGR frame
                       ▼
┌──────────────────────────────────────────────────────────────┐
│               BoundedFrameBuffer                            │
│  - realtime mode: drop oldest, deliver latest                │
│  - offline mode: block producer when full                    │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              InferencePipeline                                │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │ PitchDetect │  │ PlayerDetect  │  │ FoulDetect       │    │
│  │ (YOLO)      │  │ (YOLO)        │  │ (MVFoul rolling) │    │
│  └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘    │
│         ▼                ▼                    │               │
│  ┌──────────────┐  ┌──────────────┐           │               │
│  │ Projection   │  │ ByteTrack    │           │               │
│  │ Engine       │  │ + OnlineTeam │           │               │
│  │              │  │ Classifier   │           │               │
│  └──────┬───────┘  └──────┬───────┘           │               │
│         └────────────────┼────────────────────┘               │
│                          ▼                                   │
│                   FrameState (Pydantic)                      │
└──────────────────────┬───────────────────────────────────────┘
                       │ structured JSON (no image data)
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                     StateStore                                │
│  thread-safe shared state: latest_frame_state, metrics,       │
│  events, config, log buffer (max 200 lines), source_status    │
└──────────┬────────────────────────────┬──────────────────────┘
           │                            │
           ▼                            ▼
┌──────────────────┐    ┌──────────────────────────┐
│ FastAPI REST     │    │ WebSocket (/ws/state)     │
│ /health          │    │  - pushes FrameState      │
│ /api/status      │    │  - pushes Metrics (1Hz)   │
│ /api/config      │    │  - receives start/stop    │
│ /api/events      │    │    and update_config      │
│ /api/pipeline/*  │    └─────────────┬────────────┘
└────────┬─────────┘                  │
         │                            │
         ▼                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    Web Dashboard (React)                     │
│  VideoPanel (MJPEG) │ Pitch2D │ StatusCards │ EventAlerts    │
│  ControlPanel       │ LogPanel                              │
└──────────────────────────────────────────────────────────────┘
```

## Communication

- **WebSocket `/ws/state`** — push structured `FrameState` and
  `MetricsSnapshot` Pydantic models; receive control commands
  (`start`, `stop`, `update_config`).
- **HTTP `GET /video/stream`** — low-latency MJPEG stream
  (`multipart/x-mixed-replace`). Not used for control data.
- **HTTP REST** — configuration, event history, health.

Base64 video over WebSocket is **never** used. Frame data only travels
over the dedicated MJPEG endpoint or future WebRTC signalling.

## Layout

```
app/
├── state/        Pydantic models, StateStore, EventBus
├── pipeline/     VideoSource, BoundedFrameBuffer, InferencePipeline, recorder
├── services/     WebSocketPublisher
├── server/       FastAPI app (REST + WebSocket + MJPEG)
├── modes/        Legacy single-purpose modes
└── runtime.py    Pitch + radar drawing helpers

web/              React + TypeScript + Vite dashboard
├── src/
│   ├── components/  VideoPanel, Pitch2D, StatusCards, EventAlerts,
│   │                ControlPanel, LogPanel
│   ├── hooks/       useWebSocket
│   ├── store/       Zustand state container
│   ├── types/       TypeScript mirrors of Pydantic models
│   └── styles/      Dark control-console theme
├── vite.config.ts   Proxies /api, /ws, /video to FastAPI in dev
├── package.json
└── tsconfig.json

tests/            62 tests (45 new + 17 existing)
```

## Install

```bash
# 1. Create a Python virtualenv with uv
uv venv .venv
source .venv/bin/activate

# 2. Install the project
uv pip install -e ".[tests]"

# 3. Install JavaScript dependencies for the dashboard
cd web && npm install && cd ..
```

## Run

```bash
# Download model weights and sample videos (optional)
./tools/setup_assets.sh

# Start the FastAPI backend (loads YOLO + SigLIP for team classification)
.venv/bin/python -m app.server.main \
    --video_source assets/data/smoke_test.mp4 \
    --device cpu

# In another terminal: launch the React dev server
cd web && npm run dev
# Open http://localhost:5173
```

For RTSP sources:

```bash
.venv/bin/python -m app.server.main \
    --video_source rtsp://192.168.1.100:554/stream \
    --device cuda
```

The frontend dev server proxies `/api`, `/ws`, and `/video` to the
backend on port 8000.

For a production build:

```bash
cd web && npm run build
# The compiled bundle lands in web/dist
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET    | `/health`               | Liveness probe |
| GET    | `/api/status`           | Pipeline, source, metrics snapshot |
| GET    | `/api/config`           | Current pipeline config |
| PUT    | `/api/config`           | Update config (e.g. enable foul detection) |
| GET    | `/api/events`           | Recent events (`?limit=50&offset=0`) |
| POST   | `/api/pipeline/start`   | Start inference |
| POST   | `/api/pipeline/stop`    | Stop inference |
| GET    | `/video/stream`         | MJPEG live video |
| WS     | `/ws/state`             | Structured state push + commands |

## Modes

| Mode | Description |
|------|-------------|
| `SERVER`                  | New web dashboard mode (recommended) |
| `RADAR_DASHBOARD_LEGACY`  | PySide6 + QML fallback |
| `RADAR`                   | Offline radar processing, writes annotated video |
| `PLAYER_DETECTION`        | YOLO player detection only |
| `PITCH_DETECTION`         | YOLO pitch keypoint detection only |
| `BALL_DETECTION`          | YOLO ball detection with slicer |
| `PLAYER_TRACKING`         | YOLO + ByteTrack |
| `TEAM_CLASSIFICATION`     | YOLO + ByteTrack + offline team clustering |
| `FOUL_DETECTION`          | MVFoul rolling window with HUD overlay |

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

62 tests cover Pydantic models, the bounded buffer (realtime drop
+ offline block), the video source abstraction (factory routing,
reconnect state), the StateStore and EventBus (thread safety,
bounded buffers, exception isolation), the FastAPI REST endpoints,
the WebSocket lifecycle and commands, the legacy pitch/ball/team
core, and a synthetic-video smoke test for the full pipeline.
