# Architecture

This document describes the real-time soccer analysis pipeline after the
Web dashboard migration. It is intended for engineers who need to extend
or maintain the system.

## Goals

- **Real-time, low-latency**: process the latest frame, drop older ones
  on backpressure, never let the display or the browser control the
  inference loop.
- **Modular**: a single, large generator function has been replaced by
  independent components that can be unit-tested in isolation.
- **Bounded resources**: events, log lines, and frame buffers are all
  bounded to keep memory usage flat in long-running deployments.
- **No Base64 / no in-browser YOLO**: video frames are streamed over
  the dedicated MJPEG endpoint or future WebRTC; WebSocket only carries
  structured JSON.
- **RTSP-aware**: the same pipeline can be pointed at a local file or
  a wired IP camera; auto-reconnect is built in.

## Components

### `app/state/`

| File | Purpose |
|------|---------|
| `models.py`  | Pydantic models for `FrameState`, `GameEvent`, `PlayerState`, `MetricsSnapshot`, `PipelineConfig`, `PipelineCommand` and the `HomographyStatus` / `PlayerRole` / `SourceStatus` enums. These are the wire format between the backend and the dashboard; the TypeScript declarations in `web/src/types/messages.ts` mirror them. |
| `store.py`   | Thread-safe `StateStore`. Holds the latest frame state, the current config, the bounded event log (max 500), the bounded log buffer (max 200 lines), the latest metrics snapshot, and the source/pipeline flags. |
| `events.py`  | A small synchronous `EventBus` for component-internal signalling. The dashboard does not subscribe to it directly. |

### `app/pipeline/`

| File | Purpose |
|------|---------|
| `source.py`  | `VideoSource` abstract base + `LocalFileSource` + `RTSPSource`. The RTSP source attempts a GStreamer pipeline first (`rtspsrc latency=0`), then falls back to OpenCV/FFmpeg. The constructor is non-blocking; the first `read()` triggers the actual capture. On read failure it transparently reconnects up to 10 times. |
| `buffer.py`  | `BoundedFrameBuffer` with two modes: `REALTIME` (drop oldest, never block) and `OFFLINE` (block producer when full). `PipelineMode` is passed in by the caller. |
| `engine.py`  | `InferencePipeline` is the only place that runs the heavy models. It owns one `PitchProjectionEngine`, one `ByteTrack` tracker, one `OnlineTeamClassifier`, and (optionally) one `FoulDetector`. The internal thread runs `with torch.inference_mode():` for every frame and emits structured state into the `StateStore`. |
| `recorder.py`| `VideoRecorder` (kept for the offline modes; not used by the server in the default config). |

### `app/classification/online.py`

Wraps `TeamClassifier` (SigLIP + UMAP + KMeans) and adds an **incremental
warm-up** mode:

1. The first `warmup_frames` (default 120) are sampled at `warmup_stride`
   (default 15) to collect player crops.
2. A background thread fits the UMAP reducer and KMeans cluster.
3. Before the fit completes, `predict()` returns `-1` for every player.
4. When the fit completes, predictions switch over without a restart.
5. A pickled `(reducer, cluster_model)` pair can be saved and loaded.

This replaces the legacy two-pass scan in
`app/modes/team_classification.py:42-46` and `app/modes/radar.py:176-188`,
which scanned the entire video before the first inference frame.

### `app/services/publisher.py`

`WebSocketPublisher` is a thin async wrapper around a set of
`fastapi.WebSocket` clients. `push_loop` is started per connection and
emits a `FrameState` every ~33 ms and a `MetricsSnapshot` every 1 s.
`handle_command_text` applies `start`, `stop`, and `update_config`
commands.

### `app/server/`

The FastAPI app is built in `app/server/main.py`. It registers
the four REST routers, the WebSocket router, and the `GET /video/stream`
MJPEG generator. The MJPEG encoder re-uses the latest raw frame held in
`StateStore._latest_raw_frame` (set by the pipeline once per processed
frame), so it does not perform any extra decoding.

## Real-time vs offline

- **Real-time mode** is the default. The buffer drops the oldest frame
  when full so the display never lags behind the camera. The team
  classifier warm-up runs concurrently in a background thread.
- **Offline mode** blocks the producer when the buffer is full. Used
  when the caller needs every frame processed in order (e.g. training
  set generation).

## Timestamps

Every frame carries:

- `capture_timestamp_ms` — wall-clock time when the source handed the
  frame to the pipeline (`time.time() * 1000` at the top of
  `_run_loop`).
- `processed_timestamp_ms` — wall-clock time when the pipeline
  finished processing the frame.
- The `InferencePipeline` also tracks `inference_latency_ms` (the
  interval between model forward and result) and the running
  `end_to_end_latency_ms` (capture → processed) for the metrics
  snapshot.

These are **software** timestamps only. For multi-camera PTP
synchronization, see the *Future Work* section.

## Communication

The dashboard never receives video over WebSocket and never performs
any inference in the browser. Two independent channels:

| Channel | Direction | Payload |
|---------|-----------|---------|
| WebSocket `/ws/state` | server → client | `FrameState`, `MetricsSnapshot` |
| WebSocket `/ws/state` | client → server | `{command, params}` (start, stop, update_config) |
| HTTP `GET /video/stream` | server → client | MJPEG (multipart/x-mixed-replace) |
| HTTP REST `/api/*` | both | JSON |

The next iteration will replace MJPEG with WebRTC. The MJPEG endpoint
is intentionally narrow and side-effect free, so swapping it does not
require any pipeline change.

## Tests

62 tests:

- 17 pre-existing tests for pitch config, projection, ball tracking,
  radar dashboard, runtime helpers, and the view transformer.
- 8 Pydantic model tests (serialization roundtrips, null field
  coordinates, status enums).
- 6 BoundedFrameBuffer tests (realtime drop, offline block, close
  unblocks waiters, latest-only read).
- 12 StateStore + EventBus tests (thread safety, bounded buffers,
  subscriber exception isolation, snapshot API).
- 8 video source tests (local read, factory routing, RTSP
  non-blocking constructor, status reporting).
- 7 FastAPI REST tests (health, status, config GET/PUT, events,
  pipeline start/stop, video stream endpoint).
- 4 WebSocket tests (connect, frame_state push, start command,
  update_config command).
- 3 smoke-integration tests (synthetic video, state pipeline,
  buffer pipeline).

Run with:

```bash
.venv/bin/python -m pytest tests/ -v
```

## Future work

- **WebRTC signalling** — replace the MJPEG endpoint with WebRTC,
  reserving `/ws/webrtc` for SDP/ICE. The pipeline does not need to
  change.
- **PTP / hardware timestamping** — add a `Clock` interface in
  `app/pipeline/source.py` and let the source override
  `capture_timestamp_ms()`. Multi-camera synchronization will go here.
- **ONNX / TensorRT** — wrap the Ultralytics `YOLO` calls in a
  `DetectorBackend` interface so FP16 and TensorRT engines can be
  selected per deployment.
- **GStreamer for offline** — the `LocalFileSource` could also
  support GStreamer pipelines for hardware-accelerated decoding on
  Jetson or other edge devices.
