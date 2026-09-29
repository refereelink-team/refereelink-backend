# Field Ingest to Inference Bridge

This document covers the live backend path only:

```text
iPhone SRT/WSS
  -> FieldIngestService
  -> FFmpeg rawvideo decoder
  -> FrameJoiner
  -> FieldIngestSource
  -> InferencePipeline
  -> StateStore
  -> /ws/state and /video/stream
```

Offline `.rlcapture` replay and production deployment are separate work.

## Local prerequisites

Use a FFmpeg build with libsrt enabled. On the development Mac, the Homebrew
full build is available at:

```text
/opt/homebrew/Cellar/ffmpeg-full/9.0.2/bin/ffmpeg
/opt/homebrew/Cellar/ffmpeg-full/9.0.2/bin/ffprobe
```

Verify the capability before starting a live session:

```bash
FFMPEG_BIN='/opt/homebrew/Cellar/ffmpeg-full/9.0.2/bin/ffmpeg'
"$FFMPEG_BIN" -hide_banner -protocols 2>&1 | grep -E '^[[:space:]]+srt$'
"$FFMPEG_BIN" -hide_banner -buildconf 2>&1 | grep -- '--enable-libsrt'
```

The default `/opt/homebrew/bin/ffmpeg` may be a different build without SRT.
If the check fails, the receiver must report `receiver_available=false`; this
is an environment limitation, not evidence that the iOS transport is broken.

## Automated validation

The Python environment is managed with `uv`. When the project itself cannot be
resolved offline because build dependencies are not cached, reuse the already
synced project interpreter without resolving a new environment:

```bash
UV_CACHE_DIR=/private/tmp/refereelink-uv-cache \
uv run --no-project --python .venv/bin/python \
pytest tests/test_field_ingest_bridge.py tests/test_field_ingest.py -q

UV_CACHE_DIR=/private/tmp/refereelink-uv-cache \
uv run --no-project --python .venv/bin/python \
pytest tests/ -q

.venv/bin/ruff check app tests
```

The bridge tests cover rawvideo frame delivery, PTS matching, delayed WSS
metadata, missing pose markers, latest-frame dropping, source release, and the
explicit field pipeline start payload.

The local libsrt loopback was also verified with the full FFmpeg build listed
above: a synthetic 64x36, 5 fps MPEG-TS sender completed successfully, the
receiver decoded and delivered 10 raw frames, PTS advanced from 0 to 162000,
and no receiver error was reported. This validates the SRT listener and raw
frame bridge independently of iPhone capture and inference weights.

## Live bridge sequence

1. Start the normal FastAPI application with the libsrt-enabled FFmpeg path.
2. Register the phone-generated session and allocate a live epoch through the
   existing Field Ingest API.
3. Let iOS establish WSS and SRT. Confirm the field session status shows
   increasing telemetry and decoded-frame counters.
4. Start inference explicitly; receiving an SRT epoch never starts GPU
   inference automatically:

```json
{
  "field_session_id": "<session-id>",
  "field_stream_epoch": 1,
  "device": "cuda"
}
```

5. Read `/api/status`, `/ws/state`, and `/video/stream`. `FrameState` includes
   optional `capture_source` metadata while the image itself remains on the
   MJPEG path.
6. Stop the pipeline. The bound source releases the epoch, closes the raw
   frame queue, and terminates FFmpeg. A reconnect must allocate a new epoch
   and explicitly bind a new pipeline; frames are never mixed across epochs.

## Acceptance evidence

The bridge is considered transport/inference-ready when all of the following
are observable:

- `decoded_frame_count` and `raw_frame_count` increase together;
- WSS frame and Core Motion telemetry continue increasing;
- `transport_pts90k` joins the correct source frame ID without arrival-order
  assumptions;
- `capture_source.session_id`, `stream_epoch`, and source frame ID appear in
  `/ws/state` after inference processes a field frame;
- `decode_dropped_frames` and `join_missing_frames` remain bounded and visible;
- `/video/stream` produces the annotated inference output;
- stopping or ending an epoch releases FFmpeg, reader threads, and queues;
- real YOLO accuracy, team calibration, and event quality are reported
  separately from this bridge acceptance.

Do not commit bearer tokens, full logs, raw video captures, or device-specific
addresses. Keep only stable commands and summarized results in Git.
