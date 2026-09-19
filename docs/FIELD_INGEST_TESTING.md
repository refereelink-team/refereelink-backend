# Field Ingest Testing

This document records the reproducible receiver-boundary checks. It does not
replace full backend, inference, GPU, or production deployment validation.

## Automated checks

Run from the backend repository with the project-managed `uv` environment:

```bash
UV_CACHE_DIR=/private/tmp/refereelink-uv-cache uv run --offline pytest tests/test_field_ingest.py -q
UV_CACHE_DIR=/private/tmp/refereelink-uv-cache uv run --offline pytest tests/test_api.py tests/test_websocket.py -q
```

The field tests use an injected fake receiver for deterministic lifecycle and
protocol tests. They do not claim that real SRT packets were received.

## Receiver-only local run

The real local acceptance requires an FFmpeg binary that advertises the `srt`
protocol, plus the Mac's Tailscale address:

```bash
ffmpeg -hide_banner -protocols | grep -x srt

REFEREELINK_FIELD_TOKEN='use-a-local-token' \
REFEREELINK_SRT_BIND_HOST='<mac-tailnet-ip>' \
REFEREELINK_SRT_ADVERTISE_HOST='<mac-tailnet-ip>' \
REFEREELINK_FFMPEG_BIN='/path/to/libsrt-enabled/ffmpeg' \
REFEREELINK_FIELD_INGEST_ROOT='/tmp/refereelink-field-ingest' \
UV_CACHE_DIR=/private/tmp/refereelink-uv-cache \
uv run --offline python -m app.field_ingest.server --host 0.0.0.0 --port 8000
```

Configure the iOS field endpoint with the HTTPS Tailscale Serve URL (recommended
for iOS ATS/TLS validation) and the same bearer token. A direct Tailnet HTTP
URL is suitable only when the app's transport-security policy explicitly allows
it. The receiver-only status endpoint is:

```text
GET /api/v1/field/sessions/{session_id}
```

When SRT is available, acceptance requires a registered session, `hello_ack`,
increasing frame and Core Motion telemetry counts, an allocated SRT epoch, a
running receiver, and decoded frame count greater than zero. The status must
also expose the associated stream epoch and PTS diagnostics. No inference
result is required.

## Latest physical acceptance

On 2026-09-19, the receiver-only service passed a real iPhone-to-Mac Tailnet
WSS/SRT check after the iOS stream-epoch PTS fix. Network addresses, session
identifiers and temporary credentials are intentionally omitted from this
tracked document; the complete status response remains in the local test
workspace.

- The capability check reported `receiver_available=true`,
  `transport.srt=true`, and `srt_reason=null`.
- WSS telemetry had increasing frame/Core Motion counts with `gap_count=0`.
- SRT epoch 1 was listening and decoded 3,845 frames with no receiver error;
  MPEG-TS `time_base=1/90000`, duration approximately 128.24 seconds.
- The first decoded PTS and first iOS `transport_pts90k` were both `0`; the
  iOS telemetry PTS sequence was monotonic across 3,876 checked samples.
- The iOS screen showed “已连接” for realtime parameters.

This proves the iOS-to-local receiver communication and video/telemetry
association path. It does not claim production backend or inference
validation.

WSS can be verified independently with the iOS test launch argument
`--field-wss-only`. In that mode the app still registers the session and sends
the real frame/Core Motion telemetry batch, but intentionally skips SRT
allocation. The receiver accepts this as telemetry-only epoch `0`; it is a
communication diagnostic and must not be reported as video-receive success.

The following earlier run is retained as historical evidence:

```text
iPhone on the Tailnet
  -> internal HTTPS/WSS Tailscale Serve address
  -> local receiver on port 8000
```

The first non-isolated run observed one telemetry batch containing one frame
and four Core Motion samples. Its SRT allocation returned 503 because the
default local FFmpeg did not advertise `srt`; the latest run resolved this
environment issue with a libsrt-enabled FFmpeg binary.

Do not commit bearer tokens, full logs, large video captures, or temporary
screenshots. Keep only a stable result summary and a small number of
diagnostic images when they explain a durable behavior.
