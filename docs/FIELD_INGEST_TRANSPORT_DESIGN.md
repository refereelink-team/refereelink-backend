# RefereeLink Field Ingest and Replay Design

Status: receiver boundary implemented on `feat/backend-field-ingest-receiver`. This document describes the versioned contract and the current receiver-only implementation. Production deployment, replay workers, inference integration, and training behavior remain out of scope for this phase.

## Scope

The iOS field app produces two compatible products:

1. A self-contained `.rlcapture` session for offline replay, validation, and training preparation.
2. A live session containing H.264/MPEG-TS video over SRT and versioned telemetry over WSS. The phone keeps the local recording even when the live connection is unavailable.

The backend must preserve the distinction between:

- phone capture time and backend receive time;
- camera pose and DockKit diagnostics;
- capture drops, transport drops, decode drops, association gaps, and inference drops;
- a complete offline artifact and a live stream that is only partially received.

WatchConnectivity, active gimbal control, audio, vendor SDKs, automatic training, and pseudo-label generation are out of scope.

## Protocol versions and identifiers

All new field-ingest payloads use `schema_version: "1.0"`.

| Field | Rule |
| --- | --- |
| `device_id` | Persisted application UUID. Never use an advertising identifier. |
| `session_id` | Phone-generated UUID. It is stable across offline export and live reconnect. |
| `frame_id` | Monotonic within a capture session; never reset on network reconnect. |
| `stream_epoch` | Incremented for every live receiver allocation or encoded-format change. |
| `t_us` | Session-relative monotonic time in integer microseconds. |
| `transport_pts90k` | MPEG-TS presentation time in the 90 kHz clock for live correlation. |
| `capture_unix_us` | Wall-clock observation for diagnostics only, not the primary join key. |

The backend must reject an unknown major schema version and may accept a newer minor version only when explicitly configured. Unknown optional fields are ignored and retained in raw artifact metadata where possible.

## Public interface

The following endpoints are implemented under the field-ingest service. They are separate from the existing `/ws/state` and `/video/stream` interfaces.

| Method | Endpoint | Semantics |
| --- | --- | --- |
| `GET` | `/api/v1/field/capabilities` | Returns supported schema versions, video profiles, upload limits, and receiver availability. |
| `POST` | `/api/v1/field/sessions` | Idempotently registers a phone-created session. The idempotency key is `session_id`. |
| `GET` | `/api/v1/field/sessions/{session_id}` | Returns receiver-only session, telemetry, SRT, decode, and artifact diagnostics. |
| `GET`/`WSS` | `/ws/v1/field/sessions/{session_id}` | Authenticated telemetry, frame index, gap reports, and four-timestamp clock probes. Never carries video or base64 image data. |
| `POST` | `/api/v1/field/sessions/{session_id}/live` | Allocates one live SRT receiver and returns an epoch, Tailnet-only endpoint, and short-lived stream token. Return `409` when the inference source is busy. |
| `DELETE` | `/api/v1/field/sessions/{session_id}/live/{epoch}` | Releases an allocation. It is idempotent. |
| `GET` | `/api/v1/field/sessions/{session_id}/artifacts` | Lists complete, partial, and missing uploaded objects. |
| `PUT` | `/api/v1/field/sessions/{session_id}/artifacts/{artifact_id}` | Idempotent upload of one immutable object. `Content-SHA256` and byte length are required. |
| `POST` | `/api/v1/field/sessions/{session_id}/complete` | Verifies the final manifest and marks the session complete or incomplete. |

Field routes require the configured bearer token. Successful contract responses include `schema_version` where applicable; error responses currently use FastAPI detail payloads. Status codes are `400` invalid contract, `401` invalid device token, `404` unknown session, `409` busy or conflicting immutable object, `411` missing upload headers, `413` object too large, `422` hash/schema validation failure, and `503` receiver/authentication unavailable.

### Session registration

`POST /api/v1/field/sessions` accepts the manifest identity and a compact manifest summary. It must be safe to retry with the same `session_id` and equivalent body. A different body for an existing ID returns `409`.

```json
{
  "schema_version": "1.0",
  "device_id": "15f6ce86-97aa-4e93-a4e7-d6d2a6ef8324",
  "session_id": "e2d2a4ea-c18a-42ed-b0ae-88f2fdcc7e50",
  "mode": "realtime",
  "video_profile": "720p30-h264",
  "started_at": "2026-09-18T12:00:00Z"
}
```

### WSS hello and telemetry

The client sends `hello` immediately after the WebSocket is established. The server replies with `hello_ack`, its accepted schema version, receive limits, and a server sequence number.

Telemetry messages are envelopes with a client sequence. The server acknowledges the highest contiguous sequence and reports gaps; it does not infer missing frames from arrival order.

```json
{
  "type": "telemetry_batch",
  "schema_version": "1.0",
  "session_id": "e2d2a4ea-c18a-42ed-b0ae-88f2fdcc7e50",
  "stream_epoch": 3,
  "client_sequence": 1842,
  "items": [
    {
      "type": "frame",
      "frame_id": 1841,
      "t_us": 61366667,
      "transport_pts90k": 5523000,
      "camera_motion_sample_id": 3681,
      "camera_motion_age_us": 12000,
      "pose_missing_reason": null
    },
    {
      "type": "camera_motion",
      "sample_id": 3681,
      "t_us": 61354667,
      "source_timestamp": 9271.442,
      "pitch": 0.01,
      "yaw": -0.02,
      "roll": 0.03,
      "rotation_rate_xyz": [0.0, 0.01, -0.02],
      "reference_frame": "xArbitraryZVertical",
      "status": "streaming"
    }
  ]
}
```

The full offline `CameraMotionSample`, `CapturedFrameMetadata`, `CameraConfigurationSnapshot`, and `DockDiagnosticEvent` fields remain authoritative. The live representation may be batched, but it must not change units or coordinate semantics.

### Clock probe

During connection, send eight probes, then one every 30 seconds. Use the standard four timestamps (`t1` client send, `t2` server receive, `t3` server send, `t4` client receive) to estimate offset and uncertainty. If the uncertainty is too high, report end-to-end delay as unknown instead of presenting a guessed value. Phone `t_us` and MPEG-TS PTS remain the join keys.

## Video receiver

The live video allocation returns an SRT caller URL that is reachable only over the Tailnet, a short-lived token, and the selected profile. The receiver should use:

- SRT live/TSBPD mode;
- caller mode from the phone;
- late-packet drop and approximately 200 ms latency as the starting point;
- MPEG-TS payload size 1128 bytes;
- H.264 with an IDR about every second and no B-frames;
- an explicit `stream_epoch` in the SRT stream ID and the WSS association.
- the short-lived epoch token as the SRT passphrase, so an allocation token is
  rejected during the SRT handshake rather than only being logged or matched
  after the stream starts.

The SRT receiver stores the original MPEG-TS PTS and emits decoded frames with `session_id`, `stream_epoch`, `transport_pts90k`, and the decoder receive time. It must never use `datetime.now()` taken before `read()` as the phone capture time.

Linux should probe NVDEC first and fall back to software decode. macOS remains a development-compatible path. The receiver must accept a fresh IDR after a congestion reset or epoch change and must not attempt to decode an arbitrary inter-frame packet as a new stream.

## Frame association and existing pipeline integration

Introduce a complete internal `CapturedFrame` value:

```text
CapturedFrame {
  image/frame reference,
  session_id, stream_epoch, frame_id,
  t_us, transport_pts90k, capture_unix_us,
  camera_motion_sample, camera_configuration,
  backend_received_at
}
```

`FrameJoiner` associates a decoded video frame with the newest camera-motion sample at or before the frame, waiting at most 50 ms. No matching sample is a valid result and must carry `pose_missing_reason`; the video still proceeds. DockKit motion data is optional diagnostics and is never substituted for Core Motion.

The existing source interfaces remain compatible. The new source is an additive field-ingest adapter, not a replacement for `/ws/state` or `/video/stream`. The receiver-only phase keeps `FrameState` and the dashboard contract unchanged; future inference integration must synchronize optional source/session fields across:

- `app/state/models.py`;
- REST and WebSocket publishers;
- `web/src/types/`;
- protocol and integration tests.

Use the existing bounded buffer deliberately:

- live mode: drop oldest pending frames and keep the newest frame;
- offline replay: block the producer and process every frame in order;
- record separate counters for capture, local-recording, send, receive/decode, association, and inference drops.

Video files are stored on disk; SQLite stores session/artifact indexes and manifests, never large video BLOBs.

## Offline replay and artifact ingestion

The iOS package layout is:

```text
<session_id>.rlcapture/
  manifest.json
  video/init.mp4
  video/000001.m4s ...
  metadata/frames-*.ndjson
  metadata/motion-*.ndjson
  metadata/camera-*.ndjson
  metadata/dock-*.ndjson
  metadata/events-*.ndjson
```

The backend importer accepts either the directory or a ZIP with the same relative paths. It validates the manifest before processing media, verifies every declared size and SHA-256, and records `recording`, `stopped`, or `interrupted` lifecycle without repairing missing data silently.

Upload objects are immutable and at most 8 MiB. Larger files are split into fixed-size parts with a manifest relationship. Repeating an object with the same hash is success; a hash conflict is `409`. The final completion call checks that all required parts and metadata exist.

`CaptureReplaySource` reads original session `t_us` values and can play as fast as possible for offline tests. Playback acceleration must not replace the original session clock with wall time. The worker must support deterministic ordering and explicit missing-pose markers.

## Deployment and security

- Run the HTTPS/WSS ingress through Tailscale Serve; the SRT UDP receiver listens only on the Tailnet interface.
- Configure Tailscale ACL/grants for the field device identity and receiver port. Do not expose SRT to the public Internet.
- Use system TLS validation for HTTPS/WSS and a Keychain-backed device token on iOS. SRT uses a short-lived allocation token; rotate it on every epoch.
- Redact bearer tokens, SRT stream IDs, raw motion payloads, and exact device identifiers from ordinary logs. Use a request/session correlation ID for diagnostics.
- Keep receiver/artifact storage quotas visible. A low-disk condition is a first-class error, not an automatic deletion trigger.
- Restart recovery rebuilds the SQLite artifact index from manifests and hashes; partial objects remain marked partial until verified.

## Implementation status and remaining work

The current branch implements the first four receiver-boundary steps:

1. Pydantic wire models, compatibility aliases, and contract tests.
2. SQLite session/artifact persistence and idempotent APIs.
3. WSS telemetry, ACK/gap accounting, clock probes, and bounded NDJSON output.
4. FFmpeg/libsrt SRT receive, PTS-preserving output, decode metrics, and receiver-only startup.

Remaining work is `CaptureReplaySource`, ZIP/directory import, production quotas and auth rotation, restart recovery, and inference integration. Field acceptance uses the iOS transport branch and must be recorded separately from production backend validation.

No step should automatically start training or generate labels.

## Acceptance checklist

- A complete offline package independently decodes and validates against its manifest.
- Missing pose, DockKit-without-motion, and interrupted recording are visible in the API and replay output.
- A live session receives a fresh IDR after reconnect and never associates frames by arrival order.
- A 10-second network outage produces local iOS recording continuity and a live gap report.
- `FrameState` and Web TypeScript contracts remain backwards compatible for existing dashboards.
- Metrics distinguish transport, decode, join, and inference delay.
- Tailnet direct and DERP paths are measured separately; no latency guarantee is inferred from topology alone.
- Production backend integration is not considered complete until the real receiver, storage, and field acceptance run are deployed.
