from __future__ import annotations

import hashlib
import json
import logging
import secrets
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

from app.field_ingest.config import FieldIngestSettings
from app.field_ingest.frames import (
    CapturedFrame,
    DecodedVideoFrame,
    FrameJoiner,
    LatestFrameQueue,
)
from app.field_ingest.models import (
    CapabilityResponse,
    FieldSessionRegistration,
    LiveAllocationRequest,
    LiveAllocationResponse,
    SCHEMA_VERSION,
    TelemetryBatch,
    utc_now,
)
from app.field_ingest.receiver import SRTReceiver, probe_srt_support
from app.field_ingest.store import FieldSessionStore

logger = logging.getLogger(__name__)

VIDEO_PROFILES: dict[str, dict[str, int]] = {
    "720p30": {"width": 1280, "height": 720, "fps": 30},
    "540p24": {"width": 960, "height": 540, "fps": 24},
    "360p15": {"width": 640, "height": 360, "fps": 15},
}


@dataclass
class EpochRuntime:
    session_id: str
    stream_epoch: int
    telemetry_path: Path
    joined_path: Path
    telemetry_file: Any
    receiver: SRTReceiver | None = None
    last_client_sequence: int = 0
    gap_count: int = 0
    missing_sequences: list[int] = field(default_factory=list)
    duplicate_count: int = 0
    frame_count: int = 0
    motion_count: int = 0
    dock_count: int = 0
    received_message_count: int = 0
    last_received_at: str | None = None
    last_frame_t_us: int | None = None
    motion_samples: deque[tuple[int, dict[str, Any]]] = field(
        default_factory=lambda: deque(maxlen=512)
    )
    join_file: Any = None
    frame_joiner: FrameJoiner | None = None
    frame_queue: LatestFrameQueue = field(default_factory=LatestFrameQueue)
    consumer_attached: bool = False
    decoded_frame_count: int = 0
    joined_frame_count: int = 0
    join_missing_count: int = 0

    def close(self) -> None:
        self.frame_queue.close()
        for handle in (self.telemetry_file, self.join_file):
            try:
                handle.close()
            except Exception:
                pass


class FieldIngestService:
    def __init__(
        self,
        settings: FieldIngestSettings | None = None,
        *,
        store: FieldSessionStore | None = None,
        ffmpeg_probe: Callable[[str], tuple[bool, str]] = probe_srt_support,
        receiver_factory: Callable[..., SRTReceiver] = SRTReceiver,
    ) -> None:
        self.settings = settings or FieldIngestSettings.from_env()
        self.store = store or FieldSessionStore(self.settings.root)
        self._probe = ffmpeg_probe
        self._receiver_factory = receiver_factory
        self._lock = threading.RLock()
        self._epochs: dict[tuple[str, int], EpochRuntime] = {}

    def authorize(self, token: str | None) -> bool:
        configured = self.settings.auth_token
        return bool(configured and token and secrets.compare_digest(configured, token))

    def capabilities(self) -> CapabilityResponse:
        supported, reason = self._probe(self.settings.ffmpeg_bin)
        transport: dict[str, Any] = {
            "wss": True,
            "srt": supported,
            "srt_reason": reason or None,
            "tailnet_only": True,
        }
        return CapabilityResponse(
            schema_version=SCHEMA_VERSION,
            supported_schema_versions=[SCHEMA_VERSION],
            video_profiles=[
                {"name": "720p30", "width": 1280, "height": 720, "fps": 30, "bitrate": 1_500_000},
                {"name": "540p24", "width": 960, "height": 540, "fps": 24, "bitrate": 1_000_000},
                {"name": "360p15", "width": 640, "height": 360, "fps": 15, "bitrate": 500_000},
            ],
            receiver_available=supported
            and bool(self.settings.srt_bind_host and self.settings.srt_advertise_host),
            max_artifact_bytes=self.settings.max_artifact_bytes,
            upload_concurrency=2,
            transport=transport,
        )

    def register(self, registration: FieldSessionRegistration) -> tuple[str, dict[str, Any]]:
        if registration.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        result, data = self.store.register(registration)
        if result == "created":
            session_dir = self.settings.root / "sessions" / str(registration.session_id)
            (session_dir / "video").mkdir(parents=True, exist_ok=True)
            (session_dir / "metadata").mkdir(parents=True, exist_ok=True)
            (session_dir / "events").mkdir(parents=True, exist_ok=True)
            (session_dir / "manifest.json").write_text(
                json.dumps(registration.model_dump(mode="json", by_alias=True), indent=2) + "\n",
                encoding="utf-8",
            )
        return result, data

    def _runtime(self, session_id: str, epoch: int) -> EpochRuntime:
        runtime = self._epochs.get((session_id, epoch))
        if runtime is None:
            raise KeyError("live epoch not found")
        return runtime

    def _open_telemetry_runtime(self, session_id: str, epoch: int) -> EpochRuntime:
        session_root = self.settings.root / "sessions" / session_id
        metadata_root = session_root / "metadata"
        metadata_root.mkdir(parents=True, exist_ok=True)
        telemetry_path = metadata_root / f"telemetry-{epoch:06d}.ndjson"
        joined_path = metadata_root / f"joined-{epoch:06d}.ndjson"
        runtime = EpochRuntime(
            session_id=session_id,
            stream_epoch=epoch,
            telemetry_path=telemetry_path,
            joined_path=joined_path,
            telemetry_file=telemetry_path.open("a", encoding="utf-8"),
            join_file=joined_path.open("a", encoding="utf-8"),
            motion_samples=deque(maxlen=self.settings.max_joiner_samples),
        )
        runtime.frame_joiner = FrameJoiner(
            session_id,
            epoch,
            max_samples=self.settings.max_joiner_samples,
        )
        self._epochs[(session_id, epoch)] = runtime
        return runtime

    def allocate_live(
        self, session_id: UUID, request: LiveAllocationRequest
    ) -> LiveAllocationResponse:
        if self.store.get_session(session_id) is None:
            raise KeyError("session not found")
        profile = VIDEO_PROFILES.get(request.profile)
        if profile is None:
            raise ValueError(f"unsupported video profile: {request.profile}")
        capabilities = self.capabilities()
        if not capabilities.receiver_available:
            raise RuntimeError(
                "SRT receiver unavailable: configure a libsrt-enabled FFmpeg and Tailnet hosts"
            )
        with self._lock:
            if self.store.has_active_epoch():
                raise PermissionError("another live source is active")
            epoch = self.store.next_epoch(session_id)
            token = secrets.token_urlsafe(24)
            token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
            host = self.settings.srt_advertise_host
            bind_host = self.settings.srt_bind_host
            if not host or not bind_host:
                raise RuntimeError("SRT Tailnet bind and advertise hosts are required")
            self.store.create_epoch(
                session_id,
                epoch,
                bind_host,
                self.settings.srt_port,
                token_hash,
                request.profile,
                request.latency_ms,
            )
            epoch_dir = (
                self.settings.root / "sessions" / str(session_id) / "video" / f"epoch-{epoch:06d}"
            )
            epoch_dir.mkdir(parents=True, exist_ok=True)
            telemetry_path = (
                self.settings.root
                / "sessions"
                / str(session_id)
                / "metadata"
                / f"telemetry-{epoch:06d}.ndjson"
            )
            joined_path = (
                self.settings.root
                / "sessions"
                / str(session_id)
                / "metadata"
                / f"joined-{epoch:06d}.ndjson"
            )
            telemetry_file = telemetry_path.open("a", encoding="utf-8")
            join_file = joined_path.open("a", encoding="utf-8")
            runtime = EpochRuntime(
                session_id=str(session_id),
                stream_epoch=epoch,
                telemetry_path=telemetry_path,
                joined_path=joined_path,
                telemetry_file=telemetry_file,
                join_file=join_file,
                motion_samples=deque(maxlen=self.settings.max_joiner_samples),
            )
            runtime.frame_joiner = FrameJoiner(
                str(session_id),
                epoch,
                max_samples=self.settings.max_joiner_samples,
            )
            receiver = self._receiver_factory(
                ffmpeg_bin=self.settings.ffmpeg_bin,
                host=bind_host,
                port=self.settings.srt_port,
                session_id=str(session_id),
                stream_epoch=epoch,
                stream_token=token,
                output_path=epoch_dir / "capture.ts",
                latency_ms=request.latency_ms,
                width=profile["width"],
                height=profile["height"],
                on_frame=lambda pts: self._record_decoded_pts(str(session_id), epoch, pts),
                on_decoded_frame=lambda decoded: self._record_decoded_frame(
                    str(session_id), epoch, decoded
                ),
                on_end=lambda: self._record_receiver_end(str(session_id), epoch),
            )
            runtime.receiver = receiver
            self._epochs[(str(session_id), epoch)] = runtime
            try:
                receiver.start()
            except Exception:
                runtime.close()
                self._epochs.pop((str(session_id), epoch), None)
                self.store.release_epoch(session_id, epoch)
                raise
        return LiveAllocationResponse(
            session_id=session_id,
            stream_epoch=epoch,
            srt_host=host,
            srt_port=self.settings.srt_port,
            stream_token=token,
            latency_ms=request.latency_ms,
            profile=request.profile,
        )

    def _record_decoded_pts(self, session_id: str, epoch: int, pts90k: int) -> None:
        with self._lock:
            runtime = self._epochs.get((session_id, epoch))
            if runtime is None:
                return
            runtime.last_received_at = utc_now()

    def _record_decoded_frame(
        self, session_id: str, epoch: int, decoded: DecodedVideoFrame
    ) -> None:
        with self._lock:
            runtime = self._epochs.get((session_id, epoch))
            if runtime is None or runtime.frame_joiner is None:
                return
            runtime.last_received_at = decoded.received_at
            runtime.decoded_frame_count += 1
            ready = runtime.frame_joiner.ingest_decoded(decoded)
            self._enqueue_captured(runtime, ready)

    def _record_receiver_end(self, session_id: str, epoch: int) -> None:
        with self._lock:
            runtime = self._epochs.get((session_id, epoch))
            if runtime is None or runtime.frame_joiner is None:
                return
            self._enqueue_captured(runtime, runtime.frame_joiner.flush())
            runtime.frame_queue.close()

    def _enqueue_captured(
        self, runtime: EpochRuntime, frames: list[CapturedFrame]
    ) -> None:
        for frame in frames:
            runtime.joined_frame_count += 1
            if frame.pose_missing_reason is not None:
                runtime.join_missing_count += 1
            runtime.frame_queue.put(frame)

    def open_frame_source(self, session_id: UUID | str, epoch: int, store=None):
        from app.pipeline.source import FieldIngestSource

        key = (str(session_id), epoch)
        with self._lock:
            runtime = self._epochs.get(key)
            if runtime is None or runtime.receiver is None:
                raise KeyError("live epoch not found")
            if runtime.consumer_attached:
                raise PermissionError("live epoch already has an inference consumer")
            runtime.consumer_attached = True
            profile = self.store.get_epoch(session_id, epoch) or {}
            dimensions = VIDEO_PROFILES.get(profile.get("profile", "720p30"), VIDEO_PROFILES["720p30"])
            return FieldIngestSource(
                runtime.frame_queue,
                session_id=str(session_id),
                stream_epoch=epoch,
                fps=float(dimensions["fps"]),
                store=store,
                release_callback=lambda: self.release_frame_source(session_id, epoch),
                opened_callback=lambda: self._epoch_open(session_id, epoch),
                metrics_callback=lambda: self.frame_stream_status(session_id, epoch),
            )

    def _epoch_open(self, session_id: UUID | str, epoch: int) -> bool:
        with self._lock:
            runtime = self._epochs.get((str(session_id), epoch))
            if runtime is None or runtime.receiver is None:
                return False
            state = runtime.receiver.snapshot().get("state")
            return state in {"starting", "listening"}

    def release_frame_source(self, session_id: UUID | str, epoch: int) -> None:
        self.release_live(UUID(str(session_id)), epoch)

    def frame_stream_status(self, session_id: UUID | str, epoch: int) -> dict[str, Any]:
        with self._lock:
            runtime = self._epochs.get((str(session_id), epoch))
            if runtime is None:
                raise KeyError("live epoch not found")
            return {
                "session_id": str(session_id),
                "stream_epoch": epoch,
                "consumer_attached": runtime.consumer_attached,
                "decoded_frame_count": runtime.decoded_frame_count,
                "joined_frame_count": runtime.joined_frame_count,
                "join_missing_count": runtime.join_missing_count,
                "decode_dropped_frames": runtime.frame_queue.dropped,
                "queue_length": runtime.frame_queue.length,
                "queue_closed": runtime.frame_queue.closed,
            }

    def release_live(self, session_id: UUID, epoch: int) -> None:
        with self._lock:
            runtime = self._epochs.pop((str(session_id), epoch), None)
            if runtime is not None:
                runtime.consumer_attached = False
                if runtime.receiver is not None:
                    runtime.receiver.stop()
                runtime.close()
            self.store.release_epoch(session_id, epoch)

    def release_telemetry(self, session_id: UUID, epoch: int) -> None:
        with self._lock:
            runtime = self._epochs.get((str(session_id), epoch))
            if runtime is not None and runtime.receiver is None:
                self._epochs.pop((str(session_id), epoch), None)
                runtime.close()

    def handle_message(self, session_id: UUID, message: dict[str, Any]) -> dict[str, Any]:
        if message.get("type") == "hello":
            if str(message.get("session_id") or message.get("sessionId")) != str(session_id):
                raise ValueError("hello session_id does not match URL")
            return {
                "type": "hello_ack",
                "schema_version": SCHEMA_VERSION,
                "session_id": str(session_id),
                "server_sequence": 0,
                "max_batch_items": self.settings.max_telemetry_items,
                "clock_probe_interval_s": 30,
            }
        if message.get("type") == "clock_probe":
            return {
                "type": "clock_probe_ack",
                "schema_version": SCHEMA_VERSION,
                "client_sequence": message.get("client_sequence", message.get("clientSequence", 0)),
                "t1_client_send_us": message.get(
                    "t1_client_send_us", message.get("t1ClientSendUs")
                ),
                "t2_server_receive_us": _wall_clock_us(),
                "t3_server_send_us": _wall_clock_us(),
            }
        epoch = int(message.get("stream_epoch", message.get("streamEpoch", 0)))
        if message.get("type") == "telemetry_batch":
            batch = TelemetryBatch.model_validate(message)
            if batch.session_id != session_id:
                raise ValueError("telemetry session_id does not match URL")
            items = batch.items
            sequence = batch.client_sequence
        else:
            payload = message.get("payload", {})
            legacy_type = str(message.get("type", ""))
            item_type = {
                "frame_batch": "frame",
                "motion_batch": "camera_motion",
                "dock_state": "dock",
            }.get(legacy_type)
            if item_type is None:
                raise ValueError(f"unsupported websocket message type: {legacy_type}")
            items = [{"type": item_type, **payload}]
            sequence = message.get("client_sequence", message.get("clientSequence"))
            if sequence is None:
                sequence = 0
        return self.ingest_batch(session_id, epoch, int(sequence), items)

    def ingest_batch(
        self,
        session_id: UUID,
        epoch: int,
        sequence: int,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        with self._lock:
            runtime = self._epochs.get((str(session_id), epoch))
            if runtime is None and epoch == 0:
                runtime = self._open_telemetry_runtime(str(session_id), epoch)
            if runtime is None:
                raise KeyError("live epoch not found")
            if sequence and sequence <= runtime.last_client_sequence:
                runtime.duplicate_count += 1
            elif (
                sequence
                and runtime.last_client_sequence
                and sequence > runtime.last_client_sequence + 1
            ):
                missing_count = sequence - runtime.last_client_sequence - 1
                runtime.gap_count += missing_count
                remaining_capacity = max(0, 128 - len(runtime.missing_sequences))
                if remaining_capacity:
                    runtime.missing_sequences.extend(
                        range(
                            runtime.last_client_sequence + 1,
                            runtime.last_client_sequence
                            + 1
                            + min(missing_count, remaining_capacity),
                        )
                    )
            if sequence:
                runtime.last_client_sequence = max(runtime.last_client_sequence, sequence)
            runtime.received_message_count += 1
            runtime.last_received_at = utc_now()
            bounded_items = items[: self.settings.max_telemetry_items]
            ordered_items = sorted(
                bounded_items,
                key=lambda item: (
                    _int_value(item.get("t_us", item.get("tUs")))
                    if _int_value(item.get("t_us", item.get("tUs"))) is not None
                    else 2**63 - 1,
                    0 if item.get("type") in {"camera_motion", "motion"} else 1,
                ),
            )
            for item in ordered_items:
                normalized = {"received_at": runtime.last_received_at, **item}
                self._ingest_item(runtime, normalized)
            for item in bounded_items:
                normalized = {"received_at": runtime.last_received_at, **item}
                runtime.telemetry_file.write(json.dumps(normalized, separators=(",", ":")) + "\n")
            runtime.telemetry_file.flush()
            missing_sequences = runtime.missing_sequences[:]
            runtime.missing_sequences.clear()
            return {
                "type": "telemetry_ack",
                "schema_version": SCHEMA_VERSION,
                "client_sequence": runtime.last_client_sequence,
                "missing_sequences": missing_sequences,
                "gap_count": runtime.gap_count,
            }

    def _ingest_item(self, runtime: EpochRuntime, item: dict[str, Any]) -> None:
        item_type = item.get("type")
        if item_type == "frame":
            runtime.frame_count += 1
            t_us = _int_value(item.get("t_us", item.get("tUs")))
            runtime.last_frame_t_us = t_us
            pose = _nearest_motion(runtime.motion_samples, t_us)
            joined = {
                "frame_id": item.get("frame_id", item.get("frameId")),
                "t_us": t_us,
                "transport_pts90k": item.get("transport_pts90k"),
                "pose": pose,
                "pose_missing_reason": None if pose is not None else "no_motion_within_50ms",
                "received_at": item["received_at"],
            }
            runtime.join_file.write(json.dumps(joined, separators=(",", ":")) + "\n")
            runtime.join_file.flush()
        elif item_type in {"camera_motion", "motion"}:
            runtime.motion_count += 1
            t_us = _int_value(item.get("t_us", item.get("tUs")))
            if t_us is not None:
                runtime.motion_samples.append((t_us, item))
        elif item_type == "dock":
            runtime.dock_count += 1
        if runtime.frame_joiner is not None:
            ready = runtime.frame_joiner.ingest_item(item, item["received_at"])
            self._enqueue_captured(runtime, ready)

    def artifacts(self, session_id: UUID) -> list[dict[str, Any]]:
        if self.store.get_session(session_id) is None:
            raise KeyError("session not found")
        return self.store.list_artifacts(session_id)

    def save_artifact(
        self,
        session_id: UUID,
        artifact_id: str,
        body: bytes,
        expected_sha256: str,
        expected_length: int,
    ) -> tuple[str, dict[str, Any]]:
        if len(body) != expected_length:
            raise ValueError("content length does not match Content-Length")
        if len(body) > self.settings.max_artifact_bytes:
            raise OverflowError("artifact exceeds configured limit")
        actual_sha256 = hashlib.sha256(body).hexdigest()
        if actual_sha256.lower() != expected_sha256.lower():
            raise ValueError("sha256 does not match payload")
        if self.store.get_session(session_id) is None:
            raise KeyError("session not found")
        existing = self.store.get_artifact(session_id, artifact_id)
        if existing is not None:
            if existing["sha256"].lower() == actual_sha256.lower() and existing[
                "size_bytes"
            ] == len(body):
                return "same", existing
            return "conflict", existing
        artifact_dir = self.settings.root / "sessions" / str(session_id) / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / artifact_id
        path.write_bytes(body)
        result, data = self.store.save_artifact(
            session_id, artifact_id, len(body), actual_sha256, path
        )
        return result, data

    def complete(self, session_id: UUID) -> None:
        if self.store.get_session(session_id) is None:
            raise KeyError("session not found")
        self.store.complete(session_id)

    def status(self, session_id: UUID) -> dict[str, Any]:
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError("session not found")
        with self._lock:
            epochs = []
            for (sid, epoch), runtime in self._epochs.items():
                if sid != str(session_id):
                    continue
                receiver = runtime.receiver.snapshot() if runtime.receiver else None
                epochs.append(
                    {
                        "stream_epoch": epoch,
                        "status": "active",
                        "received_message_count": runtime.received_message_count,
                        "frame_count": runtime.frame_count,
                        "decoded_frame_count": runtime.decoded_frame_count,
                        "joined_frame_count": runtime.joined_frame_count,
                        "join_missing_count": runtime.join_missing_count,
                        "decode_dropped_frames": runtime.frame_queue.dropped,
                        "queue_length": runtime.frame_queue.length,
                        "consumer_attached": runtime.consumer_attached,
                        "motion_count": runtime.motion_count,
                        "dock_count": runtime.dock_count,
                        "last_client_sequence": runtime.last_client_sequence,
                        "gap_count": runtime.gap_count,
                        "duplicate_count": runtime.duplicate_count,
                        "last_received_at": runtime.last_received_at,
                        "receiver": receiver,
                    }
                )
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": str(session_id),
            "device_id": session["device_id"],
            "status": session["status"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "live_epochs": epochs,
            "artifacts": self.store.list_artifacts(session_id),
        }

    def close(self) -> None:
        with self._lock:
            for runtime in self._epochs.values():
                if runtime.receiver:
                    runtime.receiver.stop()
                runtime.close()
            self._epochs.clear()
        self.store.close()


def _wall_clock_us() -> int:
    import time

    return time.time_ns() // 1000


def _int_value(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _nearest_motion(
    samples: deque[tuple[int, dict[str, Any]]],
    frame_t_us: int | None,
) -> dict[str, Any] | None:
    if frame_t_us is None or not samples:
        return None
    eligible = [sample for sample in samples if sample[0] <= frame_t_us]
    if not eligible:
        return None
    sample_t_us, sample = eligible[-1]
    if frame_t_us - sample_t_us > 50_000:
        return None
    return {"sample": sample, "age_us": frame_t_us - sample_t_us}
