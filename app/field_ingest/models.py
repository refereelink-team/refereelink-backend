from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"


class WireModel(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        str_strip_whitespace=False,
    )


class FieldSessionRegistration(WireModel):
    schema_version: str = Field(
        default=SCHEMA_VERSION,
        validation_alias=AliasChoices("schema_version", "schemaVersion"),
        serialization_alias="schema_version",
    )
    session_id: UUID = Field(
        validation_alias=AliasChoices("session_id", "sessionId"),
        serialization_alias="session_id",
    )
    device_id: UUID = Field(
        validation_alias=AliasChoices("device_id", "deviceId"),
        serialization_alias="device_id",
    )
    device_name: str = Field(
        default="iPhone",
        validation_alias=AliasChoices("device_name", "deviceName"),
        serialization_alias="device_name",
    )
    capabilities: list[str] = Field(default_factory=list)


class HelloMessage(WireModel):
    type: Literal["hello"]
    schema_version: str = Field(
        default=SCHEMA_VERSION,
        validation_alias=AliasChoices("schema_version", "schemaVersion"),
        serialization_alias="schema_version",
    )
    session_id: UUID = Field(
        validation_alias=AliasChoices("session_id", "sessionId"),
        serialization_alias="session_id",
    )
    device_id: UUID | None = Field(
        default=None,
        validation_alias=AliasChoices("device_id", "deviceId"),
        serialization_alias="device_id",
    )
    client_sequence: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices("client_sequence", "clientSequence"),
        serialization_alias="client_sequence",
    )


class TelemetryBatch(WireModel):
    type: Literal["telemetry_batch"]
    schema_version: str = Field(
        default=SCHEMA_VERSION,
        validation_alias=AliasChoices("schema_version", "schemaVersion"),
        serialization_alias="schema_version",
    )
    session_id: UUID = Field(
        validation_alias=AliasChoices("session_id", "sessionId"),
        serialization_alias="session_id",
    )
    stream_epoch: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices("stream_epoch", "streamEpoch"),
        serialization_alias="stream_epoch",
    )
    client_sequence: int = Field(
        ge=0,
        validation_alias=AliasChoices("client_sequence", "clientSequence"),
        serialization_alias="client_sequence",
    )
    items: list[dict[str, Any]] = Field(default_factory=list)


class LegacyTelemetryMessage(WireModel):
    """Compatibility envelope accepted during the iOS transport migration."""

    type: str
    session_id: UUID = Field(
        validation_alias=AliasChoices("session_id", "sessionId"),
        serialization_alias="session_id",
    )
    payload: dict[str, Any] = Field(default_factory=dict)
    client_sequence: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("client_sequence", "clientSequence"),
        serialization_alias="client_sequence",
    )


class ClockProbeMessage(WireModel):
    type: Literal["clock_probe"]
    client_sequence: int = Field(
        ge=0,
        validation_alias=AliasChoices("client_sequence", "clientSequence"),
        serialization_alias="client_sequence",
    )
    t1_client_send_us: int = Field(
        validation_alias=AliasChoices("t1_client_send_us", "t1ClientSendUs"),
        serialization_alias="t1_client_send_us",
    )


class CapabilityResponse(WireModel):
    schema_version: str = Field(default=SCHEMA_VERSION, serialization_alias="schema_version")
    supported_schema_versions: list[str] = Field(
        default_factory=lambda: [SCHEMA_VERSION],
        serialization_alias="supported_schema_versions",
    )
    video_profiles: list[dict[str, Any]] = Field(
        default_factory=list,
        serialization_alias="video_profiles",
    )
    receiver_available: bool = Field(default=False, serialization_alias="receiver_available")
    max_artifact_bytes: int = Field(
        default=8 * 1024 * 1024, serialization_alias="max_artifact_bytes"
    )
    upload_concurrency: int = Field(default=2, serialization_alias="upload_concurrency")
    transport: dict[str, Any] = Field(default_factory=dict)


class LiveAllocationRequest(WireModel):
    profile: str = "720p30"
    latency_ms: int = Field(
        default=200,
        ge=20,
        le=5000,
        validation_alias=AliasChoices("latency_ms", "latencyMs"),
        serialization_alias="latency_ms",
    )


class LiveAllocationResponse(WireModel):
    session_id: UUID = Field(serialization_alias="session_id")
    stream_epoch: int = Field(serialization_alias="stream_epoch")
    srt_host: str = Field(serialization_alias="srt_host")
    srt_port: int = Field(serialization_alias="srt_port")
    stream_token: str = Field(serialization_alias="stream_token")
    latency_ms: int = Field(serialization_alias="latency_ms")
    profile: str


class ArtifactCompleteRequest(WireModel):
    manifest: dict[str, Any] = Field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
