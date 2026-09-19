"""Field capture ingest and transport boundary.

This package deliberately stays separate from the inference pipeline.  It
accepts capture sessions, telemetry, and SRT video, then exposes bounded
diagnostics that can be consumed by a future inference adapter.
"""

from app.field_ingest.service import FieldIngestService

__all__ = ["FieldIngestService"]
