"""Explainable event candidates derived from entity trajectories."""

from app.events.contact import ContactTrigger, ContactTriggerConfig, format_player_identity
from app.events.engine import EventEngine, EventEngineConfig, FoulEventAdapter

__all__ = [
    "ContactTrigger",
    "ContactTriggerConfig",
    "EventEngine",
    "EventEngineConfig",
    "FoulEventAdapter",
    "format_player_identity",
]
