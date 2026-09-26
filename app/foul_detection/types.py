"""Self-contained types for the foul detection subsystem.

The previous `offside.foul_model` adapter imported its `FoulPrediction`
from an external package (`fouls_far`) that was never shipped with this
repository. This module provides a drop-in replacement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FoulPrediction:
    """One suspicious foul candidate.

    This is intentionally a plain dataclass rather than a Pydantic model:
    the foul detection path is invoked every stride frames inside the
    inference thread, and keeping it lightweight matters more than
    validation.
    """

    confidence: float = 0.0
    action: Optional[str] = None
    severity: Optional[str] = None
    card: str = "none"
    decision: str = "no_offence"
    raw_scores: dict[str, Any] = field(default_factory=dict)


__all__ = ["FoulPrediction"]
