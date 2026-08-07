from __future__ import annotations

import numpy as np

from app.pipeline.engine import _draw_player_overlay, _format_player_overlay_label


def test_final_overlay_uses_compact_team_and_role_labels() -> None:
    assert _format_player_overlay_label(
        team_label="home",
        role_label="outfield",
        display_id=10,
    ) == "H10"
    assert _format_player_overlay_label(
        team_label="away",
        role_label="outfield",
        display_id=39,
    ) == "A39"
    assert _format_player_overlay_label(
        team_label="home",
        role_label="goalkeeper",
        display_id=6,
    ) == "HG"
    assert _format_player_overlay_label(
        team_label="away",
        role_label="goalkeeper",
        display_id=79,
    ) == "AG"
    assert _format_player_overlay_label(
        team_label="none",
        role_label="referee",
        display_id=40,
    ) == "REF"


def test_unknown_player_is_not_drawn_in_final_overlay() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    before = frame.copy()

    _draw_player_overlay(
        frame,
        bbox=(40, 30, 70, 100),
        track_id=12,
        entity_id=12,
        team_label="unknown",
        role_label="unknown",
        color=(147, 20, 255),
    )

    assert np.array_equal(frame, before)


def test_known_player_overlay_draws_compact_marker_without_full_box() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    _draw_player_overlay(
        frame,
        bbox=(40, 30, 70, 100),
        track_id=12,
        entity_id=12,
        team_label="home",
        role_label="outfield",
        color=(147, 20, 255),
    )

    # The compact marker must add pixels, while the top-left corner of the
    # previous full bounding box remains untouched.
    assert np.count_nonzero(frame) > 0
    assert np.array_equal(frame[30, 40], np.zeros(3, dtype=np.uint8))
