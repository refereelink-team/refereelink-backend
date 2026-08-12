#!/usr/bin/env python3
"""Fail when any SoccerNet sequence appears in more than one data split."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.field_registration.soccernet import (
    SoccerNetIndex,
    assert_sequence_disjoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("indexes", type=Path, nargs="+")
    arguments = parser.parse_args()
    indexes = [SoccerNetIndex.load(path) for path in arguments.indexes]
    assert_sequence_disjoint(indexes)
    print(
        "SoccerNet splits are sequence-disjoint: "
        + ", ".join(f"{index.split}={len(index.frames)}" for index in indexes)
    )


if __name__ == "__main__":
    main()
