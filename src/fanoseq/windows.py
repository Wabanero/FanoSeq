"""Sliding-window helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class SequenceWindow:
    """A single sliding window over a sequence."""

    position: int
    start: int
    end: int
    sequence: str


def iter_windows(sequence: str, window_size: int, step: int) -> Iterator[SequenceWindow]:
    """Yield fixed-width windows with 0-based coordinates."""
    if window_size <= 0:
        raise ValueError("window_size must be > 0.")
    if step <= 0:
        raise ValueError("step must be > 0.")

    position = 0
    for start in range(0, max(len(sequence) - window_size + 1, 0), step):
        end = start + window_size
        yield SequenceWindow(position=position, start=start, end=end, sequence=sequence[start:end])
        position += 1

