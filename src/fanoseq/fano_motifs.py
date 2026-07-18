"""Fano-line motif and triad counting utilities."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import pandas as pd

from fanoseq.octonion import FANO_LINES

FANO_LINE_SETS = {frozenset(line): line for line in FANO_LINES}


def fano_triad_counts(
    symbols: Iterable[str],
    symbol_to_axis: dict[str, int],
    stride: int = 1,
) -> pd.DataFrame:
    """Count sliding symbol triples whose mapped axes fall on Fano lines."""
    if stride <= 0:
        raise ValueError("stride must be > 0.")
    cleaned = [str(symbol) for symbol in symbols]
    counter: Counter[tuple[int, int, int, bool]] = Counter()
    examples: dict[tuple[int, int, int, bool], str] = {}
    for start in range(0, max(len(cleaned) - 2, 0), stride):
        triplet = cleaned[start : start + 3]
        axis_a, axis_b, axis_c = (symbol_to_axis.get(symbol, -1) for symbol in triplet)
        axes = (axis_a, axis_b, axis_c)
        if any(axis < 1 or axis > 7 for axis in axes):
            continue
        line = FANO_LINE_SETS.get(frozenset(axes))
        is_fano_line = line is not None
        key = (axis_a, axis_b, axis_c, is_fano_line)
        counter[key] += 1
        examples.setdefault(key, ",".join(triplet))

    rows: list[dict[str, object]] = []
    for (axis_a, axis_b, axis_c, is_fano_line), count in counter.items():
        canonical = FANO_LINE_SETS.get(frozenset((axis_a, axis_b, axis_c)))
        rows.append(
            {
                "axis_a": axis_a,
                "axis_b": axis_b,
                "axis_c": axis_c,
                "fano_line": str(canonical) if canonical else "NA",
                "is_fano_line": is_fano_line,
                "count": count,
                "example_symbols": examples[(axis_a, axis_b, axis_c, is_fano_line)],
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def amino_acid_axis_map() -> dict[str, int]:
    """Return an interpretable amino-acid-to-axis map for motif triads."""
    return {
        "A": 1,
        "G": 1,
        "S": 1,
        "V": 2,
        "L": 2,
        "I": 2,
        "M": 2,
        "F": 3,
        "W": 3,
        "Y": 3,
        "T": 4,
        "N": 4,
        "Q": 4,
        "K": 5,
        "R": 5,
        "H": 5,
        "D": 6,
        "E": 6,
        "C": 7,
        "P": 7,
    }


def dna_base_axis_map() -> dict[str, int]:
    """Return a simple DNA-base axis map for Fano-line triads."""
    return {"A": 1, "C": 2, "G": 3, "T": 4}
