"""Fano-line attribution for octonion products."""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fanoseq.octonion import FANO_LINES

DNA_AXIS_LABELS = {
    1: "purine/pyrimidine balance",
    2: "GC/AT balance",
    3: "amino/keto balance",
    4: "GC skew",
    5: "AT skew",
    6: "k-mer entropy",
    7: "reverse-complement symmetry",
}

PROTEIN_AXIS_LABELS = {
    1: "hydrophobicity",
    2: "net charge",
    3: "polarity",
    4: "aromaticity",
    5: "residue volume",
    6: "disorder/flexibility proxy",
    7: "repeat/low-complexity score",
}

CODON_AXIS_LABELS = {
    1: "base RY property",
    2: "base SW property",
    3: "base MK property",
    4: "position-1 gate",
    5: "position-2 gate",
    6: "position-3 gate",
    7: "wobble-position marker",
}


def axis_labels(mode: str, seq_type: str) -> dict[int, str]:
    """Return axis labels for a mode and sequence type."""
    if mode == "codon":
        return CODON_AXIS_LABELS
    if seq_type == "dna":
        return DNA_AXIS_LABELS
    if seq_type == "protein":
        return PROTEIN_AXIS_LABELS
    raise ValueError(f"Unsupported seq_type for Fano attribution: {seq_type}")


def fano_line_attribution(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    *,
    sequence_id: str,
    mode: str,
    seq_type: str,
    frame: int | str,
    position: int,
    left_object: str,
    right_object: str,
) -> list[dict[str, Any]]:
    """Return seven Fano-line contribution rows for one adjacent product."""
    labels = axis_labels(mode, seq_type)
    rows: list[dict[str, Any]] = []
    for a, b, c in FANO_LINES:
        pair_ab_to_c = float(left[a] * right[b] - left[b] * right[a])
        pair_bc_to_a = float(left[b] * right[c] - left[c] * right[b])
        pair_ca_to_b = float(left[c] * right[a] - left[a] * right[c])
        contribution_to_a = pair_bc_to_a
        contribution_to_b = pair_ca_to_b
        contribution_to_c = pair_ab_to_c
        rows.append(
            {
                "sequence_id": sequence_id,
                "mode": mode,
                "seq_type": seq_type,
                "frame": frame,
                "position": position,
                "left_object": left_object,
                "right_object": right_object,
                "fano_line": f"({a},{b},{c})",
                "axis_a": a,
                "axis_b": b,
                "axis_c": c,
                "axis_a_label": labels[a],
                "axis_b_label": labels[b],
                "axis_c_label": labels[c],
                "pair_ab_to_c": pair_ab_to_c,
                "pair_bc_to_a": pair_bc_to_a,
                "pair_ca_to_b": pair_ca_to_b,
                "contribution_to_a": contribution_to_a,
                "contribution_to_b": contribution_to_b,
                "contribution_to_c": contribution_to_c,
                "line_contribution_norm": sqrt(
                    contribution_to_a**2 + contribution_to_b**2 + contribution_to_c**2
                ),
            }
        )
    return rows

