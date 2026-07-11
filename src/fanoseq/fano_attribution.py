"""Fano-line attribution for octonion products."""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fanoseq.axis_schemes import axis_labels_for_context, resolve_axis_scheme
from fanoseq.octonion import FANO_LINES


def axis_labels(mode: str, seq_type: str, scheme_id: str | None = None) -> dict[int, str]:
    """Return axis labels for a mode and sequence type."""
    return axis_labels_for_context(seq_type=seq_type, mode=mode, scheme_id=scheme_id)


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
    axis_scheme_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return seven Fano-line contribution rows for one adjacent product."""
    scheme = resolve_axis_scheme(seq_type=seq_type, mode=mode, scheme_id=axis_scheme_id)
    labels = scheme.axis_labels()
    rows: list[dict[str, Any]] = []
    for a, b, c in FANO_LINES:
        fano_line = (a, b, c)
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
                "axis_scheme_id": scheme.scheme_id,
                "frame": frame,
                "position": position,
                "left_object": left_object,
                "right_object": right_object,
                "fano_line": f"({a},{b},{c})",
                "line_label": scheme.line_label(fano_line),
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
