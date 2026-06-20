"""Codon-level ordered octonion descriptors."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log2
from typing import Iterator

import numpy as np

from fanoseq.genetic_code import GeneticCode
from fanoseq.octonion import Octonion

BASE_PROPERTIES: dict[str, tuple[float, float, float]] = {
    "A": (1.0, -1.0, 1.0),
    "C": (-1.0, 1.0, 1.0),
    "G": (1.0, 1.0, -1.0),
    "T": (-1.0, -1.0, -1.0),
}
DNA_BASES = set(BASE_PROPERTIES)


@dataclass(frozen=True)
class CodonSlice:
    """A codon slice in a selected reading frame."""

    codon_index: int
    start: int
    end: int
    codon: str


@dataclass(frozen=True)
class CodonEncoding:
    """A codon octonion and its tabular metadata."""

    codon: str
    octonion: Octonion
    metadata: dict[str, float | str | bool]


def base_position_octonion(base: str, position: int) -> Octonion:
    """Return the position-aware base octonion B(base, position)."""
    if position not in {1, 2, 3}:
        raise ValueError("position must be 1, 2, or 3.")
    upper = base.upper()
    if upper not in DNA_BASES:
        ry, sw, mk = 0.0, 0.0, 0.0
        scalar = 0.0
    else:
        ry, sw, mk = BASE_PROPERTIES[upper]
        scalar = 1.0

    p1 = 1.0 if position == 1 else 0.0
    p2 = 1.0 if position == 2 else 0.0
    p3 = 1.0 if position == 3 else 0.0
    wobble = 1.0 if position == 3 else 0.0
    return Octonion([scalar, ry, sw, mk, p1, p2, p3, wobble])


def encode_codon(
    codon: str,
    genetic_code: GeneticCode,
    max_ambiguous_fraction: float = 0.0,
    include_stop_codons: bool = True,
    normalize: bool = False,
) -> CodonEncoding | None:
    """Encode a length-3 DNA codon as an ordered octonion product."""
    cleaned = "".join(codon.split()).upper()
    if len(cleaned) != 3:
        return None
    invalid_count = sum(base not in DNA_BASES for base in cleaned)
    ambiguous_fraction = invalid_count / 3.0
    if ambiguous_fraction > max_ambiguous_fraction:
        return None

    amino_acid = genetic_code.amino_acid(cleaned) if invalid_count == 0 else "X"
    is_stop = genetic_code.is_stop(cleaned) if invalid_count == 0 else False
    if is_stop and not include_stop_codons:
        return None

    base_octonions = [
        base_position_octonion(cleaned[0], 1),
        base_position_octonion(cleaned[1], 2),
        base_position_octonion(cleaned[2], 3),
    ]
    codon_octonion = (base_octonions[0] * base_octonions[1]) * base_octonions[2]
    if normalize:
        norm = codon_octonion.norm()
        if norm > 0:
            codon_octonion = codon_octonion / norm
    associator = base_octonions[0].associator(base_octonions[1], base_octonions[2])
    valid_count = 3 - invalid_count
    valid_fraction = valid_count / 3.0

    gc_indicators = [1.0 if base in {"G", "C"} else 0.0 for base in cleaned]
    ry = [BASE_PROPERTIES.get(base, (0.0, 0.0, 0.0))[0] for base in cleaned]
    sw = [BASE_PROPERTIES.get(base, (0.0, 0.0, 0.0))[1] for base in cleaned]
    mk = [BASE_PROPERTIES.get(base, (0.0, 0.0, 0.0))[2] for base in cleaned]
    gc_content = sum(gc_indicators) / valid_count if valid_count else np.nan

    metadata: dict[str, float | str | bool] = {
        "amino_acid": amino_acid,
        "is_start": genetic_code.is_start(cleaned) if invalid_count == 0 else False,
        "is_stop": is_stop,
        "valid_fraction": valid_fraction,
        "ambiguous_fraction": ambiguous_fraction,
        "gc_content": gc_content,
        "gc1": gc_indicators[0],
        "gc2": gc_indicators[1],
        "gc3": gc_indicators[2],
        "ry_pos1": ry[0],
        "ry_pos2": ry[1],
        "ry_pos3": ry[2],
        "sw_pos1": sw[0],
        "sw_pos2": sw[1],
        "sw_pos3": sw[2],
        "mk_pos1": mk[0],
        "mk_pos2": mk[1],
        "mk_pos3": mk[2],
        "codon_associator_score": associator.norm(),
    }
    return CodonEncoding(codon=cleaned, octonion=codon_octonion, metadata=metadata)


def iter_codons(sequence: str, frame: int, include_partial_codons: bool = False) -> Iterator[CodonSlice]:
    """Yield codons for a 0, 1, or 2 reading frame."""
    if frame not in {0, 1, 2}:
        raise ValueError("frame must be 0, 1, or 2.")
    cleaned = "".join(sequence.split()).upper()
    codon_index = 0
    for start in range(frame, len(cleaned), 3):
        codon = cleaned[start : start + 3]
        if len(codon) < 3 and not include_partial_codons:
            break
        if len(codon) < 3:
            codon = codon + "N" * (3 - len(codon))
        yield CodonSlice(codon_index=codon_index, start=start, end=min(start + 3, len(cleaned)), codon=codon)
        codon_index += 1


def codon_entropy(codons: list[str]) -> float:
    """Return Shannon entropy over codon frequencies, normalized by log2(64)."""
    if not codons:
        return 0.0
    counts = Counter(codons)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * log2(probability)
    return float(entropy / log2(64))

