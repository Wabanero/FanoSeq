"""Protein window descriptors for FanoSeq."""

from __future__ import annotations

from collections import Counter
from math import log2
from typing import Iterable

import numpy as np

from fanoseq.octonion import Octonion

PROTEIN_ALPHABET = tuple("ARNDCQEGHILKMFPSTWYV")
_PROTEIN_SET = set(PROTEIN_ALPHABET)

HYDROPATHY = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

POLARITY = {
    "A": -0.4,
    "R": 1.0,
    "N": 0.8,
    "D": 1.0,
    "C": 0.0,
    "Q": 0.8,
    "E": 1.0,
    "G": 0.1,
    "H": 0.6,
    "I": -1.0,
    "L": -1.0,
    "K": 1.0,
    "M": -0.5,
    "F": -0.8,
    "P": 0.1,
    "S": 0.7,
    "T": 0.6,
    "W": -0.4,
    "Y": 0.2,
    "V": -0.9,
}

RESIDUE_VOLUME = {
    "A": 88.6,
    "R": 173.4,
    "N": 114.1,
    "D": 111.1,
    "C": 108.5,
    "Q": 143.8,
    "E": 138.4,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "L": 166.7,
    "K": 168.6,
    "M": 162.9,
    "F": 189.9,
    "P": 112.7,
    "S": 89.0,
    "T": 116.1,
    "W": 227.8,
    "Y": 193.6,
    "V": 140.0,
}

DISORDER_PROMOTING = set("PESKQG")
ORDER_PROMOTING = set("WFYILVC")
AROMATIC = set("FWYH")


def _clean_sequence(seq: str) -> str:
    return "".join(seq.split()).upper()


def validate_protein(seq: str) -> bool:
    """Return True if the cleaned sequence contains only the 20 standard residues."""
    cleaned = _clean_sequence(seq)
    return bool(cleaned) and all(residue in _PROTEIN_SET for residue in cleaned)


def shannon_entropy(chars: Iterable[str] | str, alphabet: Iterable[str]) -> float:
    """Return Shannon entropy normalized by log2(len(alphabet))."""
    values = list(chars)
    alphabet_values = list(alphabet)
    if not values or not alphabet_values:
        return 0.0
    counts = Counter(value for value in values if value in alphabet_values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * log2(probability)
    max_entropy = log2(len(alphabet_values))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def kmer_entropy(seq: str, k: int) -> float:
    """Return normalized amino-acid k-mer entropy over observed valid k-mers."""
    if k <= 0:
        raise ValueError("k must be > 0.")
    cleaned = _clean_sequence(seq)
    if len(cleaned) < k:
        return 0.0
    kmers = [
        cleaned[i : i + k]
        for i in range(len(cleaned) - k + 1)
        if all(residue in _PROTEIN_SET for residue in cleaned[i : i + k])
    ]
    if not kmers:
        return 0.0
    counts = Counter(kmers)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * log2(probability)
    return float(entropy / log2(len(PROTEIN_ALPHABET) ** k))


def encode_protein_window(
    seq: str,
    k: int = 2,
    epsilon: float = 1e-9,
    max_ambiguous_fraction: float = 0.0,
) -> tuple[Octonion, dict[str, float]] | None:
    """Encode a protein window as an octonion plus auxiliary descriptors."""
    del epsilon
    cleaned = _clean_sequence(seq)
    if not cleaned:
        return None
    valid = [residue for residue in cleaned if residue in _PROTEIN_SET]
    ambiguous_count = len(cleaned) - len(valid)
    ambiguous_fraction = ambiguous_count / len(cleaned)
    if ambiguous_fraction > max_ambiguous_fraction or not valid:
        return None

    length = len(valid)
    counts = Counter(valid)
    valid_fraction = length / len(cleaned)
    hydro = float(np.mean([HYDROPATHY[residue] / 4.5 for residue in valid]))
    charge = (counts["K"] + counts["R"] + 0.1 * counts["H"] - counts["D"] - counts["E"]) / length
    polarity = float(np.mean([POLARITY[residue] for residue in valid]))
    aromatic = sum(counts[residue] for residue in AROMATIC) / length
    volume_values = np.array(list(RESIDUE_VOLUME.values()), dtype=float)
    v_min = float(volume_values.min())
    v_max = float(volume_values.max())
    volume = float(
        np.mean([2.0 * (RESIDUE_VOLUME[residue] - v_min) / (v_max - v_min) - 1.0 for residue in valid])
    )
    disorder_fraction = sum(counts[residue] for residue in DISORDER_PROMOTING) / length
    order_fraction = sum(counts[residue] for residue in ORDER_PROMOTING) / length
    repeat_score = 1.0 - kmer_entropy("".join(valid), k)

    components = np.array(
        [
            valid_fraction,
            hydro,
            charge,
            polarity,
            aromatic,
            volume,
            disorder_fraction - order_fraction,
            repeat_score,
        ],
        dtype=float,
    )
    metadata = {
        "mono_entropy": shannon_entropy(valid, PROTEIN_ALPHABET),
        "gc_content": np.nan,
        "valid_fraction": valid_fraction,
        "ambiguous_fraction": ambiguous_fraction,
    }
    return Octonion(components), metadata

