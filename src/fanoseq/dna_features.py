"""DNA window descriptors for FanoSeq."""

from __future__ import annotations

from collections import Counter
from math import log2
from typing import Iterable

import numpy as np

from fanoseq.octonion import Octonion

DNA_ALPHABET = ("A", "C", "G", "T")
_DNA_SET = set(DNA_ALPHABET)
_RC_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")


def _clean_sequence(seq: str) -> str:
    return "".join(seq.split()).upper()


def validate_dna(seq: str) -> bool:
    """Return True if the cleaned sequence contains only A/C/G/T."""
    cleaned = _clean_sequence(seq)
    return bool(cleaned) and all(base in _DNA_SET for base in cleaned)


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return _clean_sequence(seq).translate(_RC_TABLE)[::-1]


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
    """Return normalized k-mer entropy over valid DNA k-mers."""
    if k <= 0:
        raise ValueError("k must be > 0.")
    cleaned = _clean_sequence(seq)
    if len(cleaned) < k:
        return 0.0
    kmers = [
        cleaned[i : i + k]
        for i in range(len(cleaned) - k + 1)
        if all(base in _DNA_SET for base in cleaned[i : i + k])
    ]
    return shannon_entropy(kmers, ("".join(kmer) for kmer in _all_kmers(k)))


def reverse_complement_similarity(seq: str) -> float:
    """Return positional identity between a sequence and its reverse complement."""
    cleaned = _clean_sequence(seq)
    if not cleaned:
        return 0.0
    rc = reverse_complement(cleaned)
    matches = sum(left == right for left, right in zip(cleaned, rc, strict=True))
    return matches / len(cleaned)


def encode_dna_window(
    seq: str,
    k: int = 2,
    epsilon: float = 1e-9,
    max_ambiguous_fraction: float = 0.0,
) -> tuple[Octonion, dict[str, float]] | None:
    """Encode a DNA window as an octonion plus auxiliary descriptors."""
    cleaned = _clean_sequence(seq)
    if not cleaned:
        return None
    valid = [base for base in cleaned if base in _DNA_SET]
    ambiguous_count = len(cleaned) - len(valid)
    ambiguous_fraction = ambiguous_count / len(cleaned)
    if ambiguous_fraction > max_ambiguous_fraction or not valid:
        return None

    valid_seq = "".join(valid)
    counts = Counter(valid_seq)
    length = len(valid_seq)
    a = counts["A"]
    c = counts["C"]
    g = counts["G"]
    t = counts["T"]

    valid_fraction = length / len(cleaned)
    gc_content = (g + c) / length
    mono_entropy = shannon_entropy(valid_seq, DNA_ALPHABET)
    components = np.array(
        [
            valid_fraction,
            (a + g - c - t) / length,
            (g + c - a - t) / length,
            (a + c - g - t) / length,
            (g - c) / (g + c + epsilon),
            (a - t) / (a + t + epsilon),
            kmer_entropy(valid_seq, k),
            2.0 * reverse_complement_similarity(valid_seq) - 1.0,
        ],
        dtype=float,
    )
    metadata = {
        "mono_entropy": mono_entropy,
        "gc_content": gc_content,
        "valid_fraction": valid_fraction,
        "ambiguous_fraction": ambiguous_fraction,
    }
    return Octonion(components), metadata


def _all_kmers(k: int) -> list[str]:
    if k == 1:
        return list(DNA_ALPHABET)
    return [base + suffix for base in DNA_ALPHABET for suffix in _all_kmers(k - 1)]

