"""Baseline sequence features for benchmarking FanoSeq descriptors."""

from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Iterable, Literal

import pandas as pd

from fanoseq.codon_features import iter_codons
from fanoseq.dna_features import (
    DNA_ALPHABET,
    kmer_entropy,
    reverse_complement_similarity,
    shannon_entropy,
)
from fanoseq.fasta import FastaRecord
from fanoseq.genetic_code import GeneticCode, all_standard_codons
from fanoseq.protein_features import PROTEIN_ALPHABET

SeqType = Literal["dna", "protein"]

_DNA_SET = set(DNA_ALPHABET)
_AA_SET = set(PROTEIN_ALPHABET)
_FCGR_BITS = {
    "A": (0, 0),
    "C": (0, 1),
    "G": (1, 1),
    "T": (1, 0),
}


def build_baseline_tables(
    records: Iterable[FastaRecord],
    seq_type: SeqType,
    kmer_k: int,
    genetic_code: GeneticCode | None = None,
    frame: int | Literal["all"] = 0,
) -> dict[str, pd.DataFrame]:
    """Build benchmark baseline tables for DNA or protein records."""
    if kmer_k <= 0:
        raise ValueError("kmer_k must be > 0.")
    record_list = list(records)
    if seq_type == "dna":
        if genetic_code is None:
            raise ValueError("DNA baselines require a genetic code.")
        return build_dna_baseline_tables(record_list, kmer_k, genetic_code, frame)
    if seq_type == "protein":
        return build_protein_baseline_tables(record_list, kmer_k)
    raise ValueError("seq_type must be either 'dna' or 'protein'.")


def build_dna_baseline_tables(
    records: Iterable[FastaRecord],
    kmer_k: int,
    genetic_code: GeneticCode,
    frame: int | Literal["all"] = 0,
) -> dict[str, pd.DataFrame]:
    """Build DNA baseline feature tables: composition, k-mers/FCGR, codon usage."""
    sequence_rows: list[dict[str, object]] = []
    kmer_rows: list[dict[str, object]] = []
    kmer_matrix_rows: list[dict[str, object]] = []
    codon_rows: list[dict[str, object]] = []

    kmers = all_kmers(kmer_k, alphabet=DNA_ALPHABET)
    frames = (0, 1, 2) if frame == "all" else (int(frame),)

    for record in records:
        cleaned = _clean(record.sequence)
        valid = "".join(base for base in cleaned if base in _DNA_SET)
        counts = Counter(valid)
        length = len(cleaned)
        valid_length = len(valid)
        a = counts["A"]
        c = counts["C"]
        g = counts["G"]
        t = counts["T"]
        gc_total = g + c
        at_total = a + t
        sequence_rows.append(
            {
                "sequence_id": record.id,
                "seq_type": "dna",
                "length": length,
                "valid_length": valid_length,
                "valid_fraction": valid_length / length if length else 0.0,
                "gc_content": gc_total / valid_length if valid_length else 0.0,
                "at_content": at_total / valid_length if valid_length else 0.0,
                "gc_skew": (g - c) / gc_total if gc_total else 0.0,
                "at_skew": (a - t) / at_total if at_total else 0.0,
                "mono_entropy": shannon_entropy(valid, DNA_ALPHABET),
                "kmer_entropy": kmer_entropy(valid, kmer_k),
                "reverse_complement_similarity": reverse_complement_similarity(valid)
                if valid
                else 0.0,
            }
        )

        kmer_counts = count_kmers(valid, kmer_k, DNA_ALPHABET)
        total_kmers = sum(kmer_counts.values())
        matrix_row: dict[str, object] = {"sequence_id": record.id}
        for kmer in kmers:
            count = kmer_counts[kmer]
            frequency = count / total_kmers if total_kmers else 0.0
            x, y = fcgr_coordinates(kmer)
            kmer_rows.append(
                {
                    "sequence_id": record.id,
                    "k": kmer_k,
                    "kmer": kmer,
                    "count": count,
                    "frequency": frequency,
                    "fcgr_x": x,
                    "fcgr_y": y,
                }
            )
            matrix_row[f"kmer_{kmer}"] = frequency
        kmer_matrix_rows.append(matrix_row)

        for reading_frame in frames:
            codon_rows.extend(
                codon_usage_rows(
                    record.id,
                    cleaned,
                    reading_frame,
                    genetic_code,
                )
            )

    return {
        "baseline_sequence_features": pd.DataFrame(sequence_rows),
        "baseline_kmer_frequencies": pd.DataFrame(kmer_rows),
        "baseline_kmer_feature_matrix": pd.DataFrame(kmer_matrix_rows),
        "baseline_codon_usage": pd.DataFrame(codon_rows),
    }


def build_protein_baseline_tables(
    records: Iterable[FastaRecord],
    kmer_k: int,
) -> dict[str, pd.DataFrame]:
    """Build protein composition and amino-acid k-mer baseline tables."""
    sequence_rows: list[dict[str, object]] = []
    residue_rows: list[dict[str, object]] = []
    kmer_rows: list[dict[str, object]] = []
    kmer_matrix_rows: list[dict[str, object]] = []

    # Protein k-mer spaces grow quickly, so the wide matrix is only emitted for k <= 2.
    protein_kmers = all_kmers(kmer_k, alphabet=PROTEIN_ALPHABET) if kmer_k <= 2 else []

    for record in records:
        cleaned = _clean(record.sequence)
        valid = "".join(residue for residue in cleaned if residue in _AA_SET)
        counts = Counter(valid)
        length = len(cleaned)
        valid_length = len(valid)
        sequence_rows.append(
            {
                "sequence_id": record.id,
                "seq_type": "protein",
                "length": length,
                "valid_length": valid_length,
                "valid_fraction": valid_length / length if length else 0.0,
                "mono_entropy": shannon_entropy(valid, PROTEIN_ALPHABET),
            }
        )
        for residue in PROTEIN_ALPHABET:
            count = counts[residue]
            residue_rows.append(
                {
                    "sequence_id": record.id,
                    "residue": residue,
                    "count": count,
                    "frequency": count / valid_length if valid_length else 0.0,
                }
            )

        kmer_counts = count_kmers(valid, kmer_k, PROTEIN_ALPHABET)
        total_kmers = sum(kmer_counts.values())
        matrix_row: dict[str, object] = {"sequence_id": record.id}
        for kmer, count in sorted(kmer_counts.items()):
            frequency = count / total_kmers if total_kmers else 0.0
            kmer_rows.append(
                {
                    "sequence_id": record.id,
                    "k": kmer_k,
                    "kmer": kmer,
                    "count": count,
                    "frequency": frequency,
                }
            )
        for kmer in protein_kmers:
            matrix_row[f"kmer_{kmer}"] = kmer_counts[kmer] / total_kmers if total_kmers else 0.0
        if protein_kmers:
            kmer_matrix_rows.append(matrix_row)

    tables = {
        "baseline_sequence_features": pd.DataFrame(sequence_rows),
        "baseline_residue_composition": pd.DataFrame(residue_rows),
        "baseline_kmer_frequencies": pd.DataFrame(kmer_rows),
    }
    if kmer_matrix_rows:
        tables["baseline_kmer_feature_matrix"] = pd.DataFrame(kmer_matrix_rows)
    return tables


def count_kmers(sequence: str, k: int, alphabet: Iterable[str]) -> Counter[str]:
    """Count valid k-mers in a sequence."""
    alphabet_set = set(alphabet)
    counts: Counter[str] = Counter()
    if k <= 0:
        raise ValueError("k must be > 0.")
    for index in range(0, max(len(sequence) - k + 1, 0)):
        kmer = sequence[index : index + k]
        if all(symbol in alphabet_set for symbol in kmer):
            counts[kmer] += 1
    return counts


def all_kmers(k: int, alphabet: Iterable[str]) -> list[str]:
    """Return all k-mers for an alphabet in lexical product order."""
    symbols = tuple(alphabet)
    if k <= 0:
        raise ValueError("k must be > 0.")
    return ["".join(values) for values in product(symbols, repeat=k)]


def fcgr_coordinates(kmer: str) -> tuple[int, int]:
    """Return integer FCGR-like grid coordinates for a DNA k-mer."""
    x = 0
    y = 0
    for base in kmer:
        bit_x, bit_y = _FCGR_BITS[base]
        x = (x << 1) + bit_x
        y = (y << 1) + bit_y
    return x, y


def codon_usage_rows(
    sequence_id: str,
    sequence: str,
    frame: int,
    genetic_code: GeneticCode,
) -> list[dict[str, object]]:
    """Build codon-usage baseline rows for one sequence and frame."""
    observed_codons = [
        codon_slice.codon
        for codon_slice in iter_codons(sequence, frame=frame, include_partial_codons=False)
        if all(base in _DNA_SET for base in codon_slice.codon)
    ]
    counts = Counter(observed_codons)
    total = sum(counts.values())
    family_totals: dict[str, int] = {}
    for codon in all_standard_codons():
        aa = genetic_code.amino_acid(codon)
        family_totals[aa] = family_totals.get(aa, 0) + counts[codon]

    rows = []
    for codon in all_standard_codons():
        aa = genetic_code.amino_acid(codon)
        family = genetic_code.synonymous_codons(aa)
        family_size = len(family)
        total_for_aa = family_totals.get(aa, 0)
        expected = total_for_aa / family_size if family_size and total_for_aa else 0.0
        observed = counts[codon]
        rows.append(
            {
                "sequence_id": sequence_id,
                "frame": frame,
                "codon": codon,
                "amino_acid": aa,
                "is_stop": genetic_code.is_stop(codon),
                "count": observed,
                "frequency": observed / total if total else 0.0,
                "gc3": 1.0 if codon[2] in {"G", "C"} else 0.0,
                "synonymous_family_size": family_size,
                "rscu": observed / expected if expected else 0.0,
            }
        )
    return rows


def _clean(sequence: str) -> str:
    return "".join(sequence.split()).upper()
