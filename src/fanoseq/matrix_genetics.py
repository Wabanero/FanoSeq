"""Codon-matrix and algebraic genetic-code summaries.

The functions in this module are Petoukhov- and Fano-inspired analysis tools,
not a claim to reproduce one specific matrix-genetics formalism. They provide a
stable 8x8 codon layout, degeneracy/root summaries, Walsh-Hadamard spectra, and
dyadic-shift diagnostics that can be benchmarked against ordinary codon features.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from fanoseq.codon_features import BASE_PROPERTIES
from fanoseq.genetic_code import GeneticCode, all_standard_codons

GF8_LABELS = ("0", "1", "a", "a^2", "a+1", "a^2+a", "a^2+a+1", "a^2+1")
BASE_BITS = {
    "A": (0, 0),
    "C": (0, 1),
    "G": (1, 0),
    "T": (1, 1),
}


@dataclass(frozen=True)
class CodonMatrixEntry:
    """One cell in the canonical 8x8 codon matrix."""

    codon: str
    amino_acid: str
    row: int
    column: int
    row_gf8: str
    column_gf8: str
    root: str
    third_base: str
    is_start: bool
    is_stop: bool
    synonymous_family_size: int
    root_degeneracy: int
    root_is_strong: bool
    row_bits: str
    column_bits: str


def canonical_codon_order() -> list[str]:
    """Return the stable 64-codon order used for 8x8 matrix analyses."""
    return all_standard_codons()


def codon_to_matrix_position(codon: str) -> tuple[int, int]:
    """Return row and column in the canonical 8x8 codon matrix."""
    upper = codon.upper()
    if upper not in set(all_standard_codons()):
        raise ValueError(f"Expected an unambiguous DNA codon, got {codon!r}.")
    index = canonical_codon_order().index(upper)
    return divmod(index, 8)


def build_codon_matrix_entries(genetic_code: GeneticCode) -> pd.DataFrame:
    """Build a tidy 8x8 codon matrix with amino-acid and degeneracy metadata."""
    root_summary = build_root_degeneracy(genetic_code).set_index("root")
    rows: list[dict[str, object]] = []
    for index, codon in enumerate(canonical_codon_order()):
        row, column = divmod(index, 8)
        amino_acid = genetic_code.amino_acid(codon)
        family_size = len(genetic_code.synonymous_codons(amino_acid))
        root = codon[:2]
        row_bits = _bits_for_matrix_index(row)
        column_bits = _bits_for_matrix_index(column)
        entry = CodonMatrixEntry(
            codon=codon,
            amino_acid=amino_acid,
            row=row,
            column=column,
            row_gf8=GF8_LABELS[row],
            column_gf8=GF8_LABELS[column],
            root=root,
            third_base=codon[2],
            is_start=genetic_code.is_start(codon),
            is_stop=genetic_code.is_stop(codon),
            synonymous_family_size=family_size,
            root_degeneracy=int(root_summary.loc[root, "root_degeneracy"]),
            root_is_strong=bool(root_summary.loc[root, "root_is_strong"]),
            row_bits=row_bits,
            column_bits=column_bits,
        )
        rows.append(entry.__dict__)
    return pd.DataFrame(rows)


def build_root_degeneracy(genetic_code: GeneticCode) -> pd.DataFrame:
    """Summarize first-two-base roots across all four wobble positions."""
    rows: list[dict[str, object]] = []
    for first, second in product("ACGT", repeat=2):
        root = first + second
        codons = [root + third for third in "ACGT"]
        amino_acids = [genetic_code.amino_acid(codon) for codon in codons]
        unique_aas = sorted(set(amino_acids))
        root_is_strong = len(unique_aas) == 1
        root_properties = np.array([BASE_PROPERTIES[first], BASE_PROPERTIES[second]], dtype=float)
        rows.append(
            {
                "root": root,
                "codons": ",".join(codons),
                "amino_acids": ",".join(amino_acids),
                "unique_amino_acids": ",".join(unique_aas),
                "root_degeneracy": len(unique_aas),
                "root_is_strong": root_is_strong,
                "root_ry_mean": float(root_properties[:, 0].mean()),
                "root_sw_mean": float(root_properties[:, 1].mean()),
                "root_mk_mean": float(root_properties[:, 2].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_hadamard_spectrum(
    genetic_code: GeneticCode, value: str = "family_size"
) -> pd.DataFrame:
    """Compute a Walsh-Hadamard spectrum over the canonical 64-codon vector."""
    values = _numeric_codon_vector(genetic_code, value)
    coefficients = _walsh_hadamard(values) / np.sqrt(len(values))
    rows: list[dict[str, object]] = []
    for index, coefficient in enumerate(coefficients):
        rows.append(
            {
                "basis_index": index,
                "basis_bits": format(index, "06b"),
                "coefficient": float(coefficient),
                "abs_coefficient": float(abs(coefficient)),
                "energy": float(coefficient * coefficient),
                "value": value,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def build_dyadic_shift_summary(genetic_code: GeneticCode) -> pd.DataFrame:
    """Compare codon labels under XOR/dyadic shifts of the 64-codon vector."""
    codons = canonical_codon_order()
    amino_acids = np.array([genetic_code.amino_acid(codon) for codon in codons], dtype=object)
    family_sizes = np.array(
        [len(genetic_code.synonymous_codons(genetic_code.amino_acid(codon))) for codon in codons],
        dtype=float,
    )
    rows: list[dict[str, object]] = []
    for shift in range(1, 64):
        shifted_indices = np.array([index ^ shift for index in range(64)], dtype=int)
        same_aa = amino_acids == amino_acids[shifted_indices]
        family_delta = family_sizes - family_sizes[shifted_indices]
        rows.append(
            {
                "shift": shift,
                "shift_bits": format(shift, "06b"),
                "same_amino_acid_fraction": float(np.mean(same_aa)),
                "mean_abs_family_size_delta": float(np.mean(np.abs(family_delta))),
                "max_abs_family_size_delta": float(np.max(np.abs(family_delta))),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["same_amino_acid_fraction", "mean_abs_family_size_delta"],
        ascending=[False, True],
    )


def build_gf8_codon_labels(genetic_code: GeneticCode) -> pd.DataFrame:
    """Return GF(8)-style row/column labels for the canonical codon matrix."""
    entries = build_codon_matrix_entries(genetic_code)
    return entries[
        [
            "codon",
            "amino_acid",
            "row",
            "column",
            "row_gf8",
            "column_gf8",
            "row_bits",
            "column_bits",
            "root",
            "root_is_strong",
        ]
    ].copy()


def build_matrix_genetics_tables(genetic_code: GeneticCode) -> dict[str, pd.DataFrame]:
    """Build the complete first-pass matrix-genetics table set."""
    return {
        "codon_matrix_entries": build_codon_matrix_entries(genetic_code),
        "codon_degeneracy_roots": build_root_degeneracy(genetic_code),
        "codon_hadamard_spectrum": build_hadamard_spectrum(genetic_code),
        "codon_dyadic_shifts": build_dyadic_shift_summary(genetic_code),
        "gf8_codon_labels": build_gf8_codon_labels(genetic_code),
    }


def _numeric_codon_vector(genetic_code: GeneticCode, value: str) -> np.ndarray:
    codons = canonical_codon_order()
    if value == "family_size":
        return np.array(
            [len(genetic_code.synonymous_codons(genetic_code.amino_acid(codon))) for codon in codons],
            dtype=float,
        )
    if value == "stop":
        return np.array([1.0 if genetic_code.is_stop(codon) else 0.0 for codon in codons])
    if value == "hydrophobic_proxy":
        hydrophobic = set("AVLIMFWY")
        return np.array(
            [1.0 if genetic_code.amino_acid(codon) in hydrophobic else 0.0 for codon in codons],
            dtype=float,
        )
    raise ValueError("value must be one of: family_size, stop, hydrophobic_proxy.")


def _walsh_hadamard(values: np.ndarray) -> np.ndarray:
    transformed = np.asarray(values, dtype=float).copy()
    length = transformed.shape[0]
    if length == 0 or length & (length - 1):
        raise ValueError("Walsh-Hadamard input length must be a power of two.")
    step = 1
    while step < length:
        for start in range(0, length, step * 2):
            left = transformed[start : start + step].copy()
            right = transformed[start + step : start + step * 2].copy()
            transformed[start : start + step] = left + right
            transformed[start + step : start + step * 2] = left - right
        step *= 2
    return transformed


def _bits_for_matrix_index(index: int) -> str:
    return format(index, "03b")
