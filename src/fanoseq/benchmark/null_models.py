"""Sequence and representation null models for FanoSeq benchmarks."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import permutations

import numpy as np
import pandas as pd

from fanoseq.genetic_code import GeneticCode
from fanoseq.octonion import FANO_LINES


def mononucleotide_shuffle(sequence: str, rng: np.random.Generator) -> str:
    """Shuffle bases while preserving mononucleotide counts exactly."""
    symbols = np.asarray(list(sequence))
    rng.shuffle(symbols)
    return "".join(symbols.tolist())


def dinucleotide_preserving_shuffle(sequence: str, rng: np.random.Generator) -> str:
    """Shuffle a sequence while preserving directed dinucleotide counts exactly."""
    if len(sequence) < 2:
        return sequence
    edges: dict[str, list[str]] = defaultdict(list)
    for left, right in zip(sequence[:-1], sequence[1:]):
        edges[left].append(right)
    for outgoing in edges.values():
        rng.shuffle(outgoing)

    stack = [sequence[0]]
    path: list[str] = []
    while stack:
        node = stack[-1]
        if edges[node]:
            stack.append(edges[node].pop())
        else:
            path.append(stack.pop())
    shuffled = "".join(reversed(path))
    if len(shuffled) != len(sequence) or (
        dinucleotide_counts(shuffled) != dinucleotide_counts(sequence)
    ):
        return sequence
    return shuffled


def codon_order_shuffle(
    sequence: str,
    rng: np.random.Generator,
    *,
    frame: int = 0,
) -> str:
    """Shuffle complete in-frame codons while preserving codon counts exactly."""
    if frame not in {0, 1, 2}:
        raise ValueError("frame must be 0, 1, or 2.")
    prefix = sequence[:frame]
    coding = sequence[frame:]
    n_complete = len(coding) // 3
    codons = [coding[index * 3 : index * 3 + 3] for index in range(n_complete)]
    suffix = coding[n_complete * 3 :]
    rng.shuffle(codons)
    return prefix + "".join(codons) + suffix


def synonymous_codon_shuffle(
    sequence: str,
    genetic_code: GeneticCode,
    rng: np.random.Generator,
    *,
    frame: int = 0,
) -> str:
    """Replace codons with synonymous codons while preserving translated protein sequence."""
    if frame not in {0, 1, 2}:
        raise ValueError("frame must be 0, 1, or 2.")
    prefix = sequence[:frame]
    coding = sequence[frame:]
    n_complete = len(coding) // 3
    shuffled: list[str] = []
    for index in range(n_complete):
        codon = coding[index * 3 : index * 3 + 3].upper()
        if len(codon) != 3 or any(base not in {"A", "C", "G", "T"} for base in codon):
            shuffled.append(codon)
            continue
        amino_acid = genetic_code.amino_acid(codon)
        synonyms = genetic_code.synonymous_codons(amino_acid)
        shuffled.append(str(rng.choice(synonyms)))
    suffix = coding[n_complete * 3 :]
    return prefix + "".join(shuffled) + suffix


def translated_sequence(sequence: str, genetic_code: GeneticCode, *, frame: int = 0) -> str:
    """Translate complete valid codons for null-model tests."""
    residues = []
    for index in range(frame, len(sequence) - 2, 3):
        codon = sequence[index : index + 3].upper()
        if any(base not in {"A", "C", "G", "T"} for base in codon):
            residues.append("X")
        else:
            residues.append(genetic_code.amino_acid(codon))
    return "".join(residues)


def label_permutation(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute labels for a supervised null model."""
    permuted = np.asarray(labels).copy()
    rng.shuffle(permuted)
    return permuted


def mononucleotide_counts(sequence: str) -> Counter[str]:
    """Return single-symbol counts."""
    return Counter(sequence)


def dinucleotide_counts(sequence: str) -> Counter[str]:
    """Return directed dinucleotide counts."""
    return Counter(left + right for left, right in zip(sequence[:-1], sequence[1:]))


def codon_counts(sequence: str, *, frame: int = 0) -> Counter[str]:
    """Return complete in-frame codon counts."""
    return Counter(sequence[index : index + 3] for index in range(frame, len(sequence) - 2, 3))


def permute_imaginary_axes(
    table: pd.DataFrame,
    permutation: tuple[int, ...],
    *,
    prefix: str = "e",
) -> pd.DataFrame:
    """Permute imaginary component columns e1...e7 without calling it an automorphism."""
    _validate_axis_permutation(permutation)
    transformed = table.copy()
    original = {axis: transformed[f"{prefix}{axis}"].copy() for axis in range(1, 8)}
    for target_axis, source_axis in enumerate(permutation, start=1):
        transformed[f"{prefix}{target_axis}"] = original[source_axis]
    return transformed


def sign_flip_imaginary_axes(
    table: pd.DataFrame,
    signs: tuple[int, ...],
    *,
    prefix: str = "e",
) -> pd.DataFrame:
    """Flip signs of imaginary component columns using supplied +/-1 values."""
    if len(signs) != 7 or any(sign not in {-1, 1} for sign in signs):
        raise ValueError("signs must contain seven values from {-1, +1}.")
    transformed = table.copy()
    for axis, sign in enumerate(signs, start=1):
        transformed[f"{prefix}{axis}"] = transformed[f"{prefix}{axis}"] * sign
    return transformed


def random_orthogonal_imaginary_transform(
    table: pd.DataFrame,
    rng: np.random.Generator,
    *,
    prefix: str = "e",
) -> pd.DataFrame:
    """Apply a random orthogonal transform to e1...e7 coordinates."""
    transformed = table.copy()
    columns = [f"{prefix}{axis}" for axis in range(1, 8)]
    q, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    values = transformed[columns].to_numpy(dtype=float)
    transformed.loc[:, columns] = values @ q
    return transformed


def random_fano_line_relabeling(
    feature_matrix: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Relabel Fano-line feature columns while preserving their values as a control."""
    transformed = feature_matrix.copy()
    line_tokens = [line_key(line) for line in FANO_LINES]
    shuffled = line_tokens.copy()
    rng.shuffle(shuffled)
    mapping = dict(zip(line_tokens, shuffled))
    rename: dict[str, str] = {}
    for column in transformed.columns:
        for source, target in mapping.items():
            source_token = source.strip("()").replace(",", "_")
            target_token = target.strip("()").replace(",", "_")
            if source_token in column:
                rename[column] = column.replace(source_token, target_token)
                break
    return transformed.rename(columns=rename)


def remove_scalar_component(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """Remove features explicitly tied to scalar component e0."""
    keep = [
        column
        for column in feature_matrix.columns
        if column == "sequence_id" or ("e0" not in column and "_p0" not in column)
    ]
    return feature_matrix[keep].copy()


def random_antisymmetric_tensor(
    rng: np.random.Generator,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Generate a random antisymmetric 7x7x7 interaction tensor."""
    tensor = rng.normal(0.0, scale, size=(7, 7, 7))
    for left in range(7):
        tensor[left, left, :] = 0.0
        for right in range(left + 1, 7):
            values = tensor[left, right, :].copy()
            tensor[right, left, :] = -values
    return tensor


def is_oriented_fano_automorphism(permutation_values: tuple[int, ...]) -> bool:
    """Return True only when a coordinate permutation preserves oriented Fano products."""
    _validate_axis_permutation(permutation_values)
    oriented_lines = set(FANO_LINES)
    for line in FANO_LINES:
        mapped = tuple(permutation_values[axis - 1] for axis in line)
        if mapped not in oriented_lines:
            return False
    return True


def enumerate_oriented_fano_automorphisms() -> list[tuple[int, ...]]:
    """Enumerate coordinate permutations preserving the project Fano-line orientation."""
    return [
        tuple(candidate)
        for candidate in permutations(range(1, 8))
        if is_oriented_fano_automorphism(tuple(candidate))
    ]


def line_key(line: tuple[int, int, int]) -> str:
    """Return stable Fano-line key."""
    return f"({line[0]},{line[1]},{line[2]})"


def _validate_axis_permutation(permutation_values: tuple[int, ...]) -> None:
    if sorted(permutation_values) != list(range(1, 8)):
        raise ValueError("permutation must contain each imaginary axis 1...7 exactly once.")
