"""Reusable octonion encoding schemes for sequences and downstream models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd

from fanoseq.codon_features import BASE_PROPERTIES
from fanoseq.genetic_code import GeneticCode, all_standard_codons
from fanoseq.octonion import Octonion

DNA_BASES = "ACGT"
BASE_TO_AXIS = {"A": 1, "C": 2, "G": 3, "T": 4}
PROTEIN_GROUPS: dict[str, tuple[int, str]] = {
    "A": (1, "small_flexible"),
    "G": (1, "small_flexible"),
    "S": (1, "small_flexible"),
    "V": (2, "hydrophobic"),
    "L": (2, "hydrophobic"),
    "I": (2, "hydrophobic"),
    "M": (2, "hydrophobic"),
    "F": (3, "aromatic"),
    "W": (3, "aromatic"),
    "Y": (3, "aromatic"),
    "T": (4, "polar_uncharged"),
    "N": (4, "polar_uncharged"),
    "Q": (4, "polar_uncharged"),
    "K": (5, "positive"),
    "R": (5, "positive"),
    "H": (5, "positive"),
    "D": (6, "negative"),
    "E": (6, "negative"),
    "C": (7, "sulfur_unique"),
    "P": (0, "kink_rigid"),
}


@dataclass(frozen=True)
class EncodingSpec:
    """Description for an exposed FanoSeq octonion encoding."""

    name: str
    domain: str
    representation: str
    description: str
    output_shape: str
    implemented: bool = True


ENCODING_REGISTRY: dict[str, EncodingSpec] = {
    "dna-base-context": EncodingSpec(
        name="dna-base-context",
        domain="dna",
        representation="eight-channel-tensor",
        description="8-channel base plus previous-base context encoding.",
        output_shape="one octonion per base",
    ),
    "gf8-base": EncodingSpec(
        name="gf8-base",
        domain="dna",
        representation="algebraic-octonion",
        description="Sparse base encoding using imaginary axes and binary chemistry classes.",
        output_shape="one octonion per base",
    ),
    "octonion-walk": EncodingSpec(
        name="octonion-walk",
        domain="dna",
        representation="algebraic-octonion",
        description="Order-sensitive left-associated octonion product over DNA k-mers.",
        output_shape="one octonion per k-mer",
    ),
    "codon-embedding-init": EncodingSpec(
        name="codon-embedding-init",
        domain="dna",
        representation="eight-channel-tensor",
        description="64x8 codon table initialized from root chemistry, wobble, and degeneracy.",
        output_shape="64 codons x 8 components",
    ),
    "protein-groups": EncodingSpec(
        name="protein-groups",
        domain="protein",
        representation="eight-channel-tensor",
        description="Residue-level 8-group physicochemical octonion encoding.",
        output_shape="one octonion per residue",
    ),
    "multi-track": EncodingSpec(
        name="multi-track",
        domain="genomics",
        representation="eight-channel-tensor",
        description="Assign up to eight synchronized numeric tracks to octonion components.",
        output_shape="one octonion per genomic bin",
    ),
}


def list_encoding_specs() -> pd.DataFrame:
    """Return the available encoding schemes as a table."""
    return pd.DataFrame([spec.__dict__ for spec in ENCODING_REGISTRY.values()])


def encode_dna_base_context(sequence: str, include_scalar_mask: bool = False) -> pd.DataFrame:
    """Encode DNA as current-base plus previous-base channels.

    By default, components e0...e3 hold current-base A/C/G/T indicators and
    components e4...e7 hold previous-base A/C/G/T indicators. The scalar e0 can
    instead be used as a validity mask by setting ``include_scalar_mask`` to True;
    that compact variant trades exact one-hot separation for an explicit mask.
    """
    cleaned = "".join(sequence.split()).upper()
    rows: list[dict[str, object]] = []
    previous = "N"
    for position, base in enumerate(cleaned):
        components = np.zeros(8, dtype=float)
        valid = base in DNA_BASES
        if include_scalar_mask:
            components[0] = 1.0 if valid else 0.0
            if valid:
                components[BASE_TO_AXIS[base]] = 1.0
            if previous in DNA_BASES:
                components[BASE_TO_AXIS[previous] + 3] += 1.0
        else:
            if valid:
                components[DNA_BASES.index(base)] = 1.0
            if previous in DNA_BASES:
                components[DNA_BASES.index(previous) + 4] = 1.0
        rows.append(
            {
                "position": position,
                "base": base,
                "previous_base": previous,
                "valid": valid,
                **_component_dict("e", components),
            }
        )
        previous = base
    return pd.DataFrame(rows)


def encode_gf8_base(base: str, confidence: float = 1.0) -> Octonion:
    """Encode one DNA base as a sparse GF(8)-inspired octonion.

    The base itself occupies one of e1...e4. Components e5, e6, and e7 encode
    purine/pyrimidine, strong/weak, and amino/keto chemistry signs.
    """
    upper = base.upper()
    components = np.zeros(8, dtype=float)
    if upper not in BASE_TO_AXIS:
        return Octonion(components)
    ry, sw, mk = BASE_PROPERTIES[upper]
    components[0] = float(confidence)
    components[BASE_TO_AXIS[upper]] = float(confidence)
    components[5] = ry
    components[6] = sw
    components[7] = mk
    return Octonion(components)


def encode_octonion_walk(kmer: str, normalize: bool = False) -> Octonion | None:
    """Return the left-associated octonion walk product for a DNA k-mer."""
    cleaned = "".join(kmer.split()).upper()
    if not cleaned or any(base not in BASE_TO_AXIS for base in cleaned):
        return None
    result = _axis_octonion(BASE_TO_AXIS[cleaned[0]])
    for base in cleaned[1:]:
        result = result * _axis_octonion(BASE_TO_AXIS[base])
    if normalize:
        norm = result.norm()
        if norm > 0:
            result = result / norm
    return result


def iter_octonion_walks(
    sequence: str, k: int, step: int = 1, normalize: bool = False
) -> Iterator[dict[str, object]]:
    """Yield order-sensitive octonion-walk rows for a DNA sequence."""
    if k <= 0:
        raise ValueError("k must be > 0.")
    if step <= 0:
        raise ValueError("step must be > 0.")
    cleaned = "".join(sequence.split()).upper()
    position = 0
    for start in range(0, max(len(cleaned) - k + 1, 0), step):
        kmer = cleaned[start : start + k]
        encoded = encode_octonion_walk(kmer, normalize=normalize)
        if encoded is None:
            continue
        yield {
            "position": position,
            "start": start,
            "end": start + k,
            "kmer": kmer,
            **_component_dict("e", encoded.components),
        }
        position += 1


def build_codon_embedding_initialization(genetic_code: GeneticCode) -> pd.DataFrame:
    """Build a 64x8 codon embedding initializer with interpretable axes."""
    rows: list[dict[str, object]] = []
    family_sizes = {
        codon: len(genetic_code.synonymous_codons(genetic_code.amino_acid(codon)))
        for codon in all_standard_codons()
    }
    max_family = max(family_sizes.values()) if family_sizes else 1

    for codon in all_standard_codons():
        aa = genetic_code.amino_acid(codon)
        root = codon[:2]
        root_props = np.array([BASE_PROPERTIES[base] for base in root], dtype=float).mean(axis=0)
        wobble_props = np.array(BASE_PROPERTIES[codon[2]], dtype=float)
        family_size = family_sizes[codon]
        degeneracy_scaled = (family_size / max_family) * 2.0 - 1.0
        components = np.array(
            [
                -1.0 if genetic_code.is_stop(codon) else 1.0,
                root_props[0],
                root_props[1],
                root_props[2],
                wobble_props[0],
                wobble_props[1],
                wobble_props[2],
                degeneracy_scaled,
            ],
            dtype=float,
        )
        row = {
            "codon": codon,
            "amino_acid": aa,
            "root": root,
            "is_start": genetic_code.is_start(codon),
            "is_stop": genetic_code.is_stop(codon),
            "synonymous_family_size": family_size,
            **_component_dict("e", components),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def encode_protein_group_residue(residue: str) -> Octonion:
    """Encode one amino-acid residue into an 8-group octonion."""
    components = np.zeros(8, dtype=float)
    upper = residue.upper()
    if upper not in PROTEIN_GROUPS:
        return Octonion(components)
    axis, _ = PROTEIN_GROUPS[upper]
    components[axis] = 1.0
    return Octonion(components)


def encode_protein_groups(sequence: str) -> pd.DataFrame:
    """Encode a protein sequence into residue-level physicochemical groups."""
    rows: list[dict[str, object]] = []
    for position, residue in enumerate("".join(sequence.split()).upper()):
        axis, label = PROTEIN_GROUPS.get(residue, (-1, "unknown"))
        encoded = encode_protein_group_residue(residue)
        rows.append(
            {
                "position": position,
                "residue": residue,
                "group_axis": axis,
                "group_label": label,
                **_component_dict("e", encoded.components),
            }
        )
    return pd.DataFrame(rows)


def encode_multi_track(rows: Iterable[Iterable[float]]) -> np.ndarray:
    """Pack up to eight synchronized numeric tracks into octonion component arrays."""
    values = np.asarray([list(row) for row in rows], dtype=float)
    if values.ndim != 2:
        raise ValueError("multi-track input must be a 2D table-like object.")
    if values.shape[1] > 8:
        raise ValueError("multi-track encoding accepts at most eight tracks.")
    encoded = np.zeros((values.shape[0], 8), dtype=float)
    encoded[:, : values.shape[1]] = values
    return encoded


def registry_function(name: str) -> Callable[..., object]:
    """Return the function that implements an encoding by name."""
    functions: dict[str, Callable[..., object]] = {
        "dna-base-context": encode_dna_base_context,
        "gf8-base": encode_gf8_base,
        "octonion-walk": encode_octonion_walk,
        "codon-embedding-init": build_codon_embedding_initialization,
        "protein-groups": encode_protein_groups,
        "multi-track": encode_multi_track,
    }
    try:
        return functions[name]
    except KeyError as exc:
        choices = ", ".join(sorted(functions))
        raise ValueError(f"Unknown encoding {name!r}. Available encodings: {choices}.") from exc


def _axis_octonion(axis: int) -> Octonion:
    components = np.zeros(8, dtype=float)
    components[axis] = 1.0
    return Octonion(components)


def _component_dict(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {f"{prefix}{index}": float(values[index]) for index in range(8)}
