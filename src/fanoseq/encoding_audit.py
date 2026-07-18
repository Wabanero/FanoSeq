"""Encoding audits for FanoSeq representations.

This module deliberately separates algebraic checks from biological claims.  It
reports what the current encodings preserve, discard, duplicate, or alter under
explicit transformations and controls.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from itertools import combinations, permutations, product
from math import sqrt
from pathlib import Path
from typing import Iterable, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from fanoseq.axis_schemes import AXIS_SCHEME_REGISTRY, AxisScheme, get_axis_scheme
from fanoseq.codon_features import (
    DNA_BASES,
    base_position_octonion,
    encode_codon,
)
from fanoseq.dna_features import (
    DNA_ALPHABET,
    encode_dna_window,
    reverse_complement,
)
from fanoseq.encodings import ENCODING_REGISTRY
from fanoseq.fano_plane import FANO_LINE_KEYS
from fanoseq.fasta import FastaRecord, read_fasta
from fanoseq.genetic_code import GeneticCode, all_standard_codons, get_genetic_code
from fanoseq.matrix_genetics import build_matrix_genetics_tables
from fanoseq.octonion import FANO_LINES, Octonion
from fanoseq.octonion_numba import (
    octonion_associator,
    octonion_commutator,
    octonion_multiply,
)
from fanoseq.plots import compose_plot_multipanel
from fanoseq.protein_features import encode_protein_window
from fanoseq.windows import iter_windows

AuditCheck = Literal[
    "contracts",
    "reverse-complement",
    "permutation",
    "collision",
    "mutation",
    "redundancy",
    "codon",
]

COMPONENT_COLUMNS = [f"e{index}" for index in range(8)]
RC_COMPONENT_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0])
T_RC_DNA_WINDOW_V1 = np.diag(RC_COMPONENT_SIGNS)
TRANSITION_BASE = {"A": "G", "G": "A", "C": "T", "T": "C"}
TRANSVERSION_BASES = {
    "A": ("C", "T"),
    "G": ("C", "T"),
    "C": ("A", "G"),
    "T": ("A", "G"),
}
FANO_CONVENTION_ID = "fanoseq-fano-lines-v1"
SCHEMA_VERSION = "0.8.0"


@dataclass(frozen=True)
class EncodingAuditConfig:
    """Configuration for the encoding audit command."""

    input_path: Path
    seq_type: Literal["dna", "protein"]
    axis_scheme_id: str
    checks: tuple[str, ...]
    codon_axis_scheme_id: str = "codon-product-v1"
    window_size: int = 10
    step: int = 1
    kmer_k: int = 2
    epsilon: float = 1e-9
    max_ambiguous_fraction: float = 0.0
    codon_table: str | int = "standard"
    output_format: Literal["tsv", "parquet", "bundle"] = "tsv"
    random_seed: int = 0
    tolerance: float = 1e-9
    permutation_samples: int = 16
    max_perturbations: int = 200
    normalize_codons: bool = False


def transform_octonion_rc(
    octonion: Octonion | Iterable[float] | NDArray[np.float64],
    scheme_id: str = "dna-window-v1",
) -> Octonion:
    """Apply the implemented reverse-complement component transform.

    The transform is currently defined only for ``dna-window-v1``:
    ``diag(1, -1, 1, -1, -1, -1, 1, 1)``.
    """
    if scheme_id != "dna-window-v1":
        raise ValueError(
            "Reverse-complement component transforms are currently implemented "
            "only for scheme_id='dna-window-v1'."
        )
    values = octonion.components if isinstance(octonion, Octonion) else np.asarray(list(octonion), dtype=float)
    if values.shape != (8,):
        raise ValueError("Expected an 8-component octonion.")
    return Octonion(values * RC_COMPONENT_SIGNS)


def build_encoding_audit_tables(config: EncodingAuditConfig) -> dict[str, pd.DataFrame]:
    """Build all requested encoding-audit tables for a FASTA input."""
    _validate_audit_config(config)
    records = read_fasta(config.input_path)
    genetic_code = get_genetic_code(config.codon_table)
    checks = _normalize_checks(config.checks)
    input_hash = _sha256_file(config.input_path)

    tables: dict[str, pd.DataFrame] = {}
    contracts = build_encoding_contracts()
    tables["encoding_contracts"] = _with_run_metadata(
        contracts,
        config,
        genetic_code,
        input_hash,
    )
    tables["encoding_audit_summary"] = build_encoding_audit_summary(
        config,
        records,
        genetic_code,
        input_hash,
    )

    if "reverse-complement" in checks and config.seq_type == "dna":
        rc_tables = build_reverse_complement_audit(records, config)
        tables.update(rc_tables)

    if "codon" in checks or "collision" in checks:
        codon_tables = build_codon_audit_tables(
            genetic_code,
            axis_scheme_id=config.codon_axis_scheme_id,
            normalize=config.normalize_codons,
            tolerance=config.tolerance,
        )
        tables.update(codon_tables)

    if "redundancy" in checks and config.seq_type == "dna":
        tables.update(build_feature_redundancy_tables(records, config))

    if "mutation" in checks and config.seq_type == "dna":
        tables["mutation_sensitivity"] = build_mutation_sensitivity(records, config, genetic_code)

    if "permutation" in checks:
        axis_tables = build_axis_control_tables(records, config)
        tables.update(axis_tables)

    return {
        name: _with_run_metadata(table, config, genetic_code, input_hash)
        if not _has_run_metadata(table)
        else table
        for name, table in tables.items()
    }


def build_encoding_contracts() -> pd.DataFrame:
    """Return formal contracts for implemented and registered representations."""
    rows: list[dict[str, object]] = []
    for scheme in AXIS_SCHEME_REGISTRY.values():
        rows.append(_axis_scheme_contract(scheme))

    for spec in ENCODING_REGISTRY.values():
        rows.append(
            {
                "representation_id": spec.name,
                "source": "encoding_registry",
                "input_domain": spec.domain,
                "output_dimension": spec.output_shape,
                "representation_kind": spec.representation,
                "scalar_axis_meaning": _registry_scalar_meaning(spec.name),
                "imaginary_axis_meanings": _registry_axis_meanings(spec.name),
                "normalization": _registry_normalization(spec.name),
                "missing_data_behavior": _registry_missing_policy(spec.name),
                "ambiguity_handling": _registry_ambiguity_policy(spec.name),
                "orientation_convention": _registry_orientation(spec.name),
                "association_convention": _registry_association(spec.name),
                "known_invariances": _registry_invariances(spec.name),
                "known_non_invariances": _registry_non_invariances(spec.name),
                "information_lost": _registry_information_lost(spec.name),
                "recommended_baselines": _registry_baselines(spec.name),
                "status": "implemented" if spec.implemented else "registered",
                "representation_note": _representation_note(spec.representation),
            }
        )
    return pd.DataFrame(rows)


def build_encoding_audit_summary(
    config: EncodingAuditConfig,
    records: list[FastaRecord],
    genetic_code: GeneticCode,
    input_hash: str,
) -> pd.DataFrame:
    """Return high-level audit metadata and mathematical findings."""
    scheme = get_axis_scheme(config.axis_scheme_id)
    codon_tables = build_codon_audit_tables(
        genetic_code,
        normalize=config.normalize_codons,
        tolerance=config.tolerance,
    )
    catalog = codon_tables["codon_octonion_catalog"]
    components = catalog[COMPONENT_COLUMNS].to_numpy(dtype=float)
    singular_values = _safe_singular_values(components - components.mean(axis=0))
    rank = _safe_matrix_rank(components, tol=config.tolerance)
    collisions = codon_tables["codon_collision_report"]
    rc_statement = (
        "dna-window-v1 reverse complement maps components by "
        "diag(1,-1,1,-1,-1,-1,1,1); exhaustive finite-sequence checks are in "
        "reverse_complement_audit."
        if config.seq_type == "dna"
        else "Not applicable to protein input."
    )
    rows = [
        {
            "summary_item": "input",
            "value": str(config.input_path),
            "detail": f"{len(records)} FASTA record(s); sha256={input_hash}",
        },
        {
            "summary_item": "axis_scheme",
            "value": scheme.scheme_id,
            "detail": (
                f"{scheme.representation}; Fano convention={FANO_CONVENTION_ID}; "
                f"implemented={scheme.implemented}"
            ),
        },
        {
            "summary_item": "reverse_complement_transform",
            "value": "available" if config.seq_type == "dna" else "not_applicable",
            "detail": rc_statement,
        },
        {
            "summary_item": "codon_product_injectivity",
            "value": "injective" if collisions.empty else "non_injective",
            "detail": (
                f"64 codons, component_rank={rank}, "
                f"nonzero_singular_values={(singular_values > config.tolerance).sum()}"
            ),
        },
        {
            "summary_item": "commutator_interpretation",
            "value": "antisymmetric_interaction_expansion",
            "detail": (
                "For pure imaginary parts, [x,y]_imag = 2 times the Fano 7D cross "
                "product. Fano-line attributions are structured exterior-product terms."
            ),
        },
        {
            "summary_item": "axis_assignment_interpretation",
            "value": "axis_dependent",
            "detail": (
                "Changing biological axis labels differs from changing coordinates or "
                "the multiplication table. Non-automorphism permutations can alter "
                "line shares and rankings."
            ),
        },
        {
            "summary_item": "genetic_code",
            "value": genetic_code.name,
            "detail": f"codon_table={config.codon_table}",
        },
    ]
    return pd.DataFrame(rows)


def build_reverse_complement_audit(
    records: list[FastaRecord],
    config: EncodingAuditConfig,
    exhaustive_max_length: int = 6,
) -> dict[str, pd.DataFrame]:
    """Derive and test the dna-window-v1 reverse-complement transform."""
    if config.axis_scheme_id != "dna-window-v1":
        raise ValueError("Reverse-complement audit currently targets dna-window-v1.")

    rows: list[dict[str, object]] = []
    derivation_rows = _reverse_complement_derivation_rows(config)
    transform = pd.DataFrame(
        T_RC_DNA_WINDOW_V1,
        columns=COMPONENT_COLUMNS,
    )
    transform.insert(0, "output_component", COMPONENT_COLUMNS)
    transform.insert(0, "scheme_id", config.axis_scheme_id)

    for length in range(0, exhaustive_max_length + 1):
        for bases in product(DNA_ALPHABET, repeat=length):
            sequence = "".join(bases)
            rows.append(
                _reverse_complement_row(
                    source="exhaustive",
                    sequence_id=f"len{length}",
                    position=0,
                    start=0,
                    end=length,
                    window=sequence,
                    config=config,
                )
            )

    for record in records:
        for window in iter_windows(record.sequence, config.window_size, config.step):
            rows.append(
                _reverse_complement_row(
                    source="input_windows",
                    sequence_id=record.id,
                    position=window.position,
                    start=window.start,
                    end=window.end,
                    window=window.sequence,
                    config=config,
                )
            )
        rows.extend(_finite_window_reverse_complement_rows(record, config))

    audit = pd.DataFrame(rows)
    return {
        "reverse_complement_transform_matrix": transform,
        "reverse_complement_derivation": pd.DataFrame(derivation_rows),
        "reverse_complement_audit": audit,
    }


def build_codon_audit_tables(
    genetic_code: GeneticCode,
    axis_scheme_id: str = "codon-product-v1",
    normalize: bool = False,
    tolerance: float = 1e-9,
) -> dict[str, pd.DataFrame]:
    """Audit all 64 standard DNA codon octonions."""
    rows: list[dict[str, object]] = []
    for codon in all_standard_codons():
        encoded = encode_codon(
            codon,
            genetic_code,
            max_ambiguous_fraction=0.0,
            include_stop_codons=True,
            normalize=normalize,
        )
        assert encoded is not None
        left = encoded.octonion
        b1 = base_position_octonion(codon[0], 1)
        b2 = base_position_octonion(codon[1], 2)
        b3 = base_position_octonion(codon[2], 3)
        right = b1 * (b2 * b3)
        associator = (b1 * b2) * b3 - b1 * (b2 * b3)
        aa = genetic_code.amino_acid(codon)
        row: dict[str, object] = {
            "scheme_id": axis_scheme_id,
            "codon": codon,
            "amino_acid": aa,
            "is_start": genetic_code.is_start(codon),
            "is_stop": genetic_code.is_stop(codon),
            "synonymous_family_size": len(genetic_code.synonymous_codons(aa)),
            "norm": left.norm(),
            "right_associated_norm": right.norm(),
            "left_right_product_distance": float(np.linalg.norm(left.components - right.components)),
            "associator_norm": associator.norm(),
        }
        row.update({f"e{i}": float(left.components[i]) for i in range(8)})
        row.update({f"right_e{i}": float(right.components[i]) for i in range(8)})
        row.update({f"associator_e{i}": float(associator.components[i]) for i in range(8)})
        row.update(encoded.metadata)
        rows.append(row)

    catalog = pd.DataFrame(rows)
    collisions = _codon_collisions(catalog, tolerance)
    distance_matrix = _codon_distance_matrix(catalog)
    synonymy = _codon_synonymy_statistics(catalog, distance_matrix)
    substitution_effects = _codon_substitution_effects(catalog, genetic_code)
    rank_spectrum = _codon_rank_spectrum(catalog, tolerance)
    return {
        "codon_octonion_catalog": catalog,
        "codon_collision_report": collisions,
        "codon_distance_matrix": distance_matrix,
        "codon_synonymy_statistics": synonymy,
        "codon_substitution_effects": substitution_effects,
        "codon_geometry_rank_spectrum": rank_spectrum,
    }


def build_feature_redundancy_tables(
    records: list[FastaRecord],
    config: EncodingAuditConfig,
) -> dict[str, pd.DataFrame]:
    """Compare octonion-derived features with ordinary interaction controls."""
    sequences = _window_component_sequences(records, config)
    matrices = _feature_family_matrices(sequences)
    rows: list[dict[str, object]] = []
    spectrum_rows: list[dict[str, object]] = []
    target_by_family = _feature_family_targets(sequences, matrices)

    for family, matrix in matrices.items():
        target = target_by_family[family]
        stats = _matrix_redundancy_stats(
            family,
            matrix,
            target,
            tolerance=config.tolerance,
            random_seed=config.random_seed,
        )
        rows.append(stats)
        singular_values = cast(Iterable[float], stats.pop("_singular_values"))
        for rank_index, singular_value in enumerate(singular_values):
            spectrum_rows.append(
                {
                    "feature_family": family,
                    "singular_value_index": rank_index,
                    "singular_value": float(singular_value),
                    "energy": float(singular_value * singular_value),
                }
            )

    cross_residual = _commutator_cross_product_residual(sequences)
    rows.append(
        {
            "feature_family": "commutator_vs_2x_fano_cross_product",
            "n_rows": int(cross_residual["n_rows"]),
            "n_features": 7,
            "numerical_rank": 7 if cross_residual["n_rows"] else 0,
            "condition_number": np.nan,
            "variance_explained_pc1": np.nan,
            "mean_abs_feature_correlation": np.nan,
            "linear_r2_predict_components": np.nan,
            "tree_r2_predict_components": np.nan,
            "mutual_information_with_components": np.nan,
            "collision_rate": np.nan,
            "recoverability": "identity_check",
            "max_abs_identity_residual": cross_residual["max_abs_residual"],
            "interpretation": (
                "For pure imaginary components, the commutator is exactly twice "
                "the Fano 7D cross product up to numerical tolerance."
            ),
        }
    )
    rows.append(
        {
            "feature_family": "fano_line_attribution_vs_exterior_terms",
            "n_rows": int(cross_residual["n_rows"]),
            "n_features": len(FANO_LINES),
            "numerical_rank": np.nan,
            "condition_number": np.nan,
            "variance_explained_pc1": np.nan,
            "mean_abs_feature_correlation": np.nan,
            "linear_r2_predict_components": np.nan,
            "tree_r2_predict_components": np.nan,
            "mutual_information_with_components": np.nan,
            "collision_rate": np.nan,
            "recoverability": "structured_reexpression",
            "max_abs_identity_residual": 0.0,
            "interpretation": (
                "Each Fano-line contribution uses antisymmetric pair terms "
                "x_i*y_j - x_j*y_i from the ordinary exterior product."
            ),
        }
    )

    return {
        "feature_redundancy": pd.DataFrame(rows),
        "feature_rank_spectrum": pd.DataFrame(spectrum_rows),
    }


def build_mutation_sensitivity(
    records: list[FastaRecord],
    config: EncodingAuditConfig,
    genetic_code: GeneticCode,
) -> pd.DataFrame:
    """Build deterministic perturbation sensitivity rows."""
    rows: list[dict[str, object]] = []
    for record in records:
        original = _sequence_signature(record.sequence, config, genetic_code)
        perturbations = []
        perturbations.extend(
            dna_perturbations(
                record.sequence,
                random_seed=config.random_seed,
                max_perturbations=config.max_perturbations,
            )
        )
        perturbations.extend(
            coding_perturbations(
                record.sequence,
                genetic_code,
                random_seed=config.random_seed,
                max_perturbations=config.max_perturbations,
            )
        )
        for perturbation in perturbations[: config.max_perturbations]:
            changed = _sequence_signature(str(perturbation["sequence"]), config, genetic_code)
            row = {
                "sequence_id": record.id,
                "perturbation_id": perturbation["perturbation_id"],
                "perturbation_type": perturbation["perturbation_type"],
                "level": perturbation["level"],
                "position": perturbation.get("position", -1),
                "codon_index": perturbation.get("codon_index", -1),
                "original_symbol": perturbation.get("original_symbol", "NA"),
                "replacement_symbol": perturbation.get("replacement_symbol", "NA"),
                "position_fraction": _position_fraction(
                    int(str(perturbation.get("position", -1))),
                    len("".join(record.sequence.split())),
                ),
                "raw_component_delta_norm": float(
                    np.linalg.norm(changed["component_mean"] - original["component_mean"])
                ),
                "octonion_norm_delta": float(changed["mean_octonion_norm"] - original["mean_octonion_norm"]),
                "product_norm_delta": float(changed["mean_product_norm"] - original["mean_product_norm"]),
                "commutator_delta": float(changed["mean_commutator_norm"] - original["mean_commutator_norm"]),
                "associator_delta": float(changed["mean_associator_norm"] - original["mean_associator_norm"]),
                "transition_score_delta": float(changed["mean_transition_score"] - original["mean_transition_score"]),
                "fano_line_profile_delta_norm": float(
                    np.linalg.norm(changed["fano_profile"] - original["fano_profile"])
                ),
                "sequence_fingerprint_delta_norm": float(
                    np.linalg.norm(changed["fingerprint"] - original["fingerprint"])
                ),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def dna_perturbations(
    sequence: str,
    *,
    random_seed: int = 0,
    max_perturbations: int | None = None,
) -> list[dict[str, object]]:
    """Return deterministic DNA-level perturbations."""
    cleaned = "".join(sequence.split()).upper()
    rows: list[dict[str, object]] = []
    count = 0

    def add(row: dict[str, object]) -> None:
        nonlocal count
        if max_perturbations is not None and count >= max_perturbations:
            return
        row["perturbation_id"] = f"dna_{count:05d}"
        row["level"] = "dna"
        rows.append(row)
        count += 1

    for index, base in enumerate(cleaned):
        if base not in DNA_BASES:
            continue
        for replacement in DNA_ALPHABET:
            if replacement != base:
                add(
                    {
                        "perturbation_type": "single_nucleotide_substitution",
                        "position": index,
                        "original_symbol": base,
                        "replacement_symbol": replacement,
                        "sequence": cleaned[:index] + replacement + cleaned[index + 1 :],
                    }
                )
        add(
            {
                "perturbation_type": "transition_substitution",
                "position": index,
                "original_symbol": base,
                "replacement_symbol": TRANSITION_BASE[base],
                "sequence": cleaned[:index] + TRANSITION_BASE[base] + cleaned[index + 1 :],
            }
        )
        for replacement in TRANSVERSION_BASES[base]:
            add(
                {
                    "perturbation_type": "transversion_substitution",
                    "position": index,
                    "original_symbol": base,
                    "replacement_symbol": replacement,
                    "sequence": cleaned[:index] + replacement + cleaned[index + 1 :],
                }
            )
        inserted = "A" if base != "A" else "C"
        add(
            {
                "perturbation_type": "insertion",
                "position": index,
                "original_symbol": "",
                "replacement_symbol": inserted,
                "sequence": cleaned[:index] + inserted + cleaned[index:],
            }
        )
        add(
            {
                "perturbation_type": "deletion",
                "position": index,
                "original_symbol": base,
                "replacement_symbol": "",
                "sequence": cleaned[:index] + cleaned[index + 1 :],
            }
        )

    add({"perturbation_type": "reverse", "position": -1, "sequence": cleaned[::-1]})
    complement = cleaned.translate(str.maketrans("ACGT", "TGCA"))
    add({"perturbation_type": "complement", "position": -1, "sequence": complement})
    add({"perturbation_type": "reverse_complement", "position": -1, "sequence": reverse_complement(cleaned)})
    add(
        {
            "perturbation_type": "mononucleotide_shuffle",
            "position": -1,
            "sequence": _mononucleotide_shuffle(cleaned, random_seed),
        }
    )
    add(
        {
            "perturbation_type": "dinucleotide_preserving_shuffle",
            "position": -1,
            "sequence": dinucleotide_preserving_shuffle(cleaned, random_seed),
        }
    )
    return rows


def coding_perturbations(
    sequence: str,
    genetic_code: GeneticCode,
    *,
    random_seed: int = 0,
    max_perturbations: int | None = None,
) -> list[dict[str, object]]:
    """Return deterministic coding-sequence perturbations."""
    cleaned = "".join(sequence.split()).upper()
    codons = [
        cleaned[index : index + 3]
        for index in range(0, len(cleaned) - 2, 3)
        if all(base in DNA_BASES for base in cleaned[index : index + 3])
    ]
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(random_seed)
    count = 0

    def add(row: dict[str, object]) -> None:
        nonlocal count
        if max_perturbations is not None and count >= max_perturbations:
            return
        row["perturbation_id"] = f"coding_{count:05d}"
        row["level"] = "coding"
        rows.append(row)
        count += 1

    for codon_index, codon in enumerate(codons):
        aa = genetic_code.amino_acid(codon)
        synonyms = [candidate for candidate in genetic_code.synonymous_codons(aa) if candidate != codon]
        start = codon_index * 3
        if synonyms:
            replacement = synonyms[0]
            add(
                {
                    "perturbation_type": "synonymous_substitution",
                    "position": start,
                    "codon_index": codon_index,
                    "original_symbol": codon,
                    "replacement_symbol": replacement,
                    "sequence": cleaned[:start] + replacement + cleaned[start + 3 :],
                }
            )
        conservative = _replacement_codon_for_aa_class(aa, genetic_code, same_class=True)
        if conservative is not None:
            add(
                {
                    "perturbation_type": "conservative_amino_acid_substitution",
                    "position": start,
                    "codon_index": codon_index,
                    "original_symbol": codon,
                    "replacement_symbol": conservative,
                    "sequence": cleaned[:start] + conservative + cleaned[start + 3 :],
                }
            )
        radical = _replacement_codon_for_aa_class(aa, genetic_code, same_class=False)
        if radical is not None:
            add(
                {
                    "perturbation_type": "radical_amino_acid_substitution",
                    "position": start,
                    "codon_index": codon_index,
                    "original_symbol": codon,
                    "replacement_symbol": radical,
                    "sequence": cleaned[:start] + radical + cleaned[start + 3 :],
                }
            )

    add({"perturbation_type": "frameshift", "position": 1, "sequence": cleaned[:1] + "A" + cleaned[1:]})
    for codon_index, codon in enumerate(codons):
        if not genetic_code.is_stop(codon):
            start = codon_index * 3
            add(
                {
                    "perturbation_type": "premature_stop",
                    "position": start,
                    "codon_index": codon_index,
                    "original_symbol": codon,
                    "replacement_symbol": "TAA",
                    "sequence": cleaned[:start] + "TAA" + cleaned[start + 3 :],
                }
            )
            break

    if codons:
        shuffled = list(codons)
        rng.shuffle(shuffled)
        add({"perturbation_type": "codon_order_shuffle", "position": -1, "sequence": "".join(shuffled)})
        recoded = synonymous_recoding(cleaned, genetic_code)
        add(
            {
                "perturbation_type": "synonymous_recoding_preserving_protein",
                "position": -1,
                "sequence": recoded,
            }
        )
    return rows


def dinucleotide_preserving_shuffle(sequence: str, random_seed: int = 0) -> str:
    """Shuffle a DNA sequence while preserving directed dinucleotide counts."""
    cleaned = "".join(base for base in sequence.upper() if base in DNA_BASES)
    if len(cleaned) < 3:
        return cleaned
    adjacency: dict[str, list[str]] = {base: [] for base in DNA_ALPHABET}
    for left, right in zip(cleaned, cleaned[1:], strict=False):
        adjacency[left].append(right)
    rng = np.random.default_rng(random_seed)
    for values in adjacency.values():
        rng.shuffle(values)

    local = {base: list(values) for base, values in adjacency.items()}
    stack = [cleaned[0]]
    path: list[str] = []
    while stack:
        node = stack[-1]
        if local[node]:
            stack.append(local[node].pop())
        else:
            path.append(stack.pop())
    shuffled = "".join(reversed(path))
    if len(shuffled) != len(cleaned):
        return cleaned
    if _dinucleotide_counts(shuffled) != _dinucleotide_counts(cleaned):
        return cleaned
    return shuffled


def synonymous_recoding(sequence: str, genetic_code: GeneticCode) -> str:
    """Return a deterministic synonymous recoding that preserves translation."""
    cleaned = "".join(sequence.split()).upper()
    output: list[str] = []
    for start in range(0, len(cleaned), 3):
        codon = cleaned[start : start + 3]
        if len(codon) != 3 or any(base not in DNA_BASES for base in codon):
            output.append(codon)
            continue
        aa = genetic_code.amino_acid(codon)
        synonyms = genetic_code.synonymous_codons(aa)
        if len(synonyms) <= 1:
            output.append(codon)
        else:
            output.append(next(candidate for candidate in synonyms if candidate != codon))
    return "".join(output)


def translate_dna(sequence: str, genetic_code: GeneticCode) -> str:
    """Translate complete unambiguous codons with the supplied genetic code."""
    cleaned = "".join(sequence.split()).upper()
    amino_acids = []
    for start in range(0, len(cleaned) - 2, 3):
        codon = cleaned[start : start + 3]
        if all(base in DNA_BASES for base in codon):
            amino_acids.append(genetic_code.amino_acid(codon))
        else:
            amino_acids.append("X")
    return "".join(amino_acids)


def build_axis_control_tables(
    records: list[FastaRecord],
    config: EncodingAuditConfig,
) -> dict[str, pd.DataFrame]:
    """Compare biological label, coordinate, automorphism, and tensor controls."""
    components = _component_matrix_for_axis_audit(records, config)
    controls = _axis_controls(config)
    stability_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    canonical_profile = _fano_line_share_profile(components)
    canonical_transition = _adjacent_transition_scores(components)
    canonical_distances = _row_distance_matrix(components)

    for control in controls:
        transformed = _apply_axis_control(components, control)
        profile = _fano_line_share_profile(transformed)
        transition = _adjacent_transition_scores(transformed)
        distances = _row_distance_matrix(transformed)
        stability_rows.append(
            {
                "control_id": control["control_id"],
                "transformation_category": control["transformation_category"],
                "dominant_fano_line": FANO_LINE_KEYS[int(np.argmax(profile))] if profile.sum() else "NA",
                "dominant_fano_line_share": float(profile.max()) if profile.sum() else 0.0,
                "line_share_cosine_to_canonical": _cosine(canonical_profile, profile),
                "transition_ranking_spearman": _spearman(canonical_transition, transition),
                "sequence_distance_spearman": _spearman(
                    _upper_triangle(canonical_distances),
                    _upper_triangle(distances),
                ),
                "nearest_neighbor_overlap": _nearest_neighbor_overlap(canonical_distances, distances),
                "axis_dependent_interpretation": bool(
                    control["transformation_category"]
                    not in {"canonical", "biological_label_permutation"}
                ),
            }
        )
        control_rows.append(
            {
                "control_id": control["control_id"],
                "transformation_category": control["transformation_category"],
                "permutation": ",".join(
                    str(axis) for axis in cast(Iterable[object], control["permutation"])
                ),
                "signs": ",".join(
                    str(int(str(sign)))
                    for sign in cast(Iterable[object], control["signs"])
                ),
                "changes_biological_labels": control["changes_biological_labels"],
                "changes_coordinates": control["changes_coordinates"],
                "changes_multiplication_table": control["changes_multiplication_table"],
                "is_fano_plane_automorphism": control["is_fano_plane_automorphism"],
                "preserves_multiplication_table": control["preserves_multiplication_table"],
                "notes": control["notes"],
            }
        )
    return {
        "axis_permutation_stability": pd.DataFrame(stability_rows),
        "fano_automorphism_controls": pd.DataFrame(control_rows),
    }


def plot_encoding_audit_outputs(tables: dict[str, pd.DataFrame], output_dir: str | Path) -> list[Path]:
    """Create lightweight PNG visualizations for available audit tables."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if "codon_distance_matrix" in tables and not tables["codon_distance_matrix"].empty:
        paths.append(_plot_codon_distance_heatmap(tables["codon_distance_matrix"], base / "codon_distance_heatmap.png"))
    if "codon_octonion_catalog" in tables and not tables["codon_octonion_catalog"].empty:
        paths.append(_plot_codon_pca(tables["codon_octonion_catalog"], base / "codon_pca.png"))
        paths.append(_plot_synonymous_geometry(tables["codon_distance_matrix"], base / "synonymous_family_geometry.png"))
    if "mutation_sensitivity" in tables and not tables["mutation_sensitivity"].empty:
        paths.append(_plot_mutation_sensitivity(tables["mutation_sensitivity"], base / "mutation_sensitivity_by_position.png"))
    if "feature_rank_spectrum" in tables and not tables["feature_rank_spectrum"].empty:
        paths.append(_plot_singular_values(tables["feature_rank_spectrum"], base / "singular_value_spectrum.png"))
    if "reverse_complement_audit" in tables and not tables["reverse_complement_audit"].empty:
        paths.append(_plot_reverse_complement_residuals(tables["reverse_complement_audit"], base / "reverse_complement_residuals.png"))
    if "axis_permutation_stability" in tables and not tables["axis_permutation_stability"].empty:
        paths.append(_plot_axis_stability(tables["axis_permutation_stability"], base / "axis_permutation_stability.png"))
        paths.append(_plot_fano_profile_stability(tables["axis_permutation_stability"], base / "fano_line_profile_stability.png"))
    if paths:
        paths.append(
            compose_plot_multipanel(
                paths,
                base / "encoding_audit_multipanel.png",
                "FanoSeq encoding audit multipanel",
            )
        )
    return paths


def _validate_audit_config(config: EncodingAuditConfig) -> None:
    if config.seq_type not in {"dna", "protein"}:
        raise ValueError("--seq-type must be either 'dna' or 'protein'.")
    if config.window_size <= 0:
        raise ValueError("--window-size must be > 0.")
    if config.step <= 0:
        raise ValueError("--step must be > 0.")
    if config.kmer_k <= 0:
        raise ValueError("--kmer-k must be > 0.")
    if config.tolerance <= 0:
        raise ValueError("--tolerance must be > 0.")
    get_axis_scheme(config.axis_scheme_id)


def _normalize_checks(checks: tuple[str, ...]) -> set[str]:
    normalized: set[str] = set()
    for check in checks:
        for piece in str(check).split(","):
            value = piece.strip().lower()
            if not value:
                continue
            if value == "all":
                normalized.update(
                    {
                        "contracts",
                        "reverse-complement",
                        "permutation",
                        "collision",
                        "mutation",
                        "redundancy",
                        "codon",
                    }
                )
            else:
                normalized.add(value)
    if not normalized:
        normalized = {"contracts"}
    return normalized


def _with_run_metadata(
    table: pd.DataFrame,
    config: EncodingAuditConfig,
    genetic_code: GeneticCode,
    input_hash: str,
) -> pd.DataFrame:
    from fanoseq import __version__

    result = table.copy()
    metadata = [
        ("software_version", __version__),
        ("schema_version", SCHEMA_VERSION),
        ("scheme_id", config.axis_scheme_id),
        ("fano_convention_id", FANO_CONVENTION_ID),
        ("genetic_code_table", genetic_code.name),
        ("normalization_settings", f"codon_normalize={config.normalize_codons}"),
        ("random_seed", config.random_seed),
        ("input_hash", input_hash),
        ("tolerance", f"{config.tolerance:g}"),
    ]
    for insert_at, (column, value) in enumerate(metadata):
        output_column = column if column not in result.columns else f"audit_{column}"
        result.insert(insert_at, output_column, value)
    return result


def _has_run_metadata(table: pd.DataFrame) -> bool:
    return {"software_version", "schema_version", "input_hash"}.issubset(table.columns)


def _axis_scheme_contract(scheme: AxisScheme) -> dict[str, object]:
    return {
        "representation_id": scheme.scheme_id,
        "source": "axis_scheme_registry",
        "input_domain": f"{scheme.seq_type}:{scheme.mode}",
        "output_dimension": "8 components per encoded unit",
        "representation_kind": scheme.representation,
        "scalar_axis_meaning": scheme.scalar_axis.description,
        "imaginary_axis_meanings": "; ".join(
            f"{axis.symbol}={axis.label}" for axis in scheme.imaginary_axes
        ),
        "normalization": "; ".join(axis.normalization for axis in scheme.axes),
        "missing_data_behavior": "; ".join(axis.missing_policy for axis in scheme.axes),
        "ambiguity_handling": _scheme_ambiguity_policy(scheme),
        "orientation_convention": (
            FANO_CONVENTION_ID if scheme.representation == "algebraic-octonion" else "none"
        ),
        "association_convention": _scheme_association(scheme),
        "known_invariances": _scheme_invariances(scheme.scheme_id),
        "known_non_invariances": _scheme_non_invariances(scheme.scheme_id),
        "information_lost": _scheme_information_lost(scheme.scheme_id),
        "recommended_baselines": "; ".join(axis.benchmark_baseline for axis in scheme.axes),
        "status": scheme.status,
        "representation_note": _representation_note(scheme.representation),
    }


def _representation_note(representation: str) -> str:
    if representation == "eight-channel-tensor":
        return (
            "Eight-channel tensor only. Do not interpret as an algebraic octonion "
            "unless multiplication is explicitly used and justified."
        )
    return (
        "Algebraic octonion under the recorded Fano convention. Biological meaning "
        "comes from the axis assignment, not from the algebra alone."
    )


def _scheme_ambiguity_policy(scheme: AxisScheme) -> str:
    if scheme.scheme_id in {"dna-window-v1", "protein-sequence-v1"}:
        return "Units are skipped when ambiguous_fraction exceeds the configured threshold."
    if scheme.scheme_id == "codon-product-v1":
        return "Ambiguous codons may be skipped; retained ambiguous bases have zero chemistry."
    return "Defined by the registered missing-data policy; not runnable unless implemented."


def _scheme_association(scheme: AxisScheme) -> str:
    if scheme.scheme_id == "codon-product-v1":
        return "left-associated ordered product (B1*B2)*B3; right association audited separately"
    if scheme.representation == "algebraic-octonion":
        return "adjacent products are binary; triplets use explicit associator (xy)z - x(yz)"
    return "not applicable"


def _scheme_invariances(scheme_id: str) -> str:
    if scheme_id == "dna-window-v1":
        return (
            "Component signs under reverse complement are fixed for unambiguous windows; "
            "k-mer entropy and RC-similarity are invariant."
        )
    if scheme_id == "codon-product-v1":
        return "Deterministic for a fixed genetic code and normalization setting."
    return "No invariance claim beyond deterministic feature construction."


def _scheme_non_invariances(scheme_id: str) -> str:
    if scheme_id == "dna-window-v1":
        return "Not invariant to arbitrary axis reassignment, window phase, insertion, or deletion."
    if scheme_id == "codon-product-v1":
        return "Not invariant to codon position permutation or association convention."
    return "Not audited as runnable."


def _scheme_information_lost(scheme_id: str) -> str:
    if scheme_id == "dna-window-v1":
        return "Exact base order is lost except through k-mer entropy and RC-similarity summaries."
    if scheme_id == "protein-sequence-v1":
        return "Exact residue order and residue identities are compressed into window summaries."
    if scheme_id == "codon-product-v1":
        return "The codon can be injectively represented, but biological annotation is external."
    return "Not quantified for non-runnable schemes."


def _registry_scalar_meaning(name: str) -> str:
    values = {
        "dna-base-context": "Current-base A channel unless scalar-mask option is enabled.",
        "gf8-base": "Base confidence/validity scalar.",
        "octonion-walk": "Scalar result of left-associated basis products.",
        "codon-embedding-init": "Sense/stop sign, not an octonion scalar mechanism.",
        "protein-groups": "One residue group channel; P uses e0 in this tensor encoding.",
        "multi-track": "First supplied external track.",
    }
    return values.get(name, "Registered encoding scalar component.")


def _registry_axis_meanings(name: str) -> str:
    values = {
        "dna-base-context": "Current and previous base one-hot/context channels.",
        "gf8-base": "Base identity axis plus RY/SW/MK chemistry signs.",
        "octonion-walk": "Basis axes reached by ordered base-axis products.",
        "codon-embedding-init": "Root chemistry, wobble chemistry, and degeneracy tensor channels.",
        "protein-groups": "Residue physicochemical group one-hot channels.",
        "multi-track": "User-supplied synchronized tracks.",
    }
    return values.get(name, "Registered eight-channel meanings.")


def _registry_normalization(name: str) -> str:
    return "Optional unit-norm normalization where the encoder exposes it." if name == "octonion-walk" else "raw deterministic components"


def _registry_missing_policy(name: str) -> str:
    if name in {"gf8-base", "protein-groups"}:
        return "Unknown symbols map to the zero vector."
    if name == "octonion-walk":
        return "Invalid k-mers are skipped."
    return "Defined by the encoder input contract."


def _registry_ambiguity_policy(name: str) -> str:
    return _registry_missing_policy(name)


def _registry_orientation(name: str) -> str:
    return FANO_CONVENTION_ID if name in {"gf8-base", "octonion-walk"} else "not algebraic"


def _registry_association(name: str) -> str:
    return "left-associated product over the k-mer" if name == "octonion-walk" else "not applicable"


def _registry_invariances(name: str) -> str:
    if name == "octonion-walk":
        return "Deterministic for fixed k-mer and Fano convention."
    return "No algebraic invariance claim."


def _registry_non_invariances(name: str) -> str:
    if name == "octonion-walk":
        return "Order-sensitive; not invariant to reversal, complement, or reassociation."
    return "Axis and channel interpretations are assignment-dependent."


def _registry_information_lost(name: str) -> str:
    if name == "dna-base-context":
        return "Longer-range sequence context beyond previous base is lost."
    if name == "octonion-walk":
        return "Many k-mers can collide because repeated basis products compress sequences."
    return "Task-dependent compression; audit against raw descriptors recommended."


def _registry_baselines(name: str) -> str:
    values = {
        "dna-base-context": "one-hot DNA, k-mer counts",
        "gf8-base": "one-hot DNA, RY/SW/MK chemistry labels",
        "octonion-walk": "k-mer identity, hashed k-mers, FCGR",
        "codon-embedding-init": "codon one-hot, RSCU, CAI, learned codon embeddings",
        "protein-groups": "amino-acid one-hot, physicochemical scales",
        "multi-track": "raw track matrix, PCA of tracks",
    }
    return values.get(name, "raw descriptors")


def _reverse_complement_derivation_rows(config: EncodingAuditConfig) -> list[dict[str, object]]:
    return [
        {
            "scheme_id": config.axis_scheme_id,
            "component": "e0",
            "formula": "valid_fraction",
            "rc_formula": "valid_fraction",
            "derived_sign": 1,
            "reason": "Reverse complement preserves the count of valid bases and window length.",
        },
        {
            "scheme_id": config.axis_scheme_id,
            "component": "e1",
            "formula": "(A+G-C-T)/L",
            "rc_formula": "(T+C-G-A)/L",
            "derived_sign": -1,
            "reason": "Complement swaps A/T and C/G.",
        },
        {
            "scheme_id": config.axis_scheme_id,
            "component": "e2",
            "formula": "(G+C-A-T)/L",
            "rc_formula": "(C+G-T-A)/L",
            "derived_sign": 1,
            "reason": "GC and AT totals are unchanged.",
        },
        {
            "scheme_id": config.axis_scheme_id,
            "component": "e3",
            "formula": "(A+C-G-T)/L",
            "rc_formula": "(T+G-C-A)/L",
            "derived_sign": -1,
            "reason": "MK chemistry changes sign under complement.",
        },
        {
            "scheme_id": config.axis_scheme_id,
            "component": "e4",
            "formula": "(G-C)/(G+C+epsilon)",
            "rc_formula": "(C-G)/(C+G+epsilon)",
            "derived_sign": -1,
            "reason": "GC denominator is unchanged and numerator changes sign.",
        },
        {
            "scheme_id": config.axis_scheme_id,
            "component": "e5",
            "formula": "(A-T)/(A+T+epsilon)",
            "rc_formula": "(T-A)/(T+A+epsilon)",
            "derived_sign": -1,
            "reason": "AT denominator is unchanged and numerator changes sign.",
        },
        {
            "scheme_id": config.axis_scheme_id,
            "component": "e6",
            "formula": "normalized k-mer entropy",
            "rc_formula": "entropy of reverse-complemented k-mer multiset",
            "derived_sign": 1,
            "reason": "Reverse complement is a bijection on valid k-mers.",
        },
        {
            "scheme_id": config.axis_scheme_id,
            "component": "e7",
            "formula": "2*RC_similarity(window)-1",
            "rc_formula": "2*RC_similarity(RC(window))-1",
            "derived_sign": 1,
            "reason": "The similarity between s and RC(s) is symmetric.",
        },
    ]


def _reverse_complement_row(
    *,
    source: str,
    sequence_id: str,
    position: int,
    start: int,
    end: int,
    window: str,
    config: EncodingAuditConfig,
) -> dict[str, object]:
    encoded = encode_dna_window(
        window,
        k=config.kmer_k,
        epsilon=config.epsilon,
        max_ambiguous_fraction=config.max_ambiguous_fraction,
    )
    rc_window = reverse_complement(window)
    encoded_rc = encode_dna_window(
        rc_window,
        k=config.kmer_k,
        epsilon=config.epsilon,
        max_ambiguous_fraction=config.max_ambiguous_fraction,
    )
    base = {
        "source": source,
        "sequence_id": sequence_id,
        "position": position,
        "start": start,
        "end": end,
        "window": window,
        "reverse_complement_window": rc_window,
        "window_length": len(window),
        "kmer_k": config.kmer_k,
        "epsilon": config.epsilon,
        "max_ambiguous_fraction": config.max_ambiguous_fraction,
    }
    if encoded is None and encoded_rc is None:
        return {
            **base,
            "status": "skipped",
            "exception_reason": "empty, fully ambiguous, or ambiguity threshold exceeded",
            "max_abs_residual": np.nan,
            "residual_norm": np.nan,
        }
    if encoded is None or encoded_rc is None:
        return {
            **base,
            "status": "exception",
            "exception_reason": "one orientation was encodable and the other was skipped",
            "max_abs_residual": np.inf,
            "residual_norm": np.inf,
        }
    expected = transform_octonion_rc(encoded[0], config.axis_scheme_id).components
    observed = encoded_rc[0].components
    residual = observed - expected
    row = {
        **base,
        "status": "pass" if float(np.max(np.abs(residual))) <= config.tolerance else "fail",
        "exception_reason": "none",
        "max_abs_residual": float(np.max(np.abs(residual))),
        "residual_norm": float(np.linalg.norm(residual)),
    }
    row.update({f"original_e{i}": float(encoded[0].components[i]) for i in range(8)})
    row.update({f"expected_rc_e{i}": float(expected[i]) for i in range(8)})
    row.update({f"observed_rc_e{i}": float(observed[i]) for i in range(8)})
    row.update({f"residual_e{i}": float(residual[i]) for i in range(8)})
    return row


def _finite_window_reverse_complement_rows(
    record: FastaRecord,
    config: EncodingAuditConfig,
) -> list[dict[str, object]]:
    cleaned = "".join(record.sequence.split()).upper()
    starts = list(range(0, max(len(cleaned) - config.window_size + 1, 0), config.step))
    available = set(starts)
    rows: list[dict[str, object]] = []
    for position, start in enumerate(starts):
        mirror_start = len(cleaned) - (start + config.window_size)
        rows.append(
            {
                "source": "finite_window_effect",
                "sequence_id": record.id,
                "position": position,
                "start": start,
                "end": start + config.window_size,
                "window": cleaned[start : start + config.window_size],
                "reverse_complement_window": "NA",
                "window_length": config.window_size,
                "kmer_k": config.kmer_k,
                "epsilon": config.epsilon,
                "max_ambiguous_fraction": config.max_ambiguous_fraction,
                "status": "mapped" if mirror_start in available else "unmapped_by_step_phase",
                "exception_reason": (
                    "mirror window start is sampled"
                    if mirror_start in available
                    else "finite-window step phase omits the mirrored start"
                ),
                "max_abs_residual": np.nan,
                "residual_norm": np.nan,
                "mirror_start": mirror_start,
            }
        )
    return rows


def _codon_collisions(catalog: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    components = catalog[COMPONENT_COLUMNS].to_numpy(dtype=float)
    codons = catalog["codon"].astype(str).to_numpy()
    rounded = np.round(components / tolerance).astype(np.int64)
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, values in enumerate(rounded):
        groups.setdefault(tuple(int(value) for value in values), []).append(index)
    for indices in groups.values():
        if len(indices) > 1:
            rows.append(
                {
                    "collision_type": "near_or_exact",
                    "codons": ",".join(codons[index] for index in indices),
                    "n_codons": len(indices),
                    "max_pairwise_distance": float(
                        max(
                            np.linalg.norm(components[i] - components[j])
                            for i, j in combinations(indices, 2)
                        )
                    ),
                    "tolerance": tolerance,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["collision_type", "codons", "n_codons", "max_pairwise_distance", "tolerance"],
    )


def _codon_distance_matrix(catalog: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = catalog[COMPONENT_COLUMNS].to_numpy(dtype=float)
    codons = catalog["codon"].astype(str).to_numpy()
    amino_acids = catalog["amino_acid"].astype(str).to_numpy()
    stops = catalog["is_stop"].astype(bool).to_numpy()
    for i, j in product(range(len(catalog)), repeat=2):
        diff = values[i] - values[j]
        commutator = Octonion(values[i]).commutator(Octonion(values[j])).components
        rows.append(
            {
                "codon_a": codons[i],
                "codon_b": codons[j],
                "amino_acid_a": amino_acids[i],
                "amino_acid_b": amino_acids[j],
                "same_amino_acid": amino_acids[i] == amino_acids[j],
                "both_stop": bool(stops[i] and stops[j]),
                "euclidean_distance": float(np.linalg.norm(diff)),
                "octonion_commutator_distance": float(np.linalg.norm(commutator)),
                "component_dot": float(np.dot(values[i], values[j])),
            }
        )
    return pd.DataFrame(rows)


def _codon_synonymy_statistics(catalog: pd.DataFrame, distances: pd.DataFrame) -> pd.DataFrame:
    pair_table = distances[distances["codon_a"] < distances["codon_b"]].copy()
    rows: list[dict[str, object]] = []
    categories = {
        "synonymous_sense_pairs": pair_table[
            pair_table["same_amino_acid"] & (pair_table["amino_acid_a"] != "*")
        ],
        "nonsynonymous_sense_pairs": pair_table[
            (pair_table["amino_acid_a"] != pair_table["amino_acid_b"])
            & (pair_table["amino_acid_a"] != "*")
            & (pair_table["amino_acid_b"] != "*")
        ],
        "stop_codon_pairs": pair_table[pair_table["both_stop"]],
        "start_codon_to_other_pairs": _start_pairs(pair_table, catalog),
    }
    for category, subset in categories.items():
        rows.append(_distance_summary_row(category, subset))

    substitution_effects = _single_substitution_pairs(catalog, pair_table)
    for category, subset in {
        "first_position_substitutions": substitution_effects[substitution_effects["position"] == 1],
        "second_position_substitutions": substitution_effects[substitution_effects["position"] == 2],
        "third_position_substitutions": substitution_effects[substitution_effects["position"] == 3],
        "transition_substitutions": substitution_effects[substitution_effects["substitution_type"] == "transition"],
        "transversion_substitutions": substitution_effects[substitution_effects["substitution_type"] == "transversion"],
        "wobble_position_substitutions": substitution_effects[substitution_effects["position"] == 3],
    }.items():
        rows.append(_distance_summary_row(category, subset))

    components = catalog[COMPONENT_COLUMNS].to_numpy(dtype=float)
    rows.append(
        {
            "category": "component_geometry",
            "n_pairs": len(pair_table),
            "mean_euclidean_distance": float(pair_table["euclidean_distance"].mean()),
            "median_euclidean_distance": float(pair_table["euclidean_distance"].median()),
            "min_euclidean_distance": float(pair_table["euclidean_distance"].min()),
            "max_euclidean_distance": float(pair_table["euclidean_distance"].max()),
            "mean_commutator_distance": float(pair_table["octonion_commutator_distance"].mean()),
            "component_rank": _safe_matrix_rank(components),
            "effective_dimensionality": _effective_dimensionality(components),
        }
    )
    return pd.DataFrame(rows)


def _codon_substitution_effects(catalog: pd.DataFrame, genetic_code: GeneticCode) -> pd.DataFrame:
    by_codon = catalog.set_index("codon")
    rows: list[dict[str, object]] = []
    for codon in all_standard_codons():
        left = by_codon.loc[codon, COMPONENT_COLUMNS].to_numpy(dtype=float)
        aa = genetic_code.amino_acid(codon)
        for position in range(3):
            for replacement in DNA_ALPHABET:
                if replacement == codon[position]:
                    continue
                changed = codon[:position] + replacement + codon[position + 1 :]
                right = by_codon.loc[changed, COMPONENT_COLUMNS].to_numpy(dtype=float)
                changed_aa = genetic_code.amino_acid(changed)
                rows.append(
                    {
                        "codon": codon,
                        "mutated_codon": changed,
                        "position": position + 1,
                        "from_base": codon[position],
                        "to_base": replacement,
                        "substitution_type": _base_substitution_type(codon[position], replacement),
                        "amino_acid": aa,
                        "mutated_amino_acid": changed_aa,
                        "is_synonymous": aa == changed_aa,
                        "is_nonsynonymous": aa != changed_aa,
                        "is_stop_change": aa == "*" or changed_aa == "*",
                        "euclidean_distance": float(np.linalg.norm(left - right)),
                        "commutator_distance": Octonion(left).commutator(Octonion(right)).norm(),
                        "norm_delta": float(np.linalg.norm(right) - np.linalg.norm(left)),
                    }
                )
    return pd.DataFrame(rows)


def _codon_rank_spectrum(catalog: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    values = catalog[COMPONENT_COLUMNS].to_numpy(dtype=float)
    centered = values - values.mean(axis=0)
    singular = _safe_singular_values(centered)
    total = float(np.sum(singular * singular))
    rows = []
    for index, value in enumerate(singular):
        rows.append(
            {
                "feature_family": "codon_product_components",
                "singular_value_index": index,
                "singular_value": float(value),
                "energy": float(value * value),
                "variance_share": float(value * value / total) if total else 0.0,
                "above_tolerance": bool(value > tolerance),
            }
        )
    return pd.DataFrame(rows)


def _window_component_sequences(
    records: list[FastaRecord],
    config: EncodingAuditConfig,
) -> list[NDArray[np.float64]]:
    sequences = []
    for record in records:
        rows = []
        for window in iter_windows(record.sequence, config.window_size, config.step):
            encoded = encode_dna_window(
                window.sequence,
                k=config.kmer_k,
                epsilon=config.epsilon,
                max_ambiguous_fraction=config.max_ambiguous_fraction,
            )
            if encoded is not None:
                rows.append(encoded[0].components)
        if rows:
            sequences.append(np.vstack(rows).astype(float))
    return sequences


def _feature_family_matrices(
    sequences: list[NDArray[np.float64]],
) -> dict[str, NDArray[np.float64]]:
    windows = _stack_or_empty(sequences, 8)
    left, right = _adjacent_pairs(sequences)
    first, second, third = _triplets(sequences)
    matrices: dict[str, NDArray[np.float64]] = {
        "original_component_vector": windows,
        "adjacent_component_difference": right - left if len(left) else np.empty((0, 8)),
        "ordinary_pairwise_products": _ordinary_pairwise_products(left, right),
        "antisymmetric_exterior_products": _antisymmetric_products(left, right, include_scalar=False),
        "real_antisymmetric_control": _antisymmetric_products(left, right, include_scalar=False),
        "full_octonion_product": _batch_product(left, right),
        "commutator": _batch_commutator(left, right),
        "associator": _batch_associator(first, second, third),
        "fano_line_norms": _batch_fano_line_norms(left, right),
        "ordinary_polynomial_interaction_control": _ordinary_pairwise_products(left, right),
    }
    return matrices


def _feature_family_targets(
    sequences: list[NDArray[np.float64]],
    matrices: dict[str, NDArray[np.float64]],
) -> dict[str, NDArray[np.float64]]:
    windows = _stack_or_empty(sequences, 8)
    _, right = _adjacent_pairs(sequences)
    _, _, third = _triplets(sequences)
    targets: dict[str, NDArray[np.float64]] = {}
    for family, matrix in matrices.items():
        if family == "original_component_vector":
            targets[family] = windows[: len(matrix)]
        elif family == "associator":
            targets[family] = third[: len(matrix)]
        else:
            targets[family] = right[: len(matrix)]
    return targets


def _matrix_redundancy_stats(
    family: str,
    matrix: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    tolerance: float,
    random_seed: int,
) -> dict[str, object]:
    if matrix.size == 0 or len(matrix) == 0:
        return {
            "feature_family": family,
            "n_rows": 0,
            "n_features": matrix.shape[1] if matrix.ndim == 2 else 0,
            "numerical_rank": 0,
            "condition_number": np.nan,
            "variance_explained_pc1": np.nan,
            "mean_abs_feature_correlation": np.nan,
            "linear_r2_predict_components": np.nan,
            "tree_r2_predict_components": np.nan,
            "mutual_information_with_components": np.nan,
            "collision_rate": np.nan,
            "recoverability": "not_enough_rows",
            "max_abs_identity_residual": np.nan,
            "interpretation": "No rows available.",
            "_singular_values": [],
        }
    centered = matrix - matrix.mean(axis=0)
    singular = _safe_singular_values(centered)
    positive = singular[singular > tolerance]
    total = float(np.sum(singular * singular))
    corr = _mean_abs_correlation(matrix)
    linear_r2 = _linear_r2(matrix, target)
    tree_r2 = _tree_r2(matrix, target, random_seed)
    mi = _mutual_information(matrix, target, random_seed)
    return {
        "feature_family": family,
        "n_rows": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "numerical_rank": _safe_matrix_rank(matrix, tol=tolerance),
        "condition_number": float(positive.max() / positive.min()) if len(positive) else np.inf,
        "variance_explained_pc1": float((singular[0] * singular[0]) / total) if total else 0.0,
        "mean_abs_feature_correlation": corr,
        "linear_r2_predict_components": linear_r2,
        "tree_r2_predict_components": tree_r2,
        "mutual_information_with_components": mi,
        "collision_rate": _collision_rate(matrix, tolerance),
        "recoverability": "linearly_recoverable" if linear_r2 >= 1.0 - 1e-8 else "lossy_or_nonlinear",
        "max_abs_identity_residual": np.nan,
        "interpretation": _feature_family_interpretation(family),
        "_singular_values": [float(value) for value in singular],
    }


def _sequence_signature(
    sequence: str,
    config: EncodingAuditConfig,
    genetic_code: GeneticCode,
) -> dict[str, NDArray[np.float64] | float]:
    records = [FastaRecord(id="sequence", description="sequence", sequence=sequence)]
    windows = _window_component_sequences(records, config)
    values = _stack_or_empty(windows, 8)
    left, right = _adjacent_pairs(windows)
    first, second, third = _triplets(windows)
    products = _batch_product(left, right)
    commutators = _batch_commutator(left, right)
    associators = _batch_associator(first, second, third)
    profile = _fano_line_share_profile(values)
    component_mean = values.mean(axis=0) if len(values) else np.zeros(8)
    component_std = values.std(axis=0) if len(values) else np.zeros(8)
    codon_means = _codon_component_mean(sequence, genetic_code, config)
    fingerprint = np.concatenate(
        [
            component_mean,
            component_std,
            profile,
            codon_means,
            np.array(
                [
                    _mean_norm(values),
                    _mean_norm(products),
                    _mean_norm(commutators),
                    _mean_norm(associators),
                ],
                dtype=float,
            ),
        ]
    )
    return {
        "component_mean": component_mean,
        "fano_profile": profile,
        "fingerprint": fingerprint,
        "mean_octonion_norm": _mean_norm(values),
        "mean_product_norm": _mean_norm(products),
        "mean_commutator_norm": _mean_norm(commutators),
        "mean_associator_norm": _mean_norm(associators),
        "mean_transition_score": _mean_norm(commutators),
    }


def _axis_controls(config: EncodingAuditConfig) -> list[dict[str, object]]:
    rng = np.random.default_rng(config.random_seed)
    controls: list[dict[str, object]] = [
        _control_row(
            "canonical",
            "canonical",
            tuple(range(8)),
            tuple([1] * 8),
            changes_labels=False,
            changes_coordinates=False,
            changes_table=False,
            notes="Original axis assignment and multiplication table.",
        ),
        _control_row(
            "label_swap_e1_e2",
            "biological_label_permutation",
            tuple(range(8)),
            tuple([1] * 8),
            changes_labels=True,
            changes_coordinates=False,
            changes_table=False,
            notes="Only the biological labels are regarded as swapped; components are unchanged.",
        ),
    ]

    automorphisms = _fano_automorphism_permutations()
    for index, perm in enumerate(automorphisms[: min(3, len(automorphisms))]):
        controls.append(
            _control_row(
                f"fano_automorphism_{index}",
                "fano_plane_automorphism",
                (0, *perm),
                tuple([1] * 8),
                changes_labels=True,
                changes_coordinates=True,
                changes_table=False,
                notes="Discrete signed-free Fano-plane automorphism preserving products.",
            )
        )

    all_perm_count = 0
    if config.permutation_samples <= 0:
        return controls
    for index in range(config.permutation_samples):
        random_perm = [0, *rng.permutation(np.arange(1, 8)).tolist()]
        signs = [1, *rng.choice(np.array([-1, 1]), size=7).astype(int).tolist()]
        category = "coordinate_permutation_not_automorphism"
        if _is_fano_automorphism(tuple(random_perm[1:])):
            category = "fano_plane_automorphism"
        controls.append(
            _control_row(
                f"random_permutation_{index}",
                category,
                tuple(int(axis) for axis in random_perm),
                tuple(int(sign) for sign in signs),
                changes_labels=True,
                changes_coordinates=True,
                changes_table=False,
                notes="Random coordinate permutation/sign flip under the fixed multiplication table.",
            )
        )
        all_perm_count += 1

    controls.append(
        _control_row(
            "triad_preserving_swap",
            "permutation_preserving_selected_biological_triads",
            (0, 2, 3, 1, 4, 5, 6, 7),
            tuple([1] * 8),
            changes_labels=True,
            changes_coordinates=True,
            changes_table=False,
            notes="Cycles the e1/e2/e3 base-chemistry triad while leaving other axes fixed.",
        )
    )
    controls.append(
        _control_row(
            "random_antisymmetric_tensor",
            "random_antisymmetric_multiplication_tensor",
            tuple(range(8)),
            tuple([1] * 8),
            changes_labels=False,
            changes_coordinates=False,
            changes_table=True,
            notes="Control representing a random antisymmetric interaction tensor, not octonions.",
        )
    )
    controls.append(
        _control_row(
            "ordinary_polynomial_interactions",
            "ordinary_polynomial_interaction_features",
            tuple(range(8)),
            tuple([1] * 8),
            changes_labels=False,
            changes_coordinates=False,
            changes_table=True,
            notes="Real-valued pairwise interaction expansion without Fano multiplication.",
        )
    )
    del all_perm_count
    return controls


def _control_row(
    control_id: str,
    category: str,
    permutation_value: tuple[int, ...],
    signs: tuple[int, ...],
    *,
    changes_labels: bool,
    changes_coordinates: bool,
    changes_table: bool,
    notes: str,
) -> dict[str, object]:
    automorphism = _is_fano_automorphism(tuple(permutation_value[1:]))
    preserves = automorphism and all(sign == 1 for sign in signs)
    if category == "canonical":
        preserves = True
    if changes_table:
        preserves = False
    return {
        "control_id": control_id,
        "transformation_category": category,
        "permutation": permutation_value,
        "signs": signs,
        "changes_biological_labels": changes_labels,
        "changes_coordinates": changes_coordinates,
        "changes_multiplication_table": changes_table,
        "is_fano_plane_automorphism": automorphism,
        "preserves_multiplication_table": preserves,
        "notes": notes,
    }


def _component_matrix_for_axis_audit(
    records: list[FastaRecord],
    config: EncodingAuditConfig,
) -> NDArray[np.float64]:
    if config.seq_type == "dna":
        sequences = _window_component_sequences(records, config)
    else:
        sequences = []
        for record in records:
            rows = []
            for window in iter_windows(record.sequence, config.window_size, config.step):
                encoded = encode_protein_window(
                    window.sequence,
                    k=config.kmer_k,
                    epsilon=config.epsilon,
                    max_ambiguous_fraction=config.max_ambiguous_fraction,
                )
                if encoded is not None:
                    rows.append(encoded[0].components)
            if rows:
                sequences.append(np.vstack(rows).astype(float))
    return _stack_or_empty(sequences, 8)


def _apply_axis_control(
    values: NDArray[np.float64],
    control: dict[str, object],
) -> NDArray[np.float64]:
    if values.size == 0:
        return values
    category = str(control["transformation_category"])
    if category in {
        "random_antisymmetric_multiplication_tensor",
        "ordinary_polynomial_interaction_features",
        "biological_label_permutation",
        "canonical",
    }:
        return values.copy()
    perm = tuple(
        int(str(axis)) for axis in cast(Iterable[object], control["permutation"])
    )
    signs = np.array(
        [int(str(sign)) for sign in cast(Iterable[object], control["signs"])],
        dtype=float,
    )
    transformed = values[:, perm].copy()
    transformed *= signs
    return transformed


def _fano_line_share_profile(values: NDArray[np.float64]) -> NDArray[np.float64]:
    if len(values) < 2:
        return np.zeros(len(FANO_LINES), dtype=float)
    left = values[:-1]
    right = values[1:]
    norms = _batch_fano_line_norms(left, right).sum(axis=0)
    total = float(norms.sum())
    return norms / total if total else norms


def _adjacent_transition_scores(values: NDArray[np.float64]) -> NDArray[np.float64]:
    if len(values) < 2:
        return np.empty(0, dtype=float)
    return np.linalg.norm(_batch_commutator(values[:-1], values[1:]), axis=1)


def _fano_automorphism_permutations() -> list[tuple[int, ...]]:
    positive_triples = set()
    for a, b, c in FANO_LINES:
        positive_triples.update({(a, b, c), (b, c, a), (c, a, b)})
    automorphisms = []
    for perm in permutations(range(1, 8)):
        mapping = {axis: perm[axis - 1] for axis in range(1, 8)}
        if all(
            (mapping[a], mapping[b], mapping[c]) in positive_triples
            for a, b, c in FANO_LINES
        ):
            automorphisms.append(tuple(int(axis) for axis in perm))
    return automorphisms


def _is_fano_automorphism(perm: tuple[int, ...]) -> bool:
    if len(perm) != 7 or sorted(perm) != list(range(1, 8)):
        return False
    positive_triples = set()
    for a, b, c in FANO_LINES:
        positive_triples.update({(a, b, c), (b, c, a), (c, a, b)})
    mapping = {axis: perm[axis - 1] for axis in range(1, 8)}
    return all((mapping[a], mapping[b], mapping[c]) in positive_triples for a, b, c in FANO_LINES)


def _stack_or_empty(sequences: list[NDArray[np.float64]], width: int) -> NDArray[np.float64]:
    usable = [values for values in sequences if len(values)]
    return np.vstack(usable) if usable else np.empty((0, width), dtype=float)


def _adjacent_pairs(
    sequences: list[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    left = []
    right = []
    for values in sequences:
        if len(values) >= 2:
            left.append(values[:-1])
            right.append(values[1:])
    return _stack_or_empty(left, 8), _stack_or_empty(right, 8)


def _triplets(
    sequences: list[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    first = []
    second = []
    third = []
    for values in sequences:
        if len(values) >= 3:
            first.append(values[:-2])
            second.append(values[1:-1])
            third.append(values[2:])
    return _stack_or_empty(first, 8), _stack_or_empty(second, 8), _stack_or_empty(third, 8)


def _ordinary_pairwise_products(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.float64]:
    if len(left) == 0:
        return np.empty((0, 64), dtype=float)
    return np.einsum("ni,nj->nij", left, right).reshape(len(left), 64)


def _antisymmetric_products(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    *,
    include_scalar: bool,
) -> NDArray[np.float64]:
    axes = range(8) if include_scalar else range(1, 8)
    pairs = list(combinations(axes, 2))
    if len(left) == 0:
        return np.empty((0, len(pairs)), dtype=float)
    out = np.zeros((len(left), len(pairs)), dtype=float)
    for column, (i, j) in enumerate(pairs):
        out[:, column] = left[:, i] * right[:, j] - left[:, j] * right[:, i]
    return out


def _batch_product(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
    if len(left) == 0:
        return np.empty((0, 8), dtype=float)
    return np.vstack([octonion_multiply(left[index], right[index]) for index in range(len(left))])


def _batch_commutator(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
    if len(left) == 0:
        return np.empty((0, 8), dtype=float)
    return np.vstack([octonion_commutator(left[index], right[index]) for index in range(len(left))])


def _batch_associator(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    third: NDArray[np.float64],
) -> NDArray[np.float64]:
    if len(first) == 0:
        return np.empty((0, 8), dtype=float)
    return np.vstack(
        [octonion_associator(first[index], second[index], third[index]) for index in range(len(first))]
    )


def _batch_fano_line_norms(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.float64]:
    if len(left) == 0:
        return np.empty((0, len(FANO_LINES)), dtype=float)
    out = np.zeros((len(left), len(FANO_LINES)), dtype=float)
    for row_index in range(len(left)):
        x = left[row_index]
        y = right[row_index]
        for line_index, (a, b, c) in enumerate(FANO_LINES):
            pair_ab_to_c = x[a] * y[b] - x[b] * y[a]
            pair_bc_to_a = x[b] * y[c] - x[c] * y[b]
            pair_ca_to_b = x[c] * y[a] - x[a] * y[c]
            out[row_index, line_index] = sqrt(
                pair_ab_to_c * pair_ab_to_c
                + pair_bc_to_a * pair_bc_to_a
                + pair_ca_to_b * pair_ca_to_b
            )
    return out


def _commutator_cross_product_residual(
    sequences: list[NDArray[np.float64]],
) -> dict[str, float | int]:
    left, right = _adjacent_pairs(sequences)
    if len(left) == 0:
        return {"n_rows": 0, "max_abs_residual": np.nan}
    comm = _batch_commutator(left, right)[:, 1:]
    cross = np.vstack([_cross_product7(left[index, 1:], right[index, 1:]) for index in range(len(left))])
    residual = comm - 2.0 * cross
    return {"n_rows": len(left), "max_abs_residual": float(np.max(np.abs(residual)))}


def _cross_product7(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
    x = np.zeros(8, dtype=float)
    y = np.zeros(8, dtype=float)
    x[1:] = left
    y[1:] = right
    return octonion_multiply(x, y)[1:]


def _linear_r2(features: NDArray[np.float64], target: NDArray[np.float64]) -> float:
    if len(features) == 0 or len(target) == 0:
        return np.nan
    n = min(len(features), len(target))
    x = np.column_stack([np.ones(n), features[:n]])
    y = target[:n]
    try:
        xtx = _cross_product_matrix(x, x)
        for index in range(xtx.shape[0]):
            xtx[index, index] += 1e-12
        xty = _cross_product_matrix(x, y)
        coef = _solve_linear_system(xtx, xty)
    except ValueError:
        return np.nan
    prediction = _matmul(x, coef)
    return _r2_score(y, prediction)


def _tree_r2(
    features: NDArray[np.float64],
    target: NDArray[np.float64],
    random_seed: int,
) -> float:
    try:
        from sklearn.tree import DecisionTreeRegressor
    except Exception:
        return np.nan
    if len(features) < 2 or len(target) < 2:
        return np.nan
    n = min(len(features), len(target))
    model = DecisionTreeRegressor(max_depth=3, random_state=random_seed)
    model.fit(features[:n], target[:n])
    return _r2_score(target[:n], model.predict(features[:n]))


def _mutual_information(
    features: NDArray[np.float64],
    target: NDArray[np.float64],
    random_seed: int,
) -> float:
    try:
        from sklearn.feature_selection import mutual_info_regression
    except Exception:
        return np.nan
    if len(features) < 4 or len(target) < 4:
        return np.nan
    n = min(len(features), len(target))
    values = []
    for column in range(target.shape[1]):
        y = target[:n, column]
        if np.allclose(y, y[0]):
            continue
        try:
            mi = mutual_info_regression(features[:n], y, random_state=random_seed)
        except Exception:
            continue
        values.append(float(np.mean(mi)))
    return float(np.mean(values)) if values else 0.0


def _r2_score(target: NDArray[np.float64], prediction: NDArray[np.float64]) -> float:
    ss_res = float(np.sum((target - prediction) ** 2))
    centered = target - target.mean(axis=0)
    ss_tot = float(np.sum(centered**2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _mean_abs_correlation(matrix: NDArray[np.float64]) -> float:
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return np.nan
    std = matrix.std(axis=0)
    usable = matrix[:, std > 0]
    if usable.shape[1] < 2:
        return 0.0
    values = []
    for i in range(usable.shape[1]):
        for j in range(i + 1, usable.shape[1]):
            values.append(abs(_pearson(usable[:, i], usable[:, j])))
    return float(np.mean(values)) if values else 0.0


def _collision_rate(matrix: NDArray[np.float64], tolerance: float) -> float:
    if len(matrix) == 0:
        return np.nan
    rounded = np.round(matrix / tolerance).astype(np.int64)
    unique = {tuple(row.tolist()) for row in rounded}
    return float(1.0 - len(unique) / len(matrix))


def _feature_family_interpretation(family: str) -> str:
    interpretations = {
        "original_component_vector": "Raw encoded descriptors before octonion multiplication.",
        "adjacent_component_difference": "Ordinary first difference of adjacent descriptor vectors.",
        "ordinary_pairwise_products": "Real-valued pairwise products x_i*y_j without Fano signs.",
        "antisymmetric_exterior_products": "Ordinary antisymmetric terms x_i*y_j - x_j*y_i.",
        "real_antisymmetric_control": "Control matching antisymmetric interaction content without octonion language.",
        "full_octonion_product": "Fano-signed product plus scalar-imaginary and dot-product terms.",
        "commutator": "Pure antisymmetric Fano interaction; scalar terms cancel.",
        "associator": "Non-associativity of three adjacent octonions under the chosen algebra.",
        "fano_line_norms": "Norm summaries of exterior terms grouped by Fano line.",
        "ordinary_polynomial_interaction_control": "Polynomial control without a multiplication table.",
    }
    return interpretations.get(family, "Audit feature family.")


def _codon_component_mean(
    sequence: str,
    genetic_code: GeneticCode,
    config: EncodingAuditConfig,
) -> NDArray[np.float64]:
    cleaned = "".join(sequence.split()).upper()
    rows = []
    for start in range(0, len(cleaned) - 2, 3):
        codon = cleaned[start : start + 3]
        encoded = encode_codon(
            codon,
            genetic_code,
            max_ambiguous_fraction=config.max_ambiguous_fraction,
            include_stop_codons=True,
            normalize=config.normalize_codons,
        )
        if encoded is not None:
            rows.append(encoded.octonion.components)
    return np.mean(np.vstack(rows), axis=0) if rows else np.zeros(8)


def _mean_norm(values: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(values, axis=1).mean()) if len(values) else 0.0


def _position_fraction(position_value: int, sequence_length: int) -> float:
    if position_value < 0 or sequence_length <= 1:
        return np.nan
    return float(position_value / (sequence_length - 1))


def _mononucleotide_shuffle(sequence: str, random_seed: int) -> str:
    cleaned = "".join(sequence.split()).upper()
    rng = np.random.default_rng(random_seed)
    chars = np.array(list(cleaned), dtype="U1")
    rng.shuffle(chars)
    return "".join(chars.tolist())


def _dinucleotide_counts(sequence: str) -> Counter[str]:
    return Counter(sequence[index : index + 2] for index in range(max(len(sequence) - 1, 0)))


AA_CLASSES = {
    "A": "small",
    "G": "small",
    "S": "polar",
    "T": "polar",
    "N": "polar",
    "Q": "polar",
    "D": "negative",
    "E": "negative",
    "K": "positive",
    "R": "positive",
    "H": "positive",
    "V": "hydrophobic",
    "L": "hydrophobic",
    "I": "hydrophobic",
    "M": "hydrophobic",
    "F": "aromatic",
    "W": "aromatic",
    "Y": "aromatic",
    "C": "special",
    "P": "special",
    "*": "stop",
}


def _replacement_codon_for_aa_class(
    amino_acid: str,
    genetic_code: GeneticCode,
    *,
    same_class: bool,
) -> str | None:
    source_class = AA_CLASSES.get(amino_acid, "unknown")
    for codon in all_standard_codons():
        aa = genetic_code.amino_acid(codon)
        if aa == amino_acid or aa == "*":
            continue
        if (AA_CLASSES.get(aa, "unknown") == source_class) == same_class:
            return codon
    return None


def _base_substitution_type(left: str, right: str) -> str:
    return "transition" if TRANSITION_BASE.get(left) == right else "transversion"


def _start_pairs(pair_table: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    starts = set(catalog.loc[catalog["is_start"], "codon"].astype(str))
    return pair_table[
        pair_table["codon_a"].isin(starts)
        | pair_table["codon_b"].isin(starts)
    ]


def _single_substitution_pairs(catalog: pd.DataFrame, pair_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in pair_table.iterrows():
        left = str(row["codon_a"])
        right = str(row["codon_b"])
        diffs = [index for index in range(3) if left[index] != right[index]]
        if len(diffs) != 1:
            continue
        position_value = diffs[0] + 1
        rows.append(
            {
                **row.to_dict(),
                "position": position_value,
                "substitution_type": _base_substitution_type(left[diffs[0]], right[diffs[0]]),
            }
        )
    del catalog
    return pd.DataFrame(rows)


def _distance_summary_row(category: str, subset: pd.DataFrame) -> dict[str, object]:
    if subset.empty:
        return {
            "category": category,
            "n_pairs": 0,
            "mean_euclidean_distance": np.nan,
            "median_euclidean_distance": np.nan,
            "min_euclidean_distance": np.nan,
            "max_euclidean_distance": np.nan,
            "mean_commutator_distance": np.nan,
            "component_rank": np.nan,
            "effective_dimensionality": np.nan,
        }
    return {
        "category": category,
        "n_pairs": int(len(subset)),
        "mean_euclidean_distance": float(subset["euclidean_distance"].mean()),
        "median_euclidean_distance": float(subset["euclidean_distance"].median()),
        "min_euclidean_distance": float(subset["euclidean_distance"].min()),
        "max_euclidean_distance": float(subset["euclidean_distance"].max()),
        "mean_commutator_distance": float(subset["octonion_commutator_distance"].mean()),
        "component_rank": np.nan,
        "effective_dimensionality": np.nan,
    }


def _effective_dimensionality(values: NDArray[np.float64]) -> float:
    centered = values - values.mean(axis=0)
    singular = _safe_singular_values(centered)
    energy = singular * singular
    total = float(energy.sum())
    if total == 0:
        return 0.0
    probabilities = energy / total
    return float(1.0 / np.sum(probabilities * probabilities))


def _row_distance_matrix(values: NDArray[np.float64]) -> NDArray[np.float64]:
    if len(values) == 0:
        return np.empty((0, 0), dtype=float)
    diffs = values[:, None, :] - values[None, :, :]
    return np.linalg.norm(diffs, axis=2)


def _upper_triangle(values: NDArray[np.float64]) -> NDArray[np.float64]:
    if values.size == 0 or values.shape[0] < 2:
        return np.empty(0, dtype=float)
    return values[np.triu_indices(values.shape[0], k=1)]


def _nearest_neighbor_overlap(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> float:
    if left.shape != right.shape or left.shape[0] < 2:
        return np.nan
    left_copy = left.copy()
    right_copy = right.copy()
    np.fill_diagonal(left_copy, np.inf)
    np.fill_diagonal(right_copy, np.inf)
    return float(np.mean(np.argmin(left_copy, axis=1) == np.argmin(right_copy, axis=1)))


def _cosine(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0:
        return 1.0
    return float(np.dot(left, right) / denom)


def _spearman(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    if len(left) < 2 or len(right) < 2 or len(left) != len(right):
        return np.nan
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=float)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    if np.allclose(left_rank, left_rank[0]) or np.allclose(right_rank, right_rank[0]):
        return 1.0
    return _pearson(left_rank, right_rank)


def _safe_matrix_rank(values: NDArray[np.float64], tol: float = 1e-9) -> int:
    singular = _safe_singular_values(values)
    return int(np.sum(singular > tol))


def _safe_singular_values(values: NDArray[np.float64]) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        return np.empty(0, dtype=float)
    gram = _gram_matrix(matrix)
    eigenvalues, _ = _jacobi_eigh_symmetric(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    singular = np.sqrt(eigenvalues)
    return np.sort(singular)[::-1]


def _safe_covariance_eigh(values: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        return np.empty(0, dtype=float), np.empty((0, 0), dtype=float)
    covariance = _gram_matrix(matrix)
    eigenvalues, eigenvectors = _jacobi_eigh_symmetric(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def _gram_matrix(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    rows, columns = matrix.shape
    gram = np.zeros((columns, columns), dtype=float)
    for i in range(columns):
        for j in range(i, columns):
            total = 0.0
            for row in range(rows):
                total += float(matrix[row, i]) * float(matrix[row, j])
            gram[i, j] = total
            gram[j, i] = total
    return gram


def _cross_product_matrix(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.float64]:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.ndim != 2 or right_values.ndim != 2:
        raise ValueError("Cross-product matrices require 2D inputs.")
    if left_values.shape[0] != right_values.shape[0]:
        raise ValueError("Inputs must have the same number of rows.")
    out = np.zeros((left_values.shape[1], right_values.shape[1]), dtype=float)
    for i in range(left_values.shape[1]):
        for j in range(right_values.shape[1]):
            total = 0.0
            for row in range(left_values.shape[0]):
                total += float(left_values[row, i]) * float(right_values[row, j])
            out[i, j] = total
    return out


def _matmul(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.ndim != 2 or right_values.ndim != 2:
        raise ValueError("Matrix multiplication requires 2D inputs.")
    if left_values.shape[1] != right_values.shape[0]:
        raise ValueError("Matrix dimensions do not align.")
    out = np.zeros((left_values.shape[0], right_values.shape[1]), dtype=float)
    for i in range(left_values.shape[0]):
        for j in range(right_values.shape[1]):
            total = 0.0
            for k in range(left_values.shape[1]):
                total += float(left_values[i, k]) * float(right_values[k, j])
            out[i, j] = total
    return out


def _solve_linear_system(
    matrix: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    a = np.asarray(matrix, dtype=float).copy()
    b = np.asarray(rhs, dtype=float).copy()
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Coefficient matrix must be square.")
    if b.ndim == 1:
        b = b[:, None]
    if b.shape[0] != a.shape[0]:
        raise ValueError("Right-hand side has incompatible shape.")
    n = a.shape[0]
    for pivot in range(n):
        max_row = pivot
        max_value = abs(a[pivot, pivot])
        for row in range(pivot + 1, n):
            value = abs(a[row, pivot])
            if value > max_value:
                max_row = row
                max_value = value
        if max_value < 1e-15:
            raise ValueError("Singular linear system.")
        if max_row != pivot:
            a[[pivot, max_row], :] = a[[max_row, pivot], :]
            b[[pivot, max_row], :] = b[[max_row, pivot], :]
        pivot_value = a[pivot, pivot]
        a[pivot, :] /= pivot_value
        b[pivot, :] /= pivot_value
        for row in range(n):
            if row == pivot:
                continue
            factor = a[row, pivot]
            if factor == 0:
                continue
            a[row, :] -= factor * a[pivot, :]
            b[row, :] -= factor * b[pivot, :]
    return b


def _pearson(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        return np.nan
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denom = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denom == 0:
        return 1.0 if np.allclose(x, y) else 0.0
    return float(np.sum(x_centered * y_centered) / denom)


def _jacobi_eigh_symmetric(
    values: NDArray[np.float64],
    *,
    tolerance: float = 1e-12,
    max_sweeps: int = 200,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    matrix = np.asarray(values, dtype=float).copy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Jacobi eigensolver expects a square symmetric matrix.")
    n = matrix.shape[0]
    vectors = np.eye(n, dtype=float)
    if n == 0:
        return np.empty(0, dtype=float), vectors
    for _ in range(max_sweeps):
        off_diag = np.triu(np.abs(matrix), k=1)
        p, q = np.unravel_index(int(np.argmax(off_diag)), off_diag.shape)
        if off_diag[p, q] < tolerance:
            break
        if matrix[p, q] == 0:
            continue
        tau = (matrix[q, q] - matrix[p, p]) / (2.0 * matrix[p, q])
        t = np.sign(tau) / (abs(tau) + sqrt(1.0 + tau * tau)) if tau != 0 else 1.0
        c = 1.0 / sqrt(1.0 + t * t)
        s = t * c

        app = matrix[p, p]
        aqq = matrix[q, q]
        apq = matrix[p, q]
        matrix[p, p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        matrix[q, q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        matrix[p, q] = 0.0
        matrix[q, p] = 0.0

        for r in range(n):
            if r in {p, q}:
                continue
            arp = matrix[r, p]
            arq = matrix[r, q]
            matrix[r, p] = c * arp - s * arq
            matrix[p, r] = matrix[r, p]
            matrix[r, q] = s * arp + c * arq
            matrix[q, r] = matrix[r, q]

        vip = vectors[:, p].copy()
        viq = vectors[:, q].copy()
        vectors[:, p] = c * vip - s * viq
        vectors[:, q] = s * vip + c * viq
    return np.diag(matrix).copy(), vectors


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plot_codon_distance_heatmap(table: pd.DataFrame, path: Path) -> Path:
    codons = all_standard_codons()
    pivot = table.pivot(index="codon_a", columns="codon_b", values="euclidean_distance").reindex(
        index=codons,
        columns=codons,
    )
    return _draw_heatmap(
        pivot.to_numpy(dtype=float),
        path,
        "Codon Euclidean Distance",
        x_label="codon B (lexicographic index)",
        y_label="codon A (lexicographic index)",
        legend_label="octonion L2 distance (dimensionless)",
    )


def _plot_codon_pca(catalog: pd.DataFrame, path: Path) -> Path:
    values = catalog[COMPONENT_COLUMNS].to_numpy(dtype=float)
    centered = values - values.mean(axis=0)
    _, vectors = _safe_covariance_eigh(centered)
    coords = _matmul(centered, vectors[:, :2])
    labels = catalog["amino_acid"].astype(str).tolist()
    return _draw_scatter(
        coords,
        labels,
        path,
        "PCA of 64 Codon Octonions",
        x_label="PC1 score (dimensionless)",
        y_label="PC2 score (dimensionless)",
    )


def _plot_synonymous_geometry(distances: pd.DataFrame, path: Path) -> Path:
    table = distances[(distances["codon_a"] < distances["codon_b"]) & distances["same_amino_acid"]]
    means = table.groupby("amino_acid_a")["euclidean_distance"].mean().sort_values()
    return _draw_bar(
        means,
        path,
        "Synonymous-Family Mean Distance",
        x_label="mean octonion L2 distance (dimensionless)",
        note="descriptive geometry; no pass/fail threshold",
    )


def _plot_mutation_sensitivity(table: pd.DataFrame, path: Path) -> Path:
    grouped = table.groupby("perturbation_type")["sequence_fingerprint_delta_norm"].mean().sort_values()
    return _draw_bar(
        grouped,
        path,
        "Mutation Sensitivity",
        x_label="mean fingerprint delta L2 norm (dimensionless)",
        note="larger values indicate stronger encoding change; no fixed threshold",
    )


def _plot_singular_values(table: pd.DataFrame, path: Path) -> Path:
    grouped = table.groupby("singular_value_index")["singular_value"].mean()
    tolerance = _audit_tolerance(table)
    return _draw_line(
        grouped,
        path,
        "Singular-Value Spectrum",
        x_label="singular-value index",
        y_label="singular value (dimensionless)",
        threshold=tolerance,
        threshold_label=f"rank tolerance = {tolerance:g}",
    )


def _plot_reverse_complement_residuals(table: pd.DataFrame, path: Path) -> Path:
    values = table["max_abs_residual"].replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        values = pd.Series([0.0])
    tolerance = _audit_tolerance(table)
    return _draw_line(
        values.reset_index(drop=True),
        path,
        "Reverse-Complement Residuals",
        x_label="audited sequence/window index",
        y_label="maximum absolute residual (dimensionless)",
        threshold=tolerance,
        threshold_label=f"pass threshold = {tolerance:g}",
    )


def _plot_axis_stability(table: pd.DataFrame, path: Path) -> Path:
    grouped = table.set_index("control_id")["line_share_cosine_to_canonical"].sort_values()
    return _draw_bar(
        grouped,
        path,
        "Axis-Permutation Stability",
        x_label="cosine similarity to canonical line profile [0,1]",
        reference=1.0,
        reference_label="identity reference = 1.0",
    )


def _plot_fano_profile_stability(table: pd.DataFrame, path: Path) -> Path:
    grouped = table.set_index("control_id")["dominant_fano_line_share"].sort_values()
    return _draw_bar(
        grouped,
        path,
        "Dominant Fano-Line Share Stability",
        x_label="dominant Fano-line contribution share [0,1]",
        note="descriptive share; no pass/fail threshold",
    )


def _draw_heatmap(
    values: NDArray[np.float64],
    path: Path,
    title: str,
    *,
    x_label: str,
    y_label: str,
    legend_label: str,
) -> Path:
    width = 1000
    height = 900
    margin = 130
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font(18)
    draw.text((30, 24), title, fill=(20, 35, 45), font=font)
    finite = values[np.isfinite(values)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    cell = max(1, (width - margin - 150) // values.shape[0])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = _blue_red(values[i, j], vmin, vmax)
            x0 = margin + j * cell
            y0 = margin + i * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color)
    heat_size = cell * values.shape[0]
    draw.rectangle((margin, margin, margin + heat_size, margin + heat_size), outline=(20, 35, 45))
    draw.text((margin + heat_size // 2 - 85, margin + heat_size + 28), x_label, fill=(20, 35, 45), font=_font(13))
    draw.text((18, margin + heat_size // 2), y_label, fill=(20, 35, 45), font=_font(13))
    for tick, label in ((0, "0"), (values.shape[0] - 1, str(values.shape[0] - 1))):
        position = margin + tick * cell
        draw.text((position, margin + heat_size + 8), label, fill=(80, 90, 100), font=_font(11))
        draw.text((margin - 28, position), label, fill=(80, 90, 100), font=_font(11))
    legend_x = margin + heat_size + 45
    legend_top = margin
    legend_height = heat_size
    for offset in range(legend_height):
        fraction = 1.0 - offset / max(legend_height - 1, 1)
        value = vmin + fraction * (vmax - vmin)
        draw.line(
            (legend_x, legend_top + offset, legend_x + 24, legend_top + offset),
            fill=_blue_red(value, vmin, vmax),
        )
    draw.rectangle((legend_x, legend_top, legend_x + 24, legend_top + legend_height), outline=(20, 35, 45))
    draw.text((legend_x + 32, legend_top - 3), f"{vmax:.3g}", fill=(20, 35, 45), font=_font(11))
    draw.text((legend_x + 32, legend_top + legend_height - 12), f"{vmin:.3g}", fill=(20, 35, 45), font=_font(11))
    draw.text((legend_x - 5, legend_top + legend_height + 18), legend_label, fill=(20, 35, 45), font=_font(11))
    draw.text((30, 60), "Legend: color encodes pairwise distance; no pass/fail threshold", fill=(80, 90, 100), font=_font(12))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _draw_scatter(
    coords: NDArray[np.float64],
    labels: list[str],
    path: Path,
    title: str,
    *,
    x_label: str,
    y_label: str,
) -> Path:
    image = Image.new("RGB", (1100, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 24), title, fill=(20, 35, 45), font=_font(20))
    if len(coords):
        x = coords[:, 0]
        y = coords[:, 1]
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
        for index, (xv, yv) in enumerate(coords):
            px = _scale(float(xv), x_min, x_max, 90, 790)
            py = _scale(float(yv), y_min, y_max, 620, 90)
            color = _label_color(labels[index])
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color, outline=(30, 30, 30))
        draw.rectangle((90, 90, 790, 620), outline=(20, 35, 45))
        draw.text((360, 650), x_label, fill=(20, 35, 45), font=_font(13))
        draw.text((12, 340), y_label, fill=(20, 35, 45), font=_font(13))
        draw.text((90, 625), f"{x_min:.3g}", fill=(80, 90, 100), font=_font(11))
        draw.text((750, 625), f"{x_max:.3g}", fill=(80, 90, 100), font=_font(11))
        draw.text((45, 608), f"{y_min:.3g}", fill=(80, 90, 100), font=_font(11))
        draw.text((45, 86), f"{y_max:.3g}", fill=(80, 90, 100), font=_font(11))
        draw.text((825, 82), "Legend: amino-acid code", fill=(20, 35, 45), font=_font(13))
        for index, label in enumerate(sorted(set(labels))):
            legend_x = 825 + (index // 11) * 115
            legend_y = 115 + (index % 11) * 28
            color = _label_color(label)
            draw.ellipse((legend_x, legend_y, legend_x + 10, legend_y + 10), fill=color, outline=(30, 30, 30))
            draw.text((legend_x + 16, legend_y - 3), label, fill=(20, 35, 45), font=_font(12))
        draw.text((825, 455), "No pass/fail threshold", fill=(80, 90, 100), font=_font(12))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _draw_bar(
    values: pd.Series,
    path: Path,
    title: str,
    *,
    x_label: str,
    reference: float | None = None,
    reference_label: str | None = None,
    note: str | None = None,
) -> Path:
    image = Image.new("RGB", (1000, 760), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 24), title, fill=(20, 35, 45), font=_font(20))
    if note:
        draw.text((30, 58), note, fill=(80, 90, 100), font=_font(12))
    if not values.empty:
        clipped = values.tail(24)
        max_value = float(max(clipped.max(), reference or 0.0, 1e-12))
        chart_left = 245
        chart_right = 920
        chart_top = 90
        chart_bottom = 620
        bar_h = max(10, min(22, 485 // len(clipped)))
        y = 90
        for label, value in clipped.items():
            width = int((chart_right - chart_left) * float(value) / max_value)
            draw.rectangle((chart_left, y, chart_left + width, y + bar_h), fill=(20, 116, 153))
            draw.text((20, y), str(label)[:28], fill=(20, 35, 45), font=_font(12))
            draw.text((chart_left + width + 8, y), f"{float(value):.3g}", fill=(20, 35, 45), font=_font(12))
            y += bar_h + 6
        draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=(20, 35, 45), width=1)
        for fraction in (0.0, 0.5, 1.0):
            x = int(chart_left + fraction * (chart_right - chart_left))
            draw.line((x, chart_bottom, x, chart_bottom + 6), fill=(20, 35, 45), width=1)
            draw.text((x - 10, chart_bottom + 8), f"{fraction * max_value:.3g}", fill=(80, 90, 100), font=_font(11))
        draw.text((chart_left + 115, 660), x_label, fill=(20, 35, 45), font=_font(13))
        draw.rectangle((25, 712, 37, 724), fill=(20, 116, 153))
        draw.text((44, 709), "observed mean", fill=(20, 35, 45), font=_font(11))
        if reference is not None:
            reference_x = _scale(reference, 0.0, max_value, chart_left, chart_right)
            draw.line((reference_x, chart_top, reference_x, chart_bottom), fill=(210, 70, 55), width=2)
            draw.line((250, 718, 270, 718), fill=(210, 70, 55), width=2)
            draw.text((278, 709), reference_label or f"reference = {reference:g}", fill=(20, 35, 45), font=_font(11))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _draw_line(
    values: pd.Series,
    path: Path,
    title: str,
    *,
    x_label: str,
    y_label: str,
    threshold: float | None = None,
    threshold_label: str | None = None,
) -> Path:
    image = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 24), title, fill=(20, 35, 45), font=_font(20))
    if not values.empty:
        y_values = values.to_numpy(dtype=float)
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))
        if threshold is not None and np.isfinite(threshold):
            y_min = min(y_min, threshold)
            y_max = max(y_max, threshold)
        if y_min == y_max:
            padding = max(abs(y_min) * 0.1, 1e-12)
            y_min -= padding
            y_max += padding
        chart = (90, 90, 820, 500)
        draw.rectangle(chart, outline=(20, 35, 45))
        for fraction in (0.25, 0.5, 0.75):
            x_grid = int(chart[0] + fraction * (chart[2] - chart[0]))
            y_grid = int(chart[1] + fraction * (chart[3] - chart[1]))
            draw.line((x_grid, chart[1], x_grid, chart[3]), fill=(225, 230, 235))
            draw.line((chart[0], y_grid, chart[2], y_grid), fill=(225, 230, 235))
        points = []
        for index, value in enumerate(y_values):
            x = _scale(index, 0, max(len(y_values) - 1, 1), chart[0], chart[2])
            y = _scale(float(value), y_min, y_max, chart[3], chart[1])
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=(217, 95, 2), width=3)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(20, 116, 153))
        if threshold is not None and np.isfinite(threshold):
            threshold_y = _scale(threshold, y_min, y_max, chart[3], chart[1])
            draw.line((chart[0], threshold_y, chart[2], threshold_y), fill=(210, 70, 55), width=2)
            draw.line((230, 560, 250, 560), fill=(210, 70, 55), width=2)
            draw.text((258, 552), threshold_label or f"threshold = {threshold:g}", fill=(20, 35, 45), font=_font(11))
        draw.text((360, 535), x_label, fill=(20, 35, 45), font=_font(13))
        draw.text((90, 62), f"Y: {y_label}", fill=(80, 90, 100), font=_font(12))
        draw.text((chart[0], chart[3] + 5), "0", fill=(80, 90, 100), font=_font(11))
        draw.text((chart[2] - 25, chart[3] + 5), str(max(len(y_values) - 1, 0)), fill=(80, 90, 100), font=_font(11))
        draw.text((chart[0] - 62, chart[3] - 6), f"{y_min:.3g}", fill=(80, 90, 100), font=_font(11))
        draw.text((chart[0] - 62, chart[1] - 6), f"{y_max:.3g}", fill=(80, 90, 100), font=_font(11))
        draw.line((25, 560, 45, 560), fill=(217, 95, 2), width=3)
        draw.text((52, 552), "observed", fill=(20, 35, 45), font=_font(11))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _audit_tolerance(table: pd.DataFrame) -> float:
    if "tolerance" not in table or table.empty:
        return 1e-9
    values = pd.to_numeric(table["tolerance"], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else 1e-9


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _blue_red(value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    if not np.isfinite(value):
        return (240, 240, 240)
    t = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.0
    r = int(40 + 200 * t)
    g = int(90 + 80 * (1.0 - abs(t - 0.5) * 2.0))
    b = int(210 - 170 * t)
    return (r, g, b)


def _label_color(label: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (60 + digest[0] % 170, 60 + digest[1] % 170, 60 + digest[2] % 170)


def _scale(value: float, src_min: float, src_max: float, dst_min: int, dst_max: int) -> int:
    if not np.isfinite(value) or src_max == src_min:
        return int((dst_min + dst_max) / 2)
    fraction = (value - src_min) / (src_max - src_min)
    return int(dst_min + fraction * (dst_max - dst_min))


def matrix_genetics_contract_tables(genetic_code: GeneticCode) -> dict[str, pd.DataFrame]:
    """Expose matrix-genetics tables through the audit module for documentation tests."""
    return build_matrix_genetics_tables(genetic_code)
