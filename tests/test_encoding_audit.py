from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

from fanoseq.dna_features import DNA_ALPHABET, encode_dna_window, reverse_complement
from fanoseq.encoding_audit import (
    EncodingAuditConfig,
    build_axis_control_tables,
    build_codon_audit_tables,
    build_encoding_audit_tables,
    build_feature_redundancy_tables,
    dinucleotide_preserving_shuffle,
    dna_perturbations,
    plot_encoding_audit_outputs,
    synonymous_recoding,
    transform_octonion_rc,
    translate_dna,
)
from fanoseq.fasta import FastaRecord
from fanoseq.genetic_code import get_genetic_code


def test_dna_window_reverse_complement_transform_exhaustive_small() -> None:
    for length in range(1, 6):
        for bases in product(DNA_ALPHABET, repeat=length):
            sequence = "".join(bases)
            encoded = encode_dna_window(sequence)
            encoded_rc = encode_dna_window(reverse_complement(sequence))
            assert encoded is not None and encoded_rc is not None
            transformed = transform_octonion_rc(encoded[0], "dna-window-v1")
            assert np.allclose(transformed.components, encoded_rc[0].components)


def test_codon_audit_catalog_is_complete_and_injective() -> None:
    tables = build_codon_audit_tables(get_genetic_code("standard"))
    catalog = tables["codon_octonion_catalog"]
    collisions = tables["codon_collision_report"]
    spectrum = tables["codon_geometry_rank_spectrum"]

    assert len(catalog) == 64
    assert collisions.empty
    assert int(spectrum["above_tolerance"].sum()) == 8
    assert {"right_e0", "associator_e7", "left_right_product_distance"}.issubset(catalog.columns)

    atg = catalog[catalog["codon"] == "ATG"].iloc[0]
    associator = np.array([atg[f"associator_e{i}"] for i in range(8)], dtype=float)
    assert np.isclose(atg["left_right_product_distance"], np.linalg.norm(associator))


def test_shuffle_and_synonymous_recoding_correctness() -> None:
    sequence = "ATGGCCTTACGATTA"
    shuffled = dinucleotide_preserving_shuffle(sequence, random_seed=4)
    assert Counter(sequence[index : index + 2] for index in range(len(sequence) - 1)) == Counter(
        shuffled[index : index + 2] for index in range(len(shuffled) - 1)
    )

    code = get_genetic_code("standard")
    recoded = synonymous_recoding("ATGGCCGCTTAA", code)
    assert translate_dna(recoded, code) == translate_dna("ATGGCCGCTTAA", code)
    assert any(row["perturbation_type"] == "reverse_complement" for row in dna_perturbations("ACGT"))


def test_feature_redundancy_reports_commutator_cross_product_identity(tmp_path: Path) -> None:
    fasta = tmp_path / "seq.fasta"
    fasta.write_text(">s\nACGTAGCTTACGATCG\n", encoding="utf-8")
    config = EncodingAuditConfig(
        input_path=fasta,
        seq_type="dna",
        axis_scheme_id="dna-window-v1",
        checks=("redundancy",),
        window_size=4,
    )
    tables = build_feature_redundancy_tables(
        [FastaRecord(id="s", description="s", sequence="ACGTAGCTTACGATCG")],
        config,
    )
    redundancy = tables["feature_redundancy"]
    identity = redundancy[
        redundancy["feature_family"] == "commutator_vs_2x_fano_cross_product"
    ].iloc[0]
    assert identity["max_abs_identity_residual"] <= 1e-9
    assert "real_antisymmetric_control" in set(redundancy["feature_family"])


def test_axis_controls_distinguish_automorphisms_from_arbitrary_permutations(tmp_path: Path) -> None:
    fasta = tmp_path / "seq.fasta"
    fasta.write_text(">s\nACGTAGCTTACGATCG\n", encoding="utf-8")
    config = EncodingAuditConfig(
        input_path=fasta,
        seq_type="dna",
        axis_scheme_id="dna-window-v1",
        checks=("permutation",),
        window_size=4,
        permutation_samples=2,
    )
    tables = build_axis_control_tables(
        [FastaRecord(id="s", description="s", sequence="ACGTAGCTTACGATCG")],
        config,
    )
    controls = tables["fano_automorphism_controls"]
    assert controls[controls["transformation_category"] == "fano_plane_automorphism"][
        "preserves_multiplication_table"
    ].all()
    assert (
        controls["transformation_category"] == "random_antisymmetric_multiplication_tensor"
    ).any()


def test_encoding_audit_table_schemas(tmp_path: Path) -> None:
    fasta = tmp_path / "seq.fasta"
    fasta.write_text(">s\nACGTAGCTTACGATCG\n", encoding="utf-8")
    config = EncodingAuditConfig(
        input_path=fasta,
        seq_type="dna",
        axis_scheme_id="dna-window-v1",
        checks=("reverse-complement", "codon", "mutation", "permutation"),
        window_size=4,
        permutation_samples=2,
        max_perturbations=12,
    )
    tables = build_encoding_audit_tables(config)
    expected = {
        "encoding_audit_summary",
        "encoding_contracts",
        "reverse_complement_audit",
        "codon_octonion_catalog",
        "codon_collision_report",
        "codon_distance_matrix",
        "codon_synonymy_statistics",
        "mutation_sensitivity",
        "axis_permutation_stability",
        "fano_automorphism_controls",
    }
    assert expected.issubset(tables)
    for table in expected:
        assert {"software_version", "schema_version", "input_hash", "tolerance"}.issubset(
            tables[table].columns
        )
    plot_paths = plot_encoding_audit_outputs(tables, tmp_path / "plots")
    multipanel = tmp_path / "plots" / "encoding_audit_multipanel.png"
    assert multipanel in plot_paths
    assert multipanel.stat().st_size > 0
