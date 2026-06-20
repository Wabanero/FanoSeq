from pathlib import Path

import pandas as pd
import pytest

from fanoseq.pipeline import RunConfig, run_analysis


def test_pipeline_window_modes_on_examples(tmp_path: Path) -> None:
    dna_out = tmp_path / "dna_windows"
    outputs = run_analysis(
        RunConfig(
            input_path=Path("examples/example_dna.fasta"),
            seq_type="dna",
            mode="window",
            window_size=10,
            step=1,
            output_dir=dna_out,
        )
    )
    for filename in (
        "window_octonions.tsv",
        "octonion_products.tsv",
        "octonion_triplets.tsv",
        "fano_interactions.tsv",
    ):
        assert (dna_out / filename).exists()
        assert filename in outputs
    assert {"sequence_id", "e0", "mono_entropy"}.issubset(outputs["window_octonions.tsv"].columns)
    assert len(outputs["fano_interactions.tsv"]) == 7 * len(outputs["octonion_products.tsv"])

    protein_out = tmp_path / "protein_windows"
    protein_outputs = run_analysis(
        RunConfig(
            input_path=Path("examples/example_protein.fasta"),
            seq_type="protein",
            mode="window",
            window_size=15,
            step=1,
            output_dir=protein_out,
        )
    )
    assert (protein_out / "window_octonions.tsv").exists()
    assert protein_outputs["window_octonions.tsv"]["gc_content"].isna().all()


def test_pipeline_codon_and_both_modes_on_examples(tmp_path: Path) -> None:
    codon_out = tmp_path / "dna_codons"
    codon_outputs = run_analysis(
        RunConfig(
            input_path=Path("examples/example_dna.fasta"),
            seq_type="dna",
            mode="codon",
            frame=0,
            output_dir=codon_out,
        )
    )
    for filename in (
        "codon_octonions.tsv",
        "codon_transition_products.tsv",
        "codon_usage_fano_features.tsv",
        "codon_usage_sequence_summary.tsv",
        "fano_interactions.tsv",
    ):
        assert (codon_out / filename).exists()
        assert filename in codon_outputs
    assert {"codon", "amino_acid", "codon_associator_score", "e7"}.issubset(
        codon_outputs["codon_octonions.tsv"].columns
    )
    assert len(codon_outputs["fano_interactions.tsv"]) == 7 * len(
        codon_outputs["codon_transition_products.tsv"]
    )

    both_out = tmp_path / "dna_both"
    both_outputs = run_analysis(
        RunConfig(
            input_path=Path("examples/example_dna.fasta"),
            seq_type="dna",
            mode="both",
            frame="all",
            window_size=10,
            step=1,
            output_dir=both_out,
        )
    )
    assert "window_octonions.tsv" in both_outputs
    assert "codon_octonions.tsv" in both_outputs
    assert set(both_outputs["codon_octonions.tsv"]["frame"]) == {0, 1, 2}


def test_no_products_across_skipped_invalid_windows_or_codons(tmp_path: Path) -> None:
    dna_path = tmp_path / "invalid_windows.fasta"
    dna_path.write_text(">s\nACGTNNNNACGT\n", encoding="utf-8")
    outputs = run_analysis(
        RunConfig(
            input_path=dna_path,
            seq_type="dna",
            mode="window",
            window_size=4,
            step=4,
            output_dir=tmp_path / "invalid_window_out",
        )
    )
    assert list(outputs["window_octonions.tsv"]["position"]) == [0, 2]
    assert outputs["octonion_products.tsv"].empty

    codon_path = tmp_path / "invalid_codons.fasta"
    codon_path.write_text(">s\nATGNNNTAA\n", encoding="utf-8")
    codon_outputs = run_analysis(
        RunConfig(
            input_path=codon_path,
            seq_type="dna",
            mode="codon",
            frame=0,
            output_dir=tmp_path / "invalid_codon_out",
        )
    )
    assert list(codon_outputs["codon_octonions.tsv"]["codon_index"]) == [0, 2]
    assert codon_outputs["codon_transition_products.tsv"].empty


def test_parquet_output_format(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    output_dir = tmp_path / "parquet"
    outputs = run_analysis(
        RunConfig(
            input_path=Path("examples/example_dna.fasta"),
            seq_type="dna",
            mode="window",
            window_size=10,
            step=1,
            output_dir=output_dir,
            output_format="parquet",
        )
    )
    assert "window_octonions.parquet" in outputs
    assert (output_dir / "window_octonions.parquet").exists()
    loaded = pd.read_parquet(output_dir / "window_octonions.parquet")
    assert {"sequence_id", "e0", "mono_entropy"}.issubset(loaded.columns)


def test_bundle_output_manifest_and_partitioned_tables(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    output_dir = tmp_path / "example.fanoseq"
    outputs = run_analysis(
        RunConfig(
            input_path=Path("examples/example_dna.fasta"),
            seq_type="dna",
            mode="codon",
            frame=0,
            output_dir=output_dir,
            output_format="bundle",
        )
    )
    assert "codon_octonions.parquet" in outputs
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "codon_octonions.parquet").exists()
    manifest_text = (output_dir / "manifest.json").read_text(encoding="utf-8")
    assert '"schema_version"' in manifest_text
    loaded = pd.read_parquet(output_dir / "codon_octonions.parquet")
    assert "codon" in loaded.columns


def test_summary_only_and_top_k_transitions(tmp_path: Path) -> None:
    summary_dir = tmp_path / "summary"
    summary_outputs = run_analysis(
        RunConfig(
            input_path=Path("examples/example_dna.fasta"),
            seq_type="dna",
            mode="window",
            window_size=10,
            step=1,
            output_dir=summary_dir,
            summary_only=True,
        )
    )
    assert "window_sequence_summary.tsv" in summary_outputs
    assert (summary_dir / "window_sequence_summary.tsv").exists()
    assert not (summary_dir / "window_octonions.tsv").exists()

    topk_dir = tmp_path / "topk"
    topk_outputs = run_analysis(
        RunConfig(
            input_path=Path("examples/example_dna.fasta"),
            seq_type="dna",
            mode="window",
            window_size=10,
            step=1,
            output_dir=topk_dir,
            top_k_transitions=1,
        )
    )
    assert len(topk_outputs["octonion_products.tsv"]) == 1
    assert len(topk_outputs["fano_interactions.tsv"]) == 7
