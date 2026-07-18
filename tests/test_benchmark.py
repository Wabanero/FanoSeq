from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from fanoseq.benchmark.config import load_benchmark_config
from fanoseq.benchmark.datasets import load_benchmark_dataset
from fanoseq.benchmark.evaluation import run_benchmark_config
from fanoseq.benchmark.feature_sets import build_feature_bundle
from fanoseq.benchmark.null_models import (
    codon_counts,
    codon_order_shuffle,
    dinucleotide_counts,
    dinucleotide_preserving_shuffle,
    is_oriented_fano_automorphism,
    label_permutation,
    mononucleotide_counts,
    mononucleotide_shuffle,
    permute_imaginary_axes,
    synonymous_codon_shuffle,
    translated_sequence,
)
from fanoseq.benchmark.models import default_model_specs, make_pipeline
from fanoseq.cli import app
from fanoseq.genetic_code import get_genetic_code


def test_logistic_regression_supports_multiclass_classification() -> None:
    spec = default_model_specs(
        "classification",
        random_seed=42,
        requested=("logistic_regression",),
    )[0]
    model = make_pipeline(spec)
    features = np.array(
        [[-2.0, -1.0], [-1.8, -1.2], [0.0, 1.0], [0.2, 1.1], [2.0, -0.5], [2.2, -0.7]]
    )
    labels = np.array(["donor", "donor", "acceptor", "acceptor", "neither", "neither"])

    model.fit(features, labels)

    assert set(model.predict(features)) == set(labels)


def test_yaml_config_parsing(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    config = load_benchmark_config(config_path)
    assert config.dataset.table == tmp_path / "metadata.tsv"
    assert config.dataset.task == "classification"
    assert config.features == ("fanoseq_components", "kmer", "real_polynomial_control")
    assert config.evaluation.outer_folds == 2


def test_benchmark_is_deterministic_and_group_leakage_free(tmp_path: Path) -> None:
    config = load_benchmark_config(_write_fixture(tmp_path))
    first = run_benchmark_config(config, tmp_path / "run1")
    second = run_benchmark_config(config, tmp_path / "run2")

    first_metrics = pd.read_csv(first["benchmark_metrics"], sep="\t")
    second_metrics = pd.read_csv(second["benchmark_metrics"], sep="\t")
    primary_first = first_metrics[
        (first_metrics["level"] == "aggregate")
        & (first_metrics["metric_name"] == "balanced_accuracy")
    ][["feature_set", "model", "metric_value"]].reset_index(drop=True)
    primary_second = second_metrics[
        (second_metrics["level"] == "aggregate")
        & (second_metrics["metric_name"] == "balanced_accuracy")
    ][["feature_set", "model", "metric_value"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(primary_first, primary_second)

    folds = pd.read_csv(first["benchmark_folds"], sep="\t")
    for split_id, group in folds.groupby("split_id"):
        train_groups = set(group.loc[group["split"] == "train", "group"])
        test_groups = set(group.loc[group["split"] == "test", "group"])
        assert train_groups.isdisjoint(test_groups), split_id

    leakage = pd.read_csv(first["benchmark_leakage_checks"], sep="\t")
    assert not leakage["group_leakage_detected"].any()
    fold_rows = first_metrics[first_metrics["level"] == "fold"]
    assert fold_rows["best_params_json"].str.contains("model__estimator__C").any()
    assert (tmp_path / "run1" / "benchmark_manifest.json").exists()
    assert (tmp_path / "run1" / "benchmark_report.md").exists()
    assert first["plot"] == tmp_path / "run1" / "benchmark_multipanel.png"
    assert first["plot"].stat().st_size > 0


def test_ablation_feature_sets_are_incremental(tmp_path: Path) -> None:
    config = load_benchmark_config(_write_fixture(tmp_path))
    dataset = load_benchmark_dataset(config.dataset)
    bundle = build_feature_bundle(config, dataset, tmp_path / "features")
    assert "ablation_base_descriptors" in bundle.feature_sets
    assert "ablation_plus_octonion_products" in bundle.feature_sets
    assert (
        len(bundle.feature_sets["ablation_plus_octonion_products"].feature_columns)
        > len(bundle.feature_sets["ablation_base_descriptors"].feature_columns)
    )


def test_antisymmetric_and_randomized_fano_controls_are_available(tmp_path: Path) -> None:
    config = load_benchmark_config(
        _write_fixture(
            tmp_path,
            features=["antisymmetric_control", "randomized_fano_structure"],
        )
    )
    dataset = load_benchmark_dataset(config.dataset)
    bundle = build_feature_bundle(config, dataset, tmp_path / "features")

    antisymmetric = bundle.feature_sets["antisymmetric_control"]
    randomized = bundle.feature_sets["randomized_fano_structure"]
    assert any(
        column.startswith("antisymmetric_e1_e2") for column in antisymmetric.feature_columns
    )
    assert any(column.startswith("random_fano_e1") for column in randomized.feature_columns)
    assert len(randomized.feature_columns) < len(antisymmetric.feature_columns)


def test_sequence_and_label_null_models_preserve_expected_quantities() -> None:
    rng = np.random.default_rng(7)
    sequence = "ATGATGCCCTTT"
    mono = mononucleotide_shuffle(sequence, rng)
    assert mononucleotide_counts(mono) == mononucleotide_counts(sequence)

    dinuc = dinucleotide_preserving_shuffle(sequence, rng)
    assert dinucleotide_counts(dinuc) == dinucleotide_counts(sequence)

    codon = codon_order_shuffle(sequence, rng, frame=0)
    assert codon_counts(codon) == codon_counts(sequence)

    genetic_code = get_genetic_code("standard")
    synonymous = synonymous_codon_shuffle(sequence, genetic_code, rng, frame=0)
    assert translated_sequence(synonymous, genetic_code) == translated_sequence(
        sequence,
        genetic_code,
    )

    labels = np.asarray(["a", "a", "b", "b"])
    permuted = label_permutation(labels, rng)
    assert Counter(permuted) == Counter(labels)


def test_axis_permutations_are_not_all_automorphisms() -> None:
    assert is_oriented_fano_automorphism((1, 2, 3, 4, 5, 6, 7))
    assert not is_oriented_fano_automorphism((2, 1, 3, 4, 5, 6, 7))

    table = pd.DataFrame(
        {
            "e1": [1.0],
            "e2": [2.0],
            "e3": [3.0],
            "e4": [4.0],
            "e5": [5.0],
            "e6": [6.0],
            "e7": [7.0],
        }
    )
    transformed = permute_imaginary_axes(table, (2, 1, 3, 4, 5, 6, 7))
    assert transformed.loc[0, "e1"] == 2.0
    assert transformed.loc[0, "e2"] == 1.0


def test_dataset_validation_errors(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    config = load_benchmark_config(config_path)
    metadata = pd.read_csv(config.dataset.table, sep="\t")
    duplicate = pd.concat([metadata, metadata.iloc[[0]]], ignore_index=True)
    duplicate.to_csv(config.dataset.table, sep="\t", index=False)
    with pytest.raises(ValueError, match="Duplicate sequence IDs"):
        load_benchmark_dataset(config.dataset)

    config_path = _write_fixture(tmp_path)
    config = load_benchmark_config(config_path)
    metadata = pd.read_csv(config.dataset.table, sep="\t")
    metadata.loc[0, "label"] = np.nan
    metadata.to_csv(config.dataset.table, sep="\t", index=False)
    with pytest.raises(ValueError, match="Missing labels"):
        load_benchmark_dataset(config.dataset)

    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(
        "dataset:\n"
        "  table: metadata.tsv\n"
        "  fasta: sequences.fasta\n"
        "  task: unsupported\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported benchmark task"):
        load_benchmark_config(bad_config)


def test_parquet_output(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    config_path = _write_fixture(tmp_path, output_format="parquet")
    outputs = run_benchmark_config(load_benchmark_config(config_path), tmp_path / "parquet_run")
    assert outputs["benchmark_metrics"].suffix == ".parquet"
    metrics = pd.read_parquet(outputs["benchmark_metrics"])
    assert "balanced_accuracy" in set(metrics["metric_name"])


def test_benchmark_cli_smoke(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["benchmark", "--config", str(config_path), "--output-dir", str(tmp_path / "cli_run")],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "cli_run" / "benchmark_metrics.tsv").exists()


def test_null_models_are_evaluated_without_placeholder_metrics(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    config_path = tmp_path / "benchmark.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["features"] = ["fanoseq_components", "randomized_fano_structure"]
    payload["evaluation"]["run_ablations"] = False
    payload["null_models"] = {
        "sequence_nulls": ["mononucleotide_shuffle"],
        "sequence_null_repeats": 1,
        "representation_nulls": [
            "remove_scalar_e0",
            "imaginary_axis_permutation",
            "random_antisymmetric_tensor",
        ],
        "axis_permutation_repeats": 2,
        "random_tensor_repeats": 2,
        "randomized_fano_repeats": 2,
        "random_seed": 71,
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    outputs = run_benchmark_config(
        load_benchmark_config(config_path), tmp_path / "null_run"
    )
    nulls = pd.read_csv(outputs["benchmark_null_results"], sep="\t")

    assert set(nulls["null_model"]) == {
        "mononucleotide_shuffle",
        "remove_scalar_e0",
        "imaginary_axis_permutation",
        "random_antisymmetric_tensor",
        "randomized_fano_structure",
    }
    assert nulls["metric_value"].notna().all()
    assert nulls.groupby("null_model")["null_iteration"].nunique().to_dict()[
        "imaginary_axis_permutation"
    ] == 2
    assert nulls.groupby("null_model")["null_iteration"].nunique().to_dict()[
        "random_antisymmetric_tensor"
    ] == 2
    assert nulls.groupby("null_model")["null_iteration"].nunique().to_dict()[
        "randomized_fano_structure"
    ] == 2
    assert (tmp_path / "null_run" / "_null_models" / "mononucleotide_shuffle_000" / "sequences.fasta").exists()
    assert all(json.loads(value)["status"] == "evaluated" for value in nulls["metadata_json"])


def _write_fixture(
    tmp_path: Path,
    *,
    output_format: str = "tsv",
    features: list[str] | None = None,
) -> Path:
    metadata = pd.DataFrame(
        {
            "sequence_id": [
                "seq_pos_1",
                "seq_pos_2",
                "seq_pos_3",
                "seq_pos_4",
                "seq_neg_1",
                "seq_neg_2",
                "seq_neg_3",
                "seq_neg_4",
            ],
            "label": ["pos", "pos", "pos", "pos", "neg", "neg", "neg", "neg"],
            "subject": ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"],
        }
    )
    metadata.to_csv(tmp_path / "metadata.tsv", sep="\t", index=False)
    (tmp_path / "sequences.fasta").write_text(
        ">seq_pos_1\nATGGCCATGGCCATGGCC\n"
        ">seq_pos_2\nATGAAAGCCATGAAAGCC\n"
        ">seq_pos_3\nGCCATGGCCATGGCCATG\n"
        ">seq_pos_4\nATGGGGATGGGGATGGGG\n"
        ">seq_neg_1\nATATATATATATATATAT\n"
        ">seq_neg_2\nTTAATTAATTAATTAATT\n"
        ">seq_neg_3\nACACACACACACACACAC\n"
        ">seq_neg_4\nTCTCTCTCTCTCTCTCTC\n",
        encoding="utf-8",
    )
    config = {
        "dataset": {
            "table": "metadata.tsv",
            "fasta": "sequences.fasta",
            "id_column": "sequence_id",
            "target_column": "label",
            "group_column": "subject",
            "task": "classification",
            "seq_type": "dna",
        },
        "features": features or ["fanoseq_components", "kmer", "real_polynomial_control"],
        "feature_extraction": {"window_size": 9, "step": 3, "kmer_k": 2, "frame": 0},
        "evaluation": {
            "outer_folds": 2,
            "inner_folds": 2,
            "repeats": 1,
            "random_seed": 11,
            "split_strategy": "stratified_group",
            "primary_metric": "balanced_accuracy",
            "models": ["logistic_regression"],
            "output_format": output_format,
            "paired_permutation_rounds": 19,
        },
    }
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(_simple_yaml(config), encoding="utf-8")
    (tmp_path / "benchmark.json").write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _simple_yaml(payload: dict[str, object], indent: int = 0) -> str:
    lines: list[str] = []
    prefix = " " * indent
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_simple_yaml(value, indent + 2))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            lines.extend(f"{prefix}  - {item}" for item in value)
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines)
