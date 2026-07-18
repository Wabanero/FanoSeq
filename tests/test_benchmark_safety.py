from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fanoseq.benchmark.feature_sets import (
    FeatureSet,
    _fanoseq_commutators,
    _merge_feature_matrices,
)
from fanoseq.benchmark.models import TrainingFoldFeatureFilter
from fanoseq.benchmark.splits import (
    BenchmarkSplitError,
    make_inner_splits,
    positional_identity,
)
from fanoseq.genetic_code import get_genetic_code
from fanoseq.workflow import CompleteAnalysisConfig, run_complete_analysis


def test_grouped_splits_fail_closed_when_groups_are_missing() -> None:
    with pytest.raises(BenchmarkSplitError, match="requires dataset groups"):
        make_inner_splits(
            np.asarray(["a", "a", "b", "b"]),
            None,
            task="classification",
            requested_folds=2,
            split_strategy="stratified_group",
            random_seed=1,
        )


def test_grouped_splits_fail_closed_for_insufficient_or_confined_groups() -> None:
    with pytest.raises(BenchmarkSplitError, match="only 2 unique groups"):
        make_inner_splits(
            np.asarray(["a", "a", "b", "b"]),
            np.asarray(["g1", "g1", "g2", "g2"]),
            task="classification",
            requested_folds=3,
            split_strategy="stratified_group",
            random_seed=1,
        )
    with pytest.raises(BenchmarkSplitError, match="occurs in only 1 unique group"):
        make_inner_splits(
            np.asarray(["a", "a", "b", "b"]),
            np.asarray(["g1", "g1", "g2", "g3"]),
            task="classification",
            requested_folds=2,
            split_strategy="stratified_group",
            random_seed=1,
        )


def test_explicit_unsafe_split_fallback_is_marked() -> None:
    with pytest.warns(RuntimeWarning, match="UNSAFE split fallback"):
        folds = make_inner_splits(
            np.asarray(["a", "a", "b", "b"]),
            None,
            task="classification",
            requested_folds=2,
            split_strategy="stratified_group",
            random_seed=1,
            allow_unsafe_split_fallback=True,
        )
    assert len(folds) == 2


def test_identity_audit_is_explicitly_positional() -> None:
    assert positional_identity("ACGT", "ACGA") == 0.75
    assert positional_identity("AC", "ACGT") == 0.5


def test_empty_middle_feature_set_keeps_correct_provenance_prefix() -> None:
    first = FeatureSet("first", "x", "", pd.DataFrame({"sequence_id": ["s"], "a": [1]}), ())
    empty = FeatureSet("empty", "x", "", pd.DataFrame(), ())
    third = FeatureSet("third", "x", "", pd.DataFrame({"sequence_id": ["s"], "b": [2]}), ())
    merged = _merge_feature_matrices([first, empty, third])
    assert list(merged.columns) == ["sequence_id", "first__a", "third__b"]


def test_transition_alias_is_not_a_duplicate_benchmark_feature() -> None:
    source = pd.DataFrame(
        {
            "sequence_id": ["s", "s"],
            "commutator_score": [1.0, 2.0],
            "transition_score": [1.0, 2.0],
        }
    )
    feature_set = _fanoseq_commutators({"octonion_products": source})
    assert len(feature_set.feature_columns) == 4
    assert not any("transition" in column for column in feature_set.feature_columns)


def test_training_fold_filter_reports_degenerate_columns() -> None:
    values = np.asarray(
        [
            [1.0, 1.0, 2.0, -2.0, 0.0],
            [2.0, 2.0, 4.0, -4.0, 0.0],
            [3.0, 3.0, 6.0, -6.0, 0.0],
        ]
    )
    fitted = TrainingFoldFeatureFilter().fit(values)
    assert fitted.transform(values).shape == (3, 1)
    assert "exact_duplicate" in fitted.removed_features_[1]
    assert "perfectly_correlated" in fitted.removed_features_[2]
    assert "perfectly_correlated" in fitted.removed_features_[3]
    assert fitted.removed_features_[4] == "constant"


def test_unknown_genetic_code_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown or unsupported genetic code"):
        get_genetic_code("definitely-not-a-code")


def test_complete_analysis_rejects_conflicting_extraction_override(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="one resolved plan"):
        run_complete_analysis(
            CompleteAnalysisConfig(
                input_path=root / "examples" / "benchmark_sequences.fasta",
                benchmark_config=root / "examples" / "benchmark.yaml",
                output_dir=tmp_path,
                window_size=10,
            )
        )


def test_complete_manifest_fixture_shape() -> None:
    """Keep the rich-manifest contract machine readable in this test module."""
    required = {
        "schema_version",
        "git_commit",
        "resolved_analysis_plan",
        "input_hashes",
        "random_seeds",
        "runtime_seconds",
        "peak_traced_memory_bytes",
        "software",
        "table_dimensions",
        "outputs",
        "warnings",
        "biological_evidence_status",
    }
    assert json.loads(json.dumps(sorted(required)))
