"""Benchmark orchestration and leakage-safe model evaluation."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer, StandardScaler

import fanoseq
from fanoseq.benchmark.config import BenchmarkConfig, load_benchmark_config
from fanoseq.benchmark.datasets import (
    BenchmarkDataset,
    dataset_composition,
    input_hashes,
    load_benchmark_dataset,
)
from fanoseq.benchmark.feature_sets import (
    FeatureBundle,
    FeatureSet,
    build_feature_bundle,
    feature_quality_table,
    feature_set_table,
    matrix_for_ids,
    randomized_fano_control_feature_set,
    strongest_conventional_candidates,
)
from fanoseq.benchmark.models import (
    adapt_param_grid,
    default_model_specs,
    make_pipeline,
    scoring_name,
)
from fanoseq.benchmark.null_models import (
    codon_order_shuffle,
    dinucleotide_preserving_shuffle,
    is_oriented_fano_automorphism,
    label_permutation,
    mononucleotide_shuffle,
    random_antisymmetric_tensor,
    remove_scalar_component,
    synonymous_codon_shuffle,
)
from fanoseq.benchmark.reporting import write_markdown_report
from fanoseq.benchmark.splits import (
    FoldSpec,
    fold_assignment_table,
    leakage_audit_table,
    make_inner_splits,
    make_outer_folds,
)
from fanoseq.benchmark.statistics import (
    ablation_results_table,
    aggregate_metric_rows,
    paired_comparison_table,
)
from fanoseq.plots import plot_benchmark_multipanel
from fanoseq.io import write_outputs
from fanoseq.fasta import FastaRecord
from fanoseq.genetic_code import get_genetic_code

SCHEMA_VERSION = "1.0.0"


def run_benchmark(config_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Run a manifest-driven benchmark and write output tables, manifest, and report."""
    config = load_benchmark_config(config_path)
    return run_benchmark_config(config, output_dir)


def run_benchmark_config(config: BenchmarkConfig, output_dir: str | Path) -> dict[str, Path]:
    """Run a benchmark from an already parsed config."""
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id(config)
    dataset = load_benchmark_dataset(config.dataset)
    feature_bundle = build_feature_bundle(config, dataset, base_dir)
    folds = make_outer_folds(dataset, config.evaluation)
    fold_table = fold_assignment_table(dataset, folds)
    leakage_table = leakage_audit_table(
        dataset,
        folds,
        sequence_similarity=config.evaluation.sequence_similarity_audit,
        similarity_threshold=config.evaluation.sequence_similarity_threshold,
    )

    if dataset.task == "clustering":
        fold_metrics, predictions = _evaluate_clustering(
            run_id,
            dataset,
            feature_bundle,
            folds,
            config,
        )
    else:
        fold_metrics, predictions = _evaluate_supervised(
            run_id,
            dataset,
            feature_bundle,
            folds,
            config,
        )
    aggregate_metrics = aggregate_metric_rows(fold_metrics)
    metrics = pd.concat([fold_metrics, aggregate_metrics], ignore_index=True)

    conventional_candidates = strongest_conventional_candidates(feature_bundle.feature_sets)
    conventional = (
        {config.evaluation.primary_comparator}
        if config.evaluation.primary_comparator in conventional_candidates
        else set()
    )
    comparisons = paired_comparison_table(
        fold_metrics,
        conventional_feature_sets=conventional,
        primary_metric=config.evaluation.primary_metric,
        random_seed=config.evaluation.random_seed,
        n_rounds=config.evaluation.paired_permutation_rounds,
    )
    ablations = ablation_results_table(
        aggregate_metrics,
        primary_metric=config.evaluation.primary_metric,
    )
    null_results = _evaluate_null_models(
        run_id,
        dataset,
        feature_bundle,
        folds,
        config,
    )
    resolved_config = pd.DataFrame(
        [
            {"key": key, "value_json": json.dumps(value, sort_keys=True)}
            for key, value in config.to_dict().items()
        ]
    )
    runs = _run_table(run_id, config, dataset)
    tables = {
        "benchmark_runs": runs,
        "benchmark_folds": fold_table,
        "benchmark_leakage_checks": leakage_table,
        "benchmark_metrics": metrics,
        "benchmark_predictions": predictions,
        "benchmark_feature_sets": feature_set_table(feature_bundle.feature_sets),
        "benchmark_feature_quality": feature_quality_table(feature_bundle.feature_sets),
        "benchmark_ablation_results": ablations,
        "benchmark_null_results": null_results,
        "benchmark_permutation_tests": comparisons,
        "benchmark_config_resolved": resolved_config,
    }
    written_relative = write_outputs(
        tables,
        base_dir,
        config.evaluation.output_format,
        manifest=None,
    )
    manifest_path = _write_manifest(
        base_dir,
        run_id=run_id,
        config=config,
        dataset=dataset,
        folds=fold_table,
        feature_bundle=feature_bundle,
        output_paths=written_relative,
    )
    report_path = write_markdown_report(
        base_dir / "benchmark_report.md",
        run_id=run_id,
        dataset=dataset,
        fold_table=fold_table,
        leakage_table=leakage_table,
        metrics=metrics,
        comparisons=comparisons,
        null_results=null_results,
        ablation_results=ablations,
        primary_metric=config.evaluation.primary_metric,
    )
    plot_path = plot_benchmark_multipanel(
        tables,
        base_dir / "benchmark_multipanel.png",
        config.evaluation.primary_metric,
    )
    return {
        **{name: base_dir / relative for name, relative in written_relative.items()},
        "manifest": manifest_path,
        "report": report_path,
        "plot": plot_path,
    }


def _evaluate_supervised(
    run_id: str,
    dataset: BenchmarkDataset,
    feature_bundle: FeatureBundle,
    folds: list[FoldSpec],
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset.y is None:
        raise ValueError("Supervised evaluation requires labels.")
    specs = default_model_specs(
        dataset.task,
        random_seed=config.evaluation.random_seed,
        requested=config.evaluation.models,
    )
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for feature_set in feature_bundle.feature_sets.values():
        matrix = matrix_for_ids(feature_set, dataset.sequence_ids)
        x = matrix.drop(columns=["sequence_id"]).to_numpy(dtype=float)
        for spec in specs:
            for fold in folds:
                train_idx = fold.train_indices
                test_idx = fold.test_indices
                y_train = dataset.y[train_idx]
                y_test = dataset.y[test_idx]
                estimator, best_params = _fit_nested_model(
                    spec,
                    x[train_idx],
                    y_train,
                    groups_train=_subset(dataset.groups, train_idx),
                    task=dataset.task,
                    config=config,
                    fold=fold,
                )
                y_pred = estimator.predict(x[test_idx])
                scores = _prediction_scores(estimator, x[test_idx])
                metrics = _supervised_metrics(
                    y_test,
                    y_pred,
                    scores,
                    task=dataset.task,
                    primary_metric=config.evaluation.primary_metric,
                )
                confusion_json = _confusion_json(y_test, y_pred, dataset.task)
                for metric_name, metric_value in metrics.items():
                    metric_rows.append(
                        _metric_row(
                            run_id=run_id,
                            feature_set=feature_set.name,
                            model=spec.name,
                            fold=fold,
                            metric_name=metric_name,
                            metric_value=metric_value,
                            n_train=len(train_idx),
                            n_test=len(test_idx),
                            best_params=best_params,
                            metadata={"confusion_matrix": confusion_json},
                        )
                    )
                for local_index, sequence_index in enumerate(test_idx):
                    prediction_rows.append(
                        {
                            "run_id": run_id,
                            "feature_set": feature_set.name,
                            "model": spec.name,
                            "repeat": fold.repeat,
                            "fold": fold.fold,
                            "split_id": fold.split_id,
                            "sequence_id": dataset.sequence_ids[int(sequence_index)],
                            "y_true": y_test[local_index],
                            "y_pred": y_pred[local_index],
                            "score_json": json.dumps(_score_payload(scores, local_index)),
                        }
                    )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def _evaluate_clustering(
    run_id: str,
    dataset: BenchmarkDataset,
    feature_bundle: FeatureBundle,
    folds: list[FoldSpec],
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    n_clusters = _cluster_count(dataset)
    for feature_set in feature_bundle.feature_sets.values():
        matrix = matrix_for_ids(feature_set, dataset.sequence_ids)
        x = matrix.drop(columns=["sequence_id"]).to_numpy(dtype=float)
        for fold in folds:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[fold.train_indices])
            x_test = scaler.transform(x[fold.test_indices])
            model = KMeans(
                n_clusters=min(n_clusters, len(fold.train_indices)),
                random_state=config.evaluation.random_seed + fold.repeat,
                n_init=10,
            )
            model.fit(x_train)
            labels = model.predict(x_test)
            metrics = _clustering_metrics(dataset, fold, x_test, labels)
            for metric_name, metric_value in metrics.items():
                metric_rows.append(
                    _metric_row(
                        run_id=run_id,
                        feature_set=feature_set.name,
                        model="kmeans",
                        fold=fold,
                        metric_name=metric_name,
                        metric_value=metric_value,
                        n_train=len(fold.train_indices),
                        n_test=len(fold.test_indices),
                        best_params={"n_clusters": int(model.n_clusters)},
                        metadata={},
                    )
                )
            for local_index, sequence_index in enumerate(fold.test_indices):
                prediction_rows.append(
                    {
                        "run_id": run_id,
                        "feature_set": feature_set.name,
                        "model": "kmeans",
                        "repeat": fold.repeat,
                        "fold": fold.fold,
                        "split_id": fold.split_id,
                        "sequence_id": dataset.sequence_ids[int(sequence_index)],
                        "y_true": dataset.y[int(sequence_index)] if dataset.y is not None else "NA",
                        "y_pred": int(labels[local_index]),
                        "score_json": "{}",
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def _fit_nested_model(
    spec: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    groups_train: np.ndarray | None,
    task: str,
    config: BenchmarkConfig,
    fold: FoldSpec,
) -> tuple[Any, dict[str, object]]:
    pipeline = make_pipeline(spec, feature_selection=config.evaluation.feature_selection)
    param_grid = adapt_param_grid(spec, n_train=len(x_train))
    inner_splits = make_inner_splits(
        y_train,
        groups_train,
        task=task,
        requested_folds=config.evaluation.inner_folds,
        split_strategy=config.evaluation.split_strategy,
        random_seed=config.evaluation.random_seed + 1000 + fold.repeat * 31 + fold.fold,
        allow_unsafe_split_fallback=config.evaluation.allow_unsafe_split_fallback,
    )
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=inner_splits,
        scoring=scoring_name(config.evaluation.primary_metric, task),
        n_jobs=config.evaluation.n_jobs,
        error_score=np.nan,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, dict(search.best_params_)


def _evaluate_null_models(
    run_id: str,
    dataset: BenchmarkDataset,
    feature_bundle: FeatureBundle,
    folds: list[FoldSpec],
    config: BenchmarkConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(
        config.null_models.random_seed
        if config.null_models.random_seed is not None
        else config.evaluation.random_seed + 5000
    )
    if config.null_models.label_permutations > 0 and dataset.y is not None:
        primary_feature = next(iter(feature_bundle.feature_sets.values()))
        limited_config = _config_with_models(config, limit_models=1)
        for permutation_index in range(config.null_models.label_permutations):
            permuted_dataset = _dataset_with_y(dataset, label_permutation(dataset.y, rng))
            limited_bundle = FeatureBundle(
                feature_sets={primary_feature.name: primary_feature},
                fano_tables=feature_bundle.fano_tables,
                baseline_tables=feature_bundle.baseline_tables,
                cache_dir=feature_bundle.cache_dir,
            )
            fold_metrics, _ = _evaluate_supervised(
                run_id,
                permuted_dataset,
                limited_bundle,
                folds,
                limited_config,
            )
            primary = fold_metrics[fold_metrics["metric_name"] == config.evaluation.primary_metric]
            rows.append(
                {
                    "run_id": run_id,
                    "null_model": "label_permutation",
                    "null_iteration": permutation_index,
                    "feature_set": primary_feature.name,
                    "metric_name": config.evaluation.primary_metric,
                    "metric_value": (
                        float(primary["metric_value"].mean()) if not primary.empty else np.nan
                    ),
                    "metadata_json": "{}",
                }
            )
    if (
        config.null_models.randomized_fano_repeats > 1
        and "randomized_fano_structure" in config.features
    ):
        base_seed = (
            config.null_models.random_seed
            if config.null_models.random_seed is not None
            else config.evaluation.random_seed + 7000
        )
        for iteration in range(config.null_models.randomized_fano_repeats):
            seed = base_seed + iteration
            feature_set = randomized_fano_control_feature_set(
                feature_bundle.fano_tables,
                random_seed=seed,
            )
            _evaluate_and_append_null_bundle(
                rows,
                run_id=run_id,
                null_name="randomized_fano_structure",
                iteration=iteration,
                dataset=dataset,
                feature_sets={feature_set.name: feature_set},
                feature_bundle=feature_bundle,
                folds=folds,
                config=config,
                metadata={"status": "evaluated", "random_seed": seed},
            )
    if "remove_scalar_e0" in config.null_models.representation_nulls:
        transformed = {
            name: FeatureSet(
                name=f"{name}__remove_scalar_e0",
                family="representation_null",
                description="FanoSeq representation with scalar e0-linked columns removed.",
                matrix=remove_scalar_component(feature_set.matrix),
                source_tables=feature_set.source_tables,
            )
            for name, feature_set in feature_bundle.feature_sets.items()
            if name.startswith("fanoseq")
            and len(remove_scalar_component(feature_set.matrix).columns) > 1
        }
        _evaluate_and_append_null_bundle(
            rows,
            run_id=run_id,
            null_name="remove_scalar_e0",
            iteration=0,
            dataset=dataset,
            feature_sets=transformed,
            feature_bundle=feature_bundle,
            folds=folds,
            config=config,
            metadata={"status": "evaluated", "transform_scope": "training inputs"},
        )

    if "imaginary_axis_permutation" in config.null_models.representation_nulls:
        for iteration in range(config.null_models.axis_permutation_repeats):
            permutation = tuple(int(value) for value in rng.permutation(np.arange(1, 8)))
            transformed = {}
            for name, feature_set in feature_bundle.feature_sets.items():
                if not name.startswith("fanoseq"):
                    continue
                matrix, changed = _permute_axis_bearing_features(feature_set.matrix, permutation)
                if changed:
                    transformed[name] = FeatureSet(
                        name=f"{name}__axis_permutation",
                        family="representation_null",
                        description="Imaginary-axis coordinate permutation control.",
                        matrix=matrix,
                        source_tables=feature_set.source_tables,
                    )
            _evaluate_and_append_null_bundle(
                rows,
                run_id=run_id,
                null_name="imaginary_axis_permutation",
                iteration=iteration,
                dataset=dataset,
                feature_sets=transformed,
                feature_bundle=feature_bundle,
                folds=folds,
                config=config,
                metadata={
                    "status": "evaluated",
                    "permutation": permutation,
                    "is_oriented_fano_automorphism": is_oriented_fano_automorphism(permutation),
                },
            )

    if "random_antisymmetric_tensor" in config.null_models.representation_nulls:
        for iteration in range(config.null_models.random_tensor_repeats):
            tensor = random_antisymmetric_tensor(rng)
            feature_set = _random_tensor_feature_set(feature_bundle.fano_tables, tensor)
            _evaluate_and_append_null_bundle(
                rows,
                run_id=run_id,
                null_name="random_antisymmetric_tensor",
                iteration=iteration,
                dataset=dataset,
                feature_sets={feature_set.name: feature_set},
                feature_bundle=feature_bundle,
                folds=folds,
                config=config,
                metadata={
                    "status": "evaluated",
                    "tensor_sha256": hashlib.sha256(tensor.tobytes()).hexdigest(),
                },
            )

    for null_name in config.null_models.sequence_nulls:
        for iteration in range(config.null_models.sequence_null_repeats):
            null_dataset, preservation = _sequence_null_dataset(
                dataset,
                null_name,
                config,
                rng,
            )
            null_dir = (
                feature_bundle.cache_dir.parent
                / "_null_models"
                / f"{null_name}_{iteration:03d}"
            )
            null_fasta = null_dir / "sequences.fasta"
            _write_fasta(null_dataset.records, null_fasta)
            null_config = replace(
                config,
                dataset=replace(config.dataset, fasta=null_fasta),
            )
            null_bundle = build_feature_bundle(null_config, null_dataset, null_dir)
            _evaluate_and_append_null_bundle(
                rows,
                run_id=run_id,
                null_name=null_name,
                iteration=iteration,
                dataset=null_dataset,
                feature_sets=null_bundle.feature_sets,
                feature_bundle=null_bundle,
                folds=folds,
                config=config,
                metadata={"status": "evaluated", **preservation},
            )
    return pd.DataFrame(
        rows,
        columns=[
            "run_id",
            "null_model",
            "null_iteration",
            "feature_set",
            "model",
            "metric_name",
            "metric_value",
            "metric_std",
            "n_folds",
            "metadata_json",
        ],
    )


def _evaluate_and_append_null_bundle(
    rows: list[dict[str, object]],
    *,
    run_id: str,
    null_name: str,
    iteration: int,
    dataset: BenchmarkDataset,
    feature_sets: dict[str, FeatureSet],
    feature_bundle: FeatureBundle,
    folds: list[FoldSpec],
    config: BenchmarkConfig,
    metadata: dict[str, object],
) -> None:
    if not feature_sets:
        raise ValueError(f"Null model {null_name!r} produced no evaluable feature sets.")
    bundle = FeatureBundle(
        feature_sets=feature_sets,
        fano_tables=feature_bundle.fano_tables,
        baseline_tables=feature_bundle.baseline_tables,
        cache_dir=feature_bundle.cache_dir,
    )
    if dataset.task == "clustering":
        fold_metrics, _ = _evaluate_clustering(run_id, dataset, bundle, folds, config)
    else:
        fold_metrics, _ = _evaluate_supervised(run_id, dataset, bundle, folds, config)
    primary = fold_metrics[
        fold_metrics["metric_name"] == config.evaluation.primary_metric
    ]
    for (feature_set, model), group in primary.groupby(["feature_set", "model"], sort=True):
        values = pd.to_numeric(group["metric_value"], errors="coerce").to_numpy(dtype=float)
        rows.append(
            {
                "run_id": run_id,
                "null_model": null_name,
                "null_iteration": iteration,
                "feature_set": feature_set,
                "model": model,
                "metric_name": config.evaluation.primary_metric,
                "metric_value": float(np.mean(values)),
                "metric_std": float(np.std(values)),
                "n_folds": int(len(values)),
                "metadata_json": json.dumps(metadata, sort_keys=True),
            }
        )


def _sequence_null_dataset(
    dataset: BenchmarkDataset,
    null_name: str,
    config: BenchmarkConfig,
    rng: np.random.Generator,
) -> tuple[BenchmarkDataset, dict[str, object]]:
    frame = config.feature_extraction.frame
    if null_name in {"codon_order_shuffle", "synonymous_codon_shuffle"} and frame == "all":
        raise ValueError(
            f"Sequence null {null_name!r} requires one explicit reading frame, not frame='all'."
        )
    genetic_code = get_genetic_code(config.feature_extraction.codon_table)
    sequences: dict[str, str] = {}
    for sequence_id in dataset.sequence_ids:
        sequence = dataset.sequences[sequence_id]
        if null_name == "mononucleotide_shuffle":
            transformed = mononucleotide_shuffle(sequence, rng)
        elif null_name == "dinucleotide_preserving_shuffle":
            transformed = dinucleotide_preserving_shuffle(sequence, rng)
        elif null_name == "codon_order_shuffle":
            transformed = codon_order_shuffle(sequence, rng, frame=int(frame))
        elif null_name == "synonymous_codon_shuffle":
            transformed = synonymous_codon_shuffle(
                sequence, genetic_code, rng, frame=int(frame)
            )
        else:
            raise ValueError(f"Unsupported sequence null model: {null_name}.")
        sequences[sequence_id] = transformed
    records = tuple(
        FastaRecord(id=record.id, description=record.description, sequence=sequences[record.id])
        for record in dataset.records
    )
    null_dataset = replace(dataset, records=records, sequences=sequences)
    metadata: dict[str, object] = {
        "n_sequences": len(sequences),
        "n_changed": sum(
            sequences[sequence_id] != dataset.sequences[sequence_id]
            for sequence_id in dataset.sequence_ids
        ),
        "lengths_preserved": all(
            len(sequences[sequence_id]) == len(dataset.sequences[sequence_id])
            for sequence_id in dataset.sequence_ids
        ),
        "labels_preserved": True,
        "groups_preserved": True,
        "frame": frame,
    }
    return null_dataset, metadata


def _write_fasta(records: tuple[FastaRecord, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record in records:
        lines.extend((f">{record.id}", record.sequence))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _permute_axis_bearing_features(
    matrix: pd.DataFrame,
    permutation: tuple[int, ...],
) -> tuple[pd.DataFrame, bool]:
    transformed = matrix.copy()
    original = matrix.copy()
    changed = False
    for prefix in ("e", "p", "a"):
        for target_axis, source_axis in enumerate(permutation, start=1):
            pattern = re.compile(rf"(?<![A-Za-z0-9]){prefix}{target_axis}(?![0-9])")
            for target_column in matrix.columns:
                if not pattern.search(target_column):
                    continue
                source_column = pattern.sub(f"{prefix}{source_axis}", target_column)
                if source_column in original.columns:
                    transformed[target_column] = original[source_column].to_numpy()
                    changed = True
    return transformed, changed


def _random_tensor_feature_set(
    fano_tables: dict[str, pd.DataFrame],
    tensor: np.ndarray,
) -> FeatureSet:
    source = fano_tables.get("window_octonions", pd.DataFrame())
    rows: list[dict[str, object]] = []
    if not source.empty:
        component_columns = [f"e{axis}" for axis in range(1, 8)]
        for sequence_id, group in source.groupby("sequence_id", sort=False):
            ordered = group.sort_values("position")
            values = ordered[component_columns].to_numpy(dtype=float)
            interactions = (
                np.einsum("ni,nj,ijk->nk", values[:-1], values[1:], tensor)
                if len(values) > 1
                else np.zeros((1, 7), dtype=float)
            )
            row: dict[str, object] = {"sequence_id": sequence_id}
            for axis in range(7):
                channel = interactions[:, axis]
                row[f"random_tensor_e{axis + 1}_mean"] = float(np.mean(channel))
                row[f"random_tensor_e{axis + 1}_std"] = float(np.std(channel))
                row[f"random_tensor_e{axis + 1}_min"] = float(np.min(channel))
                row[f"random_tensor_e{axis + 1}_max"] = float(np.max(channel))
            rows.append(row)
    return FeatureSet(
        name="random_antisymmetric_tensor",
        family="representation_null",
        description="Random antisymmetric 7x7x7 tensor contracted with adjacent windows.",
        matrix=pd.DataFrame(rows, columns=None),
        source_tables=("window_octonions",),
    )


def _supervised_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None,
    *,
    task: str,
    primary_metric: str,
) -> dict[str, float]:
    if task == "regression":
        rmse = float(mean_squared_error(y_true, y_pred, squared=False))
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "rmse": rmse,
            "mae": float(mean_absolute_error(y_true, y_pred)),
            primary_metric: _metric_value(primary_metric, y_true, y_pred, scores, task),
        }
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    aucs = _auc_metrics(y_true, scores)
    metrics.update(aucs)
    if primary_metric not in metrics:
        metrics[primary_metric] = _metric_value(primary_metric, y_true, y_pred, scores, task)
    return metrics


def _metric_value(
    metric_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None,
    task: str,
) -> float:
    if metric_name == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric_name in {"macro_f1", "f1_macro"}:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if metric_name == "mcc":
        return float(matthews_corrcoef(y_true, y_pred))
    if metric_name == "r2":
        return float(r2_score(y_true, y_pred))
    if metric_name == "rmse":
        return float(mean_squared_error(y_true, y_pred, squared=False))
    if metric_name == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    aucs = _auc_metrics(y_true, scores)
    return float(aucs.get(metric_name, np.nan))


def _auc_metrics(y_true: np.ndarray, scores: np.ndarray | None) -> dict[str, float]:
    if scores is None:
        return {"roc_auc": np.nan, "pr_auc": np.nan}
    labels = np.unique(y_true)
    try:
        if len(labels) == 2:
            score_vector = (
                scores[:, 1] if scores.ndim == 2 and scores.shape[1] > 1 else scores.ravel()
            )
            return {
                "roc_auc": float(roc_auc_score(y_true, score_vector)),
                "pr_auc": float(average_precision_score(y_true == labels[1], score_vector)),
            }
        binarizer = LabelBinarizer()
        y_binary = binarizer.fit_transform(y_true)
        return {
            "roc_auc": float(roc_auc_score(y_binary, scores, average="macro", multi_class="ovr")),
            "pr_auc": np.nan,
        }
    except ValueError:
        return {"roc_auc": np.nan, "pr_auc": np.nan}


def _prediction_scores(estimator: Any, x_test: np.ndarray) -> np.ndarray | None:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x_test)
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(x_test)
        if scores.ndim == 1:
            return np.column_stack([-scores, scores])
        return scores
    return None


def _clustering_metrics(
    dataset: BenchmarkDataset,
    fold: FoldSpec,
    x_test: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if dataset.y is not None:
        y_true = dataset.y[fold.test_indices]
        metrics["adjusted_rand"] = float(adjusted_rand_score(y_true, labels))
        metrics["normalized_mutual_info"] = float(normalized_mutual_info_score(y_true, labels))
    if len(np.unique(labels)) > 1 and len(labels) > len(np.unique(labels)):
        metrics["silhouette"] = float(silhouette_score(x_test, labels))
    else:
        metrics["silhouette"] = np.nan
    return metrics


def _metric_row(
    *,
    run_id: str,
    feature_set: str,
    model: str,
    fold: FoldSpec,
    metric_name: str,
    metric_value: float,
    n_train: int,
    n_test: int,
    best_params: dict[str, object],
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "feature_set": feature_set,
        "model": model,
        "repeat": fold.repeat,
        "fold": fold.fold,
        "split_id": fold.split_id,
        "level": "fold",
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_std": np.nan,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "n_folds": np.nan,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "best_params_json": json.dumps(best_params, sort_keys=True),
        "metadata_json": json.dumps(metadata, sort_keys=True),
    }


def _confusion_json(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> dict[str, object]:
    if task == "regression":
        return {}
    labels = sorted(str(label) for label in np.unique(np.concatenate([y_true, y_pred])))
    matrix = confusion_matrix(y_true.astype(str), y_pred.astype(str), labels=labels)
    return {"labels": labels, "matrix": matrix.tolist()}


def _score_payload(scores: np.ndarray | None, index: int) -> dict[str, object]:
    if scores is None:
        return {}
    value = scores[index]
    if np.isscalar(value):
        return {"score": float(np.asarray(value).item())}
    return {"scores": [float(item) for item in np.asarray(value).ravel()]}


def _cluster_count(dataset: BenchmarkDataset) -> int:
    if dataset.y is not None:
        return max(2, len(np.unique(dataset.y)))
    return min(3, max(2, len(dataset.sequence_ids) // 2))


def _run_id(config: BenchmarkConfig) -> str:
    stem = config.config_path.stem if config.config_path is not None else "benchmark"
    digest = hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode("utf-8"))
    return f"{stem}_{digest.hexdigest()[:12]}"


def _run_table(
    run_id: str,
    config: BenchmarkConfig,
    dataset: BenchmarkDataset,
) -> pd.DataFrame:
    row = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "fanoseq_version": fanoseq.__version__,
        "task": dataset.task,
        "seq_type": dataset.seq_type,
        "n_sequences": len(dataset.sequence_ids),
        "primary_metric": config.evaluation.primary_metric,
        "random_seed": config.evaluation.random_seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    row.update(input_hashes(config.dataset))
    return pd.DataFrame([row])


def _write_manifest(
    output_dir: Path,
    *,
    run_id: str,
    config: BenchmarkConfig,
    dataset: BenchmarkDataset,
    folds: pd.DataFrame,
    feature_bundle: FeatureBundle,
    output_paths: dict[str, str],
) -> Path:
    manifest = {
        "format": "fanoseq-benchmark",
        "schema_version": SCHEMA_VERSION,
        "fanoseq_version": fanoseq.__version__,
        "run_id": run_id,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_hashes": input_hashes(config.dataset),
        "configuration": config.to_dict(),
        "random_seeds": {
            "evaluation": config.evaluation.random_seed,
            "null_models": config.null_models.random_seed,
        },
        "statistical_design": {
            "primary_metric": config.evaluation.primary_metric,
            "preregistered_primary_comparator": config.evaluation.primary_comparator,
            "comparator_available": (
                config.evaluation.primary_comparator
                in strongest_conventional_candidates(feature_bundle.feature_sets)
            ),
            "note": (
                "Paired inference uses the preregistered comparator only; full baseline "
                "rankings are descriptive and are not used to select the inferential control."
            ),
        },
        "split_assignments": {
            "table": output_paths.get("benchmark_folds"),
            "n_rows": int(len(folds)),
            "unsafe_fallback_enabled": config.evaluation.allow_unsafe_split_fallback,
            "unsafe_fallback_used": bool(
                folds.get("unsafe_split_fallback", pd.Series(dtype=bool)).astype(bool).any()
            ),
            "fallback_reasons": sorted(
                reason
                for reason in folds.get(
                    "split_fallback_reason", pd.Series(dtype=str)
                ).astype(str).unique()
                if reason
            ),
        },
        "feature_definitions": feature_set_table(feature_bundle.feature_sets).to_dict("records"),
        "dataset_composition": dataset_composition(dataset),
        "software_versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "fanoseq": fanoseq.__version__,
        },
        "output_paths": output_paths,
        "feature_cache_dir": str(feature_bundle.cache_dir),
        "caution": (
            "FanoSeq descriptors are engineered features. Benchmark superiority is evidence "
            "about prediction, not proof of intrinsic octonionic biology."
        ),
    }
    path = output_dir / "benchmark_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _subset(values: np.ndarray | None, indices: np.ndarray) -> np.ndarray | None:
    if values is None:
        return None
    return values[indices]


def _dataset_with_y(dataset: BenchmarkDataset, y: np.ndarray) -> BenchmarkDataset:
    return BenchmarkDataset(
        metadata=dataset.metadata,
        records=dataset.records,
        sequence_ids=dataset.sequence_ids,
        sequences=dataset.sequences,
        y=y,
        groups=dataset.groups,
        task=dataset.task,
        seq_type=dataset.seq_type,
        id_column=dataset.id_column,
        target_column=dataset.target_column,
        group_column=dataset.group_column,
        parent_column=dataset.parent_column,
    )


def _config_with_models(config: BenchmarkConfig, *, limit_models: int) -> BenchmarkConfig:
    models = config.evaluation.models
    if not models:
        defaults = default_model_specs(
            config.dataset.task,
            random_seed=config.evaluation.random_seed,
            requested=(),
        )
        models = tuple(spec.name for spec in defaults[:limit_models])
    else:
        models = models[:limit_models]
    evaluation = config.evaluation.__class__(
        **{**config.evaluation.__dict__, "models": models, "paired_permutation_rounds": 0}
    )
    return BenchmarkConfig(
        dataset=config.dataset,
        features=config.features,
        feature_extraction=config.feature_extraction,
        evaluation=evaluation,
        null_models=config.null_models,
        config_path=config.config_path,
    )
