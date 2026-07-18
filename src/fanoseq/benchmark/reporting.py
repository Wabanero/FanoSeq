"""Markdown reporting for benchmark runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fanoseq.benchmark.datasets import BenchmarkDataset, dataset_composition


def write_markdown_report(
    path: Path,
    *,
    run_id: str,
    dataset: BenchmarkDataset,
    fold_table: pd.DataFrame,
    leakage_table: pd.DataFrame,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    null_results: pd.DataFrame,
    ablation_results: pd.DataFrame,
    primary_metric: str,
) -> Path:
    """Write a compact benchmark report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# FanoSeq Benchmark Report: {run_id}",
        "",
        "## Task Description",
        (
            f"Task: `{dataset.task}`. The benchmark evaluates sequence-level feature "
            "families under leakage-controlled cross-validation."
        ),
        "",
        "## Dataset Composition",
        _composition_text(dataset),
        "",
        "## Split Design",
        _split_text(fold_table),
        "",
        "## Leakage Checks",
        _leakage_text(leakage_table),
        "",
        "## Baseline Ranking",
        _ranking_text(metrics, primary_metric),
        "",
        "## FanoSeq Versus Best Conventional Baseline",
        _comparison_text(comparisons),
        "",
        "## Null-Model Results",
        _null_text(null_results),
        "",
        "## Ablation Results",
        _ablation_text(ablation_results),
        "",
        "## Interpretation Guidance",
        (
            "Predictive superiority, not visual complexity, is the relevant test. "
            "Fano-plane products, commutators, associators, and line summaries should "
            "be considered useful only when they improve held-out performance over "
            "ordinary sequence features and survive null-model controls."
        ),
        "",
        (
            "The benchmark treats octonion/Fano-plane descriptors as engineered features. "
            "It does not claim that biological sequences are intrinsically octonionic."
        ),
        "",
        "## Limitations",
        (
            "Small datasets, weak group labels, related sequences across folds, or sparse "
            "class labels can make performance estimates unstable. Treat corrected "
            "p-values and effect sizes as screening evidence, not proof of mechanism."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _composition_text(dataset: BenchmarkDataset) -> str:
    composition = dataset_composition(dataset)
    parts = [
        f"Sequences: {composition['n_sequences']}",
        f"sequence type: `{composition['seq_type']}`",
        f"length range: {composition['min_length']}..{composition['max_length']}",
    ]
    if "label_counts" in composition:
        parts.append(f"labels: `{composition['label_counts']}`")
    if "n_groups" in composition:
        parts.append(f"groups: {composition['n_groups']}")
    return "; ".join(parts) + "."


def _split_text(folds: pd.DataFrame) -> str:
    if folds.empty:
        return "No fold table was produced."
    n_splits = folds["split_id"].nunique()
    n_repeats = folds["repeat"].nunique() if "repeat" in folds.columns else 1
    return f"Recorded {n_splits} outer splits across {n_repeats} repeat(s)."


def _leakage_text(leakage: pd.DataFrame) -> str:
    if leakage.empty:
        return "No leakage table was produced."
    group_leaks = int(leakage["group_leakage_detected"].sum())
    similarity_leaks = int(leakage["sequence_similarity_leakage_detected"].sum())
    unsafe_fallbacks = int(
        leakage.get("unsafe_split_fallback", pd.Series(dtype=bool)).astype(bool).sum()
    )
    if unsafe_fallbacks:
        reasons = sorted(
            reason
            for reason in leakage.get(
                "split_fallback_reason", pd.Series(dtype=str)
            ).astype(str).unique()
            if reason
        )
        return (
            f"WARNING: {unsafe_fallbacks} split(s) used the explicitly enabled unsafe "
            f"fallback. Reasons: {'; '.join(reasons)} Group-overlap detections: "
            f"{group_leaks}; positional-identity detections: {similarity_leaks}."
        )
    if group_leaks == 0 and similarity_leaks == 0:
        return "No group leakage or audited sequence-similarity leakage was detected."
    return (
        f"Detected {group_leaks} split(s) with group overlap and "
        f"{similarity_leaks} split(s) above the configured similarity threshold."
    )


def _ranking_text(metrics: pd.DataFrame, primary_metric: str) -> str:
    aggregate = metrics[
        (metrics["level"] == "aggregate") & (metrics["metric_name"] == primary_metric)
    ].copy()
    if aggregate.empty:
        return "No aggregate primary-metric rows were produced."
    top = aggregate.sort_values("metric_value", ascending=False).head(8)
    return _markdown_table(top[["feature_set", "model", "metric_value", "ci95_low", "ci95_high"]])


def _comparison_text(comparisons: pd.DataFrame) -> str:
    if comparisons.empty:
        return "No paired comparisons were available."
    fanoseq = comparisons[comparisons["feature_set"].astype(str).str.startswith("fanoseq")]
    if fanoseq.empty:
        return "No FanoSeq feature set was included in paired comparisons."
    best = fanoseq.sort_values("mean_difference", ascending=False).head(1).iloc[0]
    if float(best["mean_difference"]) <= 0:
        return (
            "FanoSeq feature families did not outperform the strongest ordinary baseline "
            "for the best observed paired comparison in this run."
        )
    return (
        "Best FanoSeq paired comparison: "
        f"`{best['feature_set']}` vs `{best['best_conventional_feature_set']}` "
        f"with mean difference {float(best['mean_difference']):.4f}."
    )


def _null_text(null_results: pd.DataFrame) -> str:
    if null_results.empty:
        return "No optional null-model evaluations were requested."
    return _markdown_table(null_results.head(10))


def _ablation_text(ablation: pd.DataFrame) -> str:
    if ablation.empty:
        return "No ablation rows were produced."
    columns = [
        "model",
        "feature_set",
        "metric_value",
        "increment_from_previous",
        "ci95_low",
        "ci95_high",
    ]
    return _markdown_table(ablation[columns].head(20))


def _markdown_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "No rows."
    text = table.copy()
    for column in text.columns:
        text[column] = text[column].map(_format_cell)
    header = "| " + " | ".join(str(column) for column in text.columns) + " |"
    separator = "| " + " | ".join("---" for _ in text.columns) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in text.columns) + " |"
        for _, row in text.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
