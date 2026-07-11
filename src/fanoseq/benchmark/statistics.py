"""Statistical summaries and paired benchmark comparisons."""

from __future__ import annotations

import json
from math import sqrt

import numpy as np
import pandas as pd

from fanoseq.benchmark.models import higher_is_better


def aggregate_metric_rows(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-fold metric rows with mean, SD, and normal CIs."""
    if fold_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_columns = ["run_id", "feature_set", "model", "metric_name"]
    for group_key, group in fold_metrics.groupby(group_columns, sort=False):
        run_id, feature_set, model, metric_name = group_key
        values = pd.to_numeric(group["metric_value"], errors="coerce").dropna().to_numpy(float)
        mean = float(np.mean(values)) if values.size else np.nan
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        half_width = 1.96 * std / sqrt(values.size) if values.size > 1 else 0.0
        rows.append(
            {
                "run_id": run_id,
                "feature_set": feature_set,
                "model": model,
                "repeat": -1,
                "fold": -1,
                "split_id": "aggregate",
                "level": "aggregate",
                "metric_name": metric_name,
                "metric_value": mean,
                "metric_std": std,
                "ci95_low": mean - half_width if values.size else np.nan,
                "ci95_high": mean + half_width if values.size else np.nan,
                "n_folds": int(values.size),
                "n_train": np.nan,
                "n_test": np.nan,
                "best_params_json": "{}",
                "metadata_json": "{}",
            }
        )
    return pd.DataFrame(rows)


def paired_comparison_table(
    fold_metrics: pd.DataFrame,
    *,
    conventional_feature_sets: set[str],
    primary_metric: str,
    random_seed: int,
    n_rounds: int,
) -> pd.DataFrame:
    """Compare every feature set against the strongest conventional baseline."""
    metric_rows = fold_metrics[
        (fold_metrics["level"] == "fold") & (fold_metrics["metric_name"] == primary_metric)
    ].copy()
    if metric_rows.empty or not conventional_feature_sets:
        return pd.DataFrame(
            columns=[
                "feature_set",
                "model",
                "primary_metric",
                "best_conventional_feature_set",
                "mean_difference",
                "effect_size_dz",
                "paired_permutation_p_value",
                "p_value_bh",
                "n_pairs",
            ]
        )
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    for model, model_rows in metric_rows.groupby("model", sort=False):
        baseline_rows = model_rows[model_rows["feature_set"].isin(conventional_feature_sets)]
        if baseline_rows.empty:
            continue
        baseline_scores = (
            baseline_rows.groupby("feature_set")["metric_value"].mean().sort_values(ascending=False)
        )
        if not higher_is_better(primary_metric):
            baseline_scores = baseline_scores.sort_values(ascending=True)
        baseline_name = str(baseline_scores.index[0])
        baseline = model_rows[model_rows["feature_set"] == baseline_name]
        baseline_keyed = baseline.set_index(["repeat", "fold"])["metric_value"]
        for feature_set, candidate in model_rows.groupby("feature_set", sort=False):
            candidate_keyed = candidate.set_index(["repeat", "fold"])["metric_value"]
            joined = pd.concat([candidate_keyed, baseline_keyed], axis=1, join="inner")
            joined.columns = ["candidate", "baseline"]
            diffs = joined["candidate"].to_numpy(float) - joined["baseline"].to_numpy(float)
            if not higher_is_better(primary_metric):
                diffs *= -1.0
            mean_diff = float(np.mean(diffs)) if diffs.size else np.nan
            std_diff = float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0
            effect = mean_diff / std_diff if std_diff > 0 else 0.0
            p_value = _paired_sign_permutation_p_value(diffs, rng, n_rounds)
            rows.append(
                {
                    "feature_set": feature_set,
                    "model": model,
                    "primary_metric": primary_metric,
                    "best_conventional_feature_set": baseline_name,
                    "mean_difference": mean_diff,
                    "effect_size_dz": effect,
                    "paired_permutation_p_value": p_value,
                    "p_value_bh": np.nan,
                    "n_pairs": int(diffs.size),
                    "fold_differences_json": json.dumps([float(value) for value in diffs]),
                }
            )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_value_bh"] = _benjamini_hochberg(
            table["paired_permutation_p_value"].to_numpy(float)
        )
    return table


def ablation_results_table(
    aggregate_metrics: pd.DataFrame,
    *,
    primary_metric: str,
) -> pd.DataFrame:
    """Return primary-metric aggregate rows for the incremental ablation family."""
    if aggregate_metrics.empty:
        return pd.DataFrame()
    order = [
        "ablation_base_descriptors",
        "ablation_plus_octonion_products",
        "ablation_plus_commutators",
        "ablation_plus_associators",
        "ablation_plus_fano_line_summaries",
    ]
    rows = aggregate_metrics[
        (aggregate_metrics["feature_set"].isin(order))
        & (aggregate_metrics["metric_name"] == primary_metric)
    ].copy()
    if rows.empty:
        return rows
    rows["ablation_stage"] = rows["feature_set"].map(
        {name: index for index, name in enumerate(order)}
    )
    rows = rows.sort_values(["model", "ablation_stage"])
    rows["increment_from_previous"] = rows.groupby("model")["metric_value"].diff().fillna(0.0)
    return rows[
        [
            "run_id",
            "model",
            "feature_set",
            "ablation_stage",
            "metric_name",
            "metric_value",
            "metric_std",
            "ci95_low",
            "ci95_high",
            "increment_from_previous",
            "n_folds",
        ]
    ]


def _paired_sign_permutation_p_value(
    diffs: np.ndarray,
    rng: np.random.Generator,
    n_rounds: int,
) -> float:
    if diffs.size == 0:
        return np.nan
    observed = abs(float(np.mean(diffs)))
    if n_rounds <= 0:
        return np.nan
    hits = 0
    for _ in range(n_rounds):
        signs = rng.choice([-1.0, 1.0], size=diffs.size)
        statistic = abs(float(np.mean(diffs * signs)))
        hits += int(statistic >= observed)
    return float((hits + 1) / (n_rounds + 1))


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    adjusted = np.full_like(p_values, fill_value=np.nan, dtype=float)
    finite_mask = np.isfinite(p_values)
    finite_values = p_values[finite_mask]
    if finite_values.size == 0:
        return adjusted
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    n = ranked.size
    corrected = np.empty(n, dtype=float)
    previous = 1.0
    for reverse_index in range(n - 1, -1, -1):
        rank = reverse_index + 1
        value = min(previous, ranked[reverse_index] * n / rank)
        corrected[reverse_index] = value
        previous = value
    reordered = np.empty(n, dtype=float)
    reordered[order] = corrected
    adjusted[finite_mask] = reordered
    return adjusted
