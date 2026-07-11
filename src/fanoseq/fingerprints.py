"""Sequence-level fingerprints for downstream comparison and modelling."""

from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENT_COLUMNS = [f"e{index}" for index in range(8)]
SUMMARY_STATS = ("mean", "std", "min", "max", "q25", "q50", "q75")


def component_fingerprint(
    table: pd.DataFrame,
    group_columns: list[str],
    component_prefix: str = "e",
    value_columns: list[str] | None = None,
    name_prefix: str = "",
) -> pd.DataFrame:
    """Summarize component and score columns into one row per group."""
    component_columns = [f"{component_prefix}{index}" for index in range(8)]
    selected_columns = [column for column in component_columns if column in table.columns]
    if value_columns:
        selected_columns.extend(column for column in value_columns if column in table.columns)
    selected_columns = list(dict.fromkeys(selected_columns))
    if table.empty or not selected_columns:
        return pd.DataFrame(columns=group_columns)

    rows: list[dict[str, object]] = []
    for group_key, group in table.groupby(group_columns, sort=False, dropna=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row: dict[str, object] = dict(zip(group_columns, key_values))
        for column in selected_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(dtype=float)
            stats = _summary_stats(values)
            for stat_name, stat_value in stats.items():
                row[f"{name_prefix}{column}_{stat_name}"] = stat_value
        rows.append(row)
    return pd.DataFrame(rows)


def build_sequence_fingerprints(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge available FanoSeq summary tables into one downstream feature table."""
    frames: list[pd.DataFrame] = []

    window_summary = tables.get("window_sequence_summary")
    if window_summary is not None and not window_summary.empty:
        frames.append(_prefix_numeric_columns(window_summary, "window_", keep=["sequence_id"]))

    codon_summary = tables.get("codon_usage_sequence_summary")
    if codon_summary is not None and not codon_summary.empty:
        numeric = codon_summary.select_dtypes(include=["number"]).columns.tolist()
        numeric = [column for column in numeric if column != "frame"]
        aggregated = (
            codon_summary.groupby("sequence_id", sort=False)[numeric]
            .agg(["mean", "max"])
            .reset_index()
        )
        aggregated.columns = [
            "sequence_id"
            if column[0] == "sequence_id"
            else f"codon_{column[0]}_{column[1]}"
            for column in aggregated.columns
        ]
        frames.append(aggregated)

    usage = tables.get("codon_usage_fano_features")
    if usage is not None and not usage.empty:
        usage_fp = component_fingerprint(
            usage,
            group_columns=["sequence_id"],
            component_prefix="mean_e",
            value_columns=["frequency", "rscu", "mean_codon_associator_score"],
            name_prefix="usage_",
        )
        frames.append(usage_fp)

    fano_lines = tables.get("fano_line_features")
    if fano_lines is not None and not fano_lines.empty:
        numeric = fano_lines.select_dtypes(include=["number"]).columns.tolist()
        numeric = [column for column in numeric if column != "frame"]
        if numeric:
            aggregated = (
                fano_lines.groupby("sequence_id", sort=False)[numeric]
                .agg(["mean", "max"])
                .reset_index()
            )
            aggregated.columns = [
                "sequence_id"
                if column[0] == "sequence_id"
                else f"fano_{column[0]}_{column[1]}"
                for column in aggregated.columns
            ]
            frames.append(aggregated)

    if not frames:
        return pd.DataFrame(columns=["sequence_id"])

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="sequence_id", how="outer")
    numeric_columns = merged.select_dtypes(include=["number"]).columns
    merged.loc[:, numeric_columns] = merged.loc[:, numeric_columns].fillna(0.0)
    return merged


def _prefix_numeric_columns(df: pd.DataFrame, prefix: str, keep: list[str]) -> pd.DataFrame:
    renamed = df.copy()
    rename_map = {
        column: f"{prefix}{column}"
        for column in renamed.columns
        if column not in keep and pd.api.types.is_numeric_dtype(renamed[column])
    }
    return renamed.rename(columns=rename_map)


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {name: 0.0 for name in SUMMARY_STATS}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q50": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
    }
