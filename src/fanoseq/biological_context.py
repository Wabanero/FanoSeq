"""Validated genomic-context and synchronized multi-track inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

CONTEXT_REQUIRED_COLUMNS = (
    "sequence_id",
    "genome_build",
    "chromosome",
    "start",
    "end",
    "strand",
    "feature_type",
)
TRACK_REQUIRED_COLUMNS = (
    "sequence_id",
    "start",
    "end",
    "track_name",
    "value",
)


def validate_genomic_context(table: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize sequence-level genomic context metadata."""
    _require_columns(table, CONTEXT_REQUIRED_COLUMNS, "genomic context")
    result = table.copy()
    result["sequence_id"] = result["sequence_id"].astype(str)
    if result["sequence_id"].duplicated().any():
        raise ValueError("Genomic context must contain one row per sequence_id.")
    result["start"] = pd.to_numeric(result["start"], errors="raise").astype(int)
    result["end"] = pd.to_numeric(result["end"], errors="raise").astype(int)
    if (result["start"] < 0).any() or (result["end"] <= result["start"]).any():
        raise ValueError("Genomic context coordinates require 0 <= start < end.")
    invalid_strands = set(result["strand"].astype(str)) - {"+", "-", "."}
    if invalid_strands:
        raise ValueError(f"Unsupported genomic-context strand values: {sorted(invalid_strands)}.")
    for column in ("genome_build", "chromosome", "feature_type"):
        if result[column].isna().any() or (result[column].astype(str).str.len() == 0).any():
            raise ValueError(f"Genomic context column {column!r} contains missing values.")
    return result


def validate_multitrack_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate a long synchronized-track table keyed by sequence windows."""
    _require_columns(table, TRACK_REQUIRED_COLUMNS, "multi-track table")
    result = table.copy()
    result["sequence_id"] = result["sequence_id"].astype(str)
    result["track_name"] = result["track_name"].astype(str)
    result["start"] = pd.to_numeric(result["start"], errors="raise").astype(int)
    result["end"] = pd.to_numeric(result["end"], errors="raise").astype(int)
    result["value"] = pd.to_numeric(result["value"], errors="raise").astype(float)
    if (result["start"] < 0).any() or (result["end"] <= result["start"]).any():
        raise ValueError("Multi-track coordinates require 0 <= start < end.")
    keys = ["sequence_id", "start", "end", "track_name"]
    if result.duplicated(keys).any():
        raise ValueError("Multi-track table contains duplicate sequence/window/track rows.")
    if not np.isfinite(result["value"].to_numpy(dtype=float)).all():
        raise ValueError("Multi-track values must be finite numeric observations.")
    return result.sort_values(keys).reset_index(drop=True)


def align_tracks_to_windows(
    windows: pd.DataFrame,
    tracks: pd.DataFrame,
    *,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Pivot synchronized long tracks onto exact FanoSeq window coordinates."""
    _require_columns(windows, ("sequence_id", "start", "end"), "window table")
    validated = validate_multitrack_table(tracks)
    wide = validated.pivot(
        index=["sequence_id", "start", "end"],
        columns="track_name",
        values="value",
    ).reset_index()
    wide.columns.name = None
    aligned = windows.copy()
    aligned["sequence_id"] = aligned["sequence_id"].astype(str)
    aligned = aligned.merge(
        wide,
        on=["sequence_id", "start", "end"],
        how="left",
        validate="one_to_one",
    )
    track_columns = [
        column
        for column in wide.columns
        if column not in {"sequence_id", "start", "end"}
    ]
    if not allow_missing and track_columns and aligned[track_columns].isna().any().any():
        raise ValueError(
            "One or more FanoSeq windows lack synchronized track values; use an explicit "
            "missing-data policy before encoding."
        )
    return aligned


def _require_columns(table: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}.")
