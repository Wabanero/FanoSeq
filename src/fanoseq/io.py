"""Output helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal

import pandas as pd

OutputFormat = Literal["tsv", "parquet", "bundle"]


def write_tsv(df: pd.DataFrame, path: str | Path) -> None:
    """Write a TSV file after rounding numeric columns to six decimals."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rounded = round_numeric(df)
    rounded.to_csv(output_path, sep="\t", index=False, na_rep="NA")


def write_parquet(df: pd.DataFrame, path: str | Path, partitioned: bool = False) -> None:
    """Write a Parquet file or partitioned Parquet dataset."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rounded = round_numeric(df)
    if partitioned and not rounded.empty:
        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()
        partition_cols = _partition_columns(rounded)
        rounded.to_parquet(output_path, index=False, partition_cols=partition_cols)
        return
    rounded.to_parquet(output_path, index=False)


def write_outputs(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
    output_format: OutputFormat,
    manifest: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write named output tables and return table-name to relative-path mapping."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    for stem, dataframe in tables.items():
        if output_format == "tsv":
            relative_path = f"{stem}.tsv"
            write_tsv(dataframe, base / relative_path)
        elif output_format == "parquet":
            relative_path = f"{stem}.parquet"
            write_parquet(dataframe, base / relative_path)
        elif output_format == "bundle":
            relative_path = f"{stem}.parquet"
            write_parquet(dataframe, base / relative_path, partitioned=True)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        written[stem] = relative_path

    if output_format == "bundle":
        manifest_payload = dict(manifest or {})
        manifest_payload["tables"] = [
            {
                "name": stem,
                "path": relative_path,
                "rows": int(len(tables[stem])),
                "columns": list(tables[stem].columns),
            }
            for stem, relative_path in written.items()
        ]
        (base / "manifest.json").write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return written


def round_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with numeric columns rounded to six decimals."""
    rounded = df.copy()
    numeric_columns = rounded.select_dtypes(include=["number"]).columns
    rounded.loc[:, numeric_columns] = rounded.loc[:, numeric_columns].round(6)
    return rounded


def _partition_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    if "sequence_id" in df.columns:
        columns.append("sequence_id")
    if "frame" in df.columns and not df.empty and pd.api.types.is_integer_dtype(df["frame"]):
        columns.append("frame")
    return columns
