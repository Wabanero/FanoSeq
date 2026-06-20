"""AI-ready tensor export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def table_to_component_tensor(
    table: pd.DataFrame,
    group_column: str = "sequence_id",
    component_prefix: str = "e",
    order_column: str = "position",
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Convert a component table into a padded tensor shaped [N, 8, L]."""
    if group_column not in table.columns:
        raise ValueError(f"Table must contain {group_column!r}.")
    component_columns = [f"{component_prefix}{index}" for index in range(8)]
    missing = [column for column in component_columns if column not in table.columns]
    if missing:
        raise ValueError(f"Table is missing component columns: {', '.join(missing)}.")

    sequence_ids = [str(value) for value in table[group_column].drop_duplicates().tolist()]
    lengths = np.zeros(len(sequence_ids), dtype=np.int64)
    max_length = 0
    groups: list[pd.DataFrame] = []
    for sequence_id in sequence_ids:
        group = table[table[group_column].astype(str) == sequence_id]
        if order_column in group.columns:
            group = group.sort_values(order_column)
        groups.append(group)
        max_length = max(max_length, len(group))

    tensor = np.zeros((len(sequence_ids), 8, max_length), dtype=np.float32)
    for row_index, group in enumerate(groups):
        values = group[component_columns].to_numpy(dtype=np.float32)
        length = values.shape[0]
        lengths[row_index] = length
        if length:
            tensor[row_index, :, :length] = values.T
    return tensor, sequence_ids, lengths


def write_tensor_npz(
    table: pd.DataFrame,
    output_path: str | Path,
    group_column: str = "sequence_id",
    component_prefix: str = "e",
    order_column: str = "position",
) -> Path:
    """Write a padded [N, 8, L] component tensor to an NPZ archive."""
    tensor, sequence_ids, lengths = table_to_component_tensor(
        table,
        group_column=group_column,
        component_prefix=component_prefix,
        order_column=order_column,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=tensor,
        sequence_id=np.asarray(sequence_ids, dtype=object),
        length=lengths,
        component_prefix=np.asarray(component_prefix),
        group_column=np.asarray(group_column),
        order_column=np.asarray(order_column),
    )
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a FanoSeq TSV or Parquet table for tensor export."""
    table_path = Path(path)
    if table_path.suffix.lower() == ".parquet":
        return pd.read_parquet(table_path)
    if table_path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(table_path, sep="\t")
    if table_path.suffix.lower() == ".csv":
        return pd.read_csv(table_path)
    raise ValueError("Expected a .tsv, .csv, or .parquet table.")
