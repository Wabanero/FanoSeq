"""Dataset loading and validation for benchmark runs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fanoseq.benchmark.config import DatasetConfig
from fanoseq.fasta import FastaRecord, read_fasta


@dataclass(frozen=True)
class BenchmarkDataset:
    """Metadata, sequences, labels, and groups in one benchmark order."""

    metadata: pd.DataFrame
    records: tuple[FastaRecord, ...]
    sequence_ids: tuple[str, ...]
    sequences: dict[str, str]
    y: np.ndarray | None
    groups: np.ndarray | None
    task: str
    seq_type: str
    id_column: str
    target_column: str | None
    group_column: str | None
    parent_column: str | None


def load_benchmark_dataset(config: DatasetConfig) -> BenchmarkDataset:
    """Load metadata and FASTA records, then validate benchmark invariants."""
    metadata = _read_table(config.table)
    _require_columns(metadata, [config.id_column], "metadata")
    if config.target_column is not None:
        _require_columns(metadata, [config.target_column], "metadata")
    if config.group_column is not None:
        _require_columns(metadata, [config.group_column], "metadata")
    if config.parent_column is not None:
        _require_columns(metadata, [config.parent_column], "metadata")

    ids = metadata[config.id_column].astype(str)
    duplicates = ids[ids.duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"Duplicate sequence IDs in metadata: {', '.join(duplicates[:5])}.")

    records = read_fasta(config.fasta)
    fasta_ids = [record.id for record in records]
    fasta_duplicate_ids = sorted(
        {record_id for record_id in fasta_ids if fasta_ids.count(record_id) > 1}
    )
    if fasta_duplicate_ids:
        raise ValueError(
            f"Duplicate sequence IDs in FASTA: {', '.join(fasta_duplicate_ids[:5])}."
        )
    sequence_by_id = {record.id: record.sequence for record in records}
    missing_fasta = [sequence_id for sequence_id in ids if sequence_id not in sequence_by_id]
    if missing_fasta:
        raise ValueError(
            f"Metadata IDs without FASTA sequence: {', '.join(missing_fasta[:5])}."
        )
    if metadata.empty:
        raise ValueError("Benchmark metadata is empty.")

    if config.task != "clustering":
        assert config.target_column is not None
        labels = metadata[config.target_column]
        if labels.isna().any():
            missing = metadata.loc[labels.isna(), config.id_column].astype(str).tolist()
            raise ValueError(f"Missing labels for sequence IDs: {', '.join(missing[:5])}.")
        if config.task == "regression":
            y: np.ndarray | None = pd.to_numeric(labels, errors="raise").to_numpy(dtype=float)
        else:
            y = labels.astype(str).to_numpy()
    else:
        if config.target_column is not None and config.target_column in metadata.columns:
            y = metadata[config.target_column].astype(str).to_numpy()
        else:
            y = None

    groups = None
    group_column = config.group_column or config.parent_column
    if group_column is not None:
        groups = metadata[group_column].astype(str).to_numpy()

    ordered_records = tuple(
        FastaRecord(
            id=str(sequence_id),
            description=str(sequence_id),
            sequence=sequence_by_id[str(sequence_id)],
        )
        for sequence_id in ids
    )
    return BenchmarkDataset(
        metadata=metadata.copy(),
        records=ordered_records,
        sequence_ids=tuple(ids),
        sequences={record.id: record.sequence for record in ordered_records},
        y=y,
        groups=groups,
        task=config.task,
        seq_type=config.seq_type,
        id_column=config.id_column,
        target_column=config.target_column,
        group_column=config.group_column,
        parent_column=config.parent_column,
    )


def input_hashes(config: DatasetConfig) -> dict[str, str]:
    """Return stable SHA-256 hashes for benchmark inputs."""
    return {
        "metadata_sha256": sha256_file(config.table),
        "fasta_sha256": sha256_file(config.fasta),
    }


def dataset_composition(dataset: BenchmarkDataset) -> dict[str, object]:
    """Summarize dataset composition for manifests and reports."""
    lengths = [len(dataset.sequences[sequence_id]) for sequence_id in dataset.sequence_ids]
    payload: dict[str, object] = {
        "n_sequences": len(dataset.sequence_ids),
        "seq_type": dataset.seq_type,
        "task": dataset.task,
        "min_length": int(min(lengths)) if lengths else 0,
        "max_length": int(max(lengths)) if lengths else 0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }
    if dataset.y is not None and dataset.task != "regression":
        values, counts = np.unique(dataset.y, return_counts=True)
        payload["label_counts"] = {str(value): int(count) for value, count in zip(values, counts)}
    if dataset.groups is not None:
        payload["n_groups"] = int(len(np.unique(dataset.groups)))
    return payload


def sha256_file(path: str | Path) -> str:
    """Return a SHA-256 file digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported metadata table format: {path.suffix}")


def _require_columns(table: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}.")
