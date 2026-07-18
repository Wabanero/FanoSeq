"""Tool-agnostic preparation of homology-cluster groups for benchmark splits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fanoseq.benchmark.datasets import sha256_file


@dataclass(frozen=True)
class HomologyGroupConfig:
    """Inputs and provenance for attaching external homology clusters."""

    metadata_path: Path
    cluster_table_path: Path
    output_path: Path
    id_column: str = "sequence_id"
    member_column: str = "sequence_id"
    cluster_column: str = "homology_cluster"
    output_group_column: str = "homology_cluster"
    clustering_tool: str = "external"
    clustering_tool_version: str = "unknown"
    minimum_identity: float | None = None
    minimum_coverage: float | None = None
    allow_singletons: bool = False


def prepare_homology_groups(config: HomologyGroupConfig) -> tuple[Path, Path]:
    """Attach one externally computed homology cluster to every metadata row.

    The external clustering method is deliberately not approximated by FanoSeq's
    positional-identity audit. Users can supply MMseqs2, CD-HIT, BLAST/graph, or another
    domain-appropriate cluster table and record its thresholds and version here.
    """
    for name, value in (
        ("minimum_identity", config.minimum_identity),
        ("minimum_coverage", config.minimum_coverage),
    ):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1 when supplied.")
    metadata = _read_table(config.metadata_path)
    clusters = _read_table(config.cluster_table_path)
    _require_columns(metadata, (config.id_column,), "metadata")
    _require_columns(
        clusters,
        (config.member_column, config.cluster_column),
        "cluster table",
    )
    metadata = metadata.copy()
    metadata[config.id_column] = metadata[config.id_column].astype(str)
    if metadata[config.id_column].duplicated().any():
        duplicates = metadata.loc[
            metadata[config.id_column].duplicated(), config.id_column
        ].unique()
        raise ValueError(f"Duplicate metadata sequence IDs: {', '.join(duplicates[:5])}.")

    assignments = clusters[[config.member_column, config.cluster_column]].copy()
    assignments[config.member_column] = assignments[config.member_column].astype(str)
    assignments[config.cluster_column] = assignments[config.cluster_column].astype(str)
    conflicts = assignments.groupby(config.member_column)[config.cluster_column].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(
            "Sequence IDs assigned to multiple homology clusters: "
            + ", ".join(conflicts.index.astype(str).tolist()[:5])
            + "."
        )
    assignments = assignments.drop_duplicates(subset=[config.member_column], keep="first")
    mapping = assignments.set_index(config.member_column)[config.cluster_column]
    metadata[config.output_group_column] = metadata[config.id_column].map(mapping)
    missing = metadata.loc[
        metadata[config.output_group_column].isna(), config.id_column
    ].astype(str)
    if not missing.empty and not config.allow_singletons:
        raise ValueError(
            f"Missing homology-cluster assignments for {len(missing)} sequence(s): "
            + ", ".join(missing.head(5))
            + "."
        )
    if not missing.empty:
        metadata.loc[metadata[config.output_group_column].isna(), config.output_group_column] = (
            "singleton__" + missing
        )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(config.output_path, sep="\t", index=False)
    manifest_path = config.output_path.with_suffix(".homology.json")
    manifest = {
        "format": "fanoseq-homology-groups",
        "schema_version": "1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "metadata_path": str(config.metadata_path.resolve()),
            "metadata_sha256": sha256_file(config.metadata_path),
            "cluster_table_path": str(config.cluster_table_path.resolve()),
            "cluster_table_sha256": sha256_file(config.cluster_table_path),
        },
        "columns": {
            "sequence_id": config.id_column,
            "external_member": config.member_column,
            "external_cluster": config.cluster_column,
            "benchmark_group": config.output_group_column,
        },
        "clustering": {
            "tool": config.clustering_tool,
            "version": config.clustering_tool_version,
            "minimum_identity": config.minimum_identity,
            "minimum_coverage": config.minimum_coverage,
        },
        "n_sequences": int(len(metadata)),
        "n_groups": int(metadata[config.output_group_column].nunique()),
        "n_singletons_created": int(len(missing)),
        "output_path": str(config.output_path.resolve()),
        "output_sha256": sha256_file(config.output_path),
        "instruction": (
            f"Set dataset.group_column to {config.output_group_column!r}; grouped splitting "
            "will then fail closed if the assignments cannot support the requested folds."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return config.output_path, manifest_path


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path.suffix}.")


def _require_columns(table: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}.")
