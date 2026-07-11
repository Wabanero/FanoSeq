"""Feature-set construction for benchmark comparisons and ablations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from fanoseq.baselines import build_baseline_tables
from fanoseq.benchmark.config import BenchmarkConfig
from fanoseq.benchmark.datasets import BenchmarkDataset
from fanoseq.genetic_code import get_genetic_code
from fanoseq.pipeline import Mode, RunConfig, SeqType, run_analysis


@dataclass(frozen=True)
class FeatureSet:
    """One sequence-level design matrix."""

    name: str
    family: str
    description: str
    matrix: pd.DataFrame
    source_tables: tuple[str, ...]

    @property
    def feature_columns(self) -> list[str]:
        """Return numeric feature columns excluding the sequence key."""
        return [column for column in self.matrix.columns if column != "sequence_id"]


@dataclass(frozen=True)
class FeatureBundle:
    """All benchmark feature matrices and provenance metadata."""

    feature_sets: dict[str, FeatureSet]
    fano_tables: dict[str, pd.DataFrame]
    baseline_tables: dict[str, pd.DataFrame]
    cache_dir: Path


FANO_FEATURES = {
    "fanoseq_components",
    "fanoseq_products",
    "fanoseq_commutators",
    "fanoseq_associators",
    "fanoseq_fano_lines",
    "fanoseq_fingerprints",
}
CONVENTIONAL_FEATURES = {"nucleotide_composition", "kmer", "fcgr", "codon_usage"}
CONTROL_FEATURES = {"real_polynomial_control", "random_projection_control"}


def build_feature_bundle(
    config: BenchmarkConfig,
    dataset: BenchmarkDataset,
    output_dir: Path,
) -> FeatureBundle:
    """Build requested sequence-level design matrices."""
    cache_dir = output_dir / "_feature_cache"
    fano_tables = _build_fano_tables(config, dataset, cache_dir)
    baseline_tables = _build_baseline_tables(config, dataset)
    feature_sets = _requested_feature_sets(config, dataset, fano_tables, baseline_tables)
    if config.evaluation.run_ablations:
        feature_sets.update(build_ablation_feature_sets(feature_sets))
    return FeatureBundle(
        feature_sets=feature_sets,
        fano_tables=fano_tables,
        baseline_tables=baseline_tables,
        cache_dir=cache_dir,
    )


def feature_set_table(feature_sets: dict[str, FeatureSet]) -> pd.DataFrame:
    """Return feature-set provenance and dimensionality."""
    rows = []
    for feature_set in feature_sets.values():
        rows.append(
            {
                "feature_set": feature_set.name,
                "family": feature_set.family,
                "description": feature_set.description,
                "n_features": len(feature_set.feature_columns),
                "source_tables": ",".join(feature_set.source_tables),
                "columns_json": json.dumps(feature_set.feature_columns),
            }
        )
    return pd.DataFrame(rows)


def matrix_for_ids(feature_set: FeatureSet, sequence_ids: tuple[str, ...]) -> pd.DataFrame:
    """Return a matrix aligned to the benchmark sequence order."""
    matrix = feature_set.matrix.copy()
    if "sequence_id" not in matrix.columns:
        raise ValueError(f"Feature set {feature_set.name} is missing sequence_id.")
    matrix["sequence_id"] = matrix["sequence_id"].astype(str)
    matrix = matrix.drop_duplicates(subset=["sequence_id"], keep="first")
    aligned = pd.DataFrame({"sequence_id": list(sequence_ids)}).merge(
        matrix,
        on="sequence_id",
        how="left",
    )
    numeric = aligned.drop(columns=["sequence_id"]).apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(0.0)
    return pd.concat([aligned[["sequence_id"]], numeric], axis=1)


def build_ablation_feature_sets(feature_sets: dict[str, FeatureSet]) -> dict[str, FeatureSet]:
    """Build incremental Fano feature-family ablations."""
    components = feature_sets.get("fanoseq_components")
    if components is None:
        return {}
    stages: list[tuple[str, str, list[str]]] = [
        ("ablation_base_descriptors", "base descriptors", ["fanoseq_components"]),
        (
            "ablation_plus_octonion_products",
            "+ octonion products",
            ["fanoseq_components", "fanoseq_products"],
        ),
        (
            "ablation_plus_commutators",
            "+ commutators",
            ["fanoseq_components", "fanoseq_products", "fanoseq_commutators"],
        ),
        (
            "ablation_plus_associators",
            "+ associators",
            [
                "fanoseq_components",
                "fanoseq_products",
                "fanoseq_commutators",
                "fanoseq_associators",
            ],
        ),
        (
            "ablation_plus_fano_line_summaries",
            "+ Fano-line summaries",
            [
                "fanoseq_components",
                "fanoseq_products",
                "fanoseq_commutators",
                "fanoseq_associators",
                "fanoseq_fano_lines",
            ],
        ),
    ]
    ablations: dict[str, FeatureSet] = {}
    for name, description, members in stages:
        available = [feature_sets[member] for member in members if member in feature_sets]
        if not available:
            continue
        ablations[name] = FeatureSet(
            name=name,
            family="ablation",
            description=description,
            matrix=_merge_feature_matrices(available),
            source_tables=tuple(
                source for feature_set in available for source in feature_set.source_tables
            ),
        )
    return ablations


def strongest_conventional_candidates(feature_sets: dict[str, FeatureSet]) -> set[str]:
    """Return feature-set names that count as ordinary biological baselines or controls."""
    return {
        name
        for name, feature_set in feature_sets.items()
        if feature_set.family in {"conventional", "control"}
    }


def _requested_feature_sets(
    config: BenchmarkConfig,
    dataset: BenchmarkDataset,
    fano_tables: dict[str, pd.DataFrame],
    baseline_tables: dict[str, pd.DataFrame],
) -> dict[str, FeatureSet]:
    builders = {
        "fanoseq_components": lambda: _fanoseq_components(fano_tables),
        "fanoseq_products": lambda: _fanoseq_products(fano_tables),
        "fanoseq_commutators": lambda: _fanoseq_commutators(fano_tables),
        "fanoseq_associators": lambda: _fanoseq_associators(fano_tables),
        "fanoseq_fano_lines": lambda: _fanoseq_fano_lines(fano_tables),
        "fanoseq_fingerprints": lambda: _fanoseq_fingerprints(fano_tables),
        "nucleotide_composition": lambda: _nucleotide_composition(baseline_tables),
        "kmer": lambda: _kmer_matrix(baseline_tables),
        "fcgr": lambda: _fcgr_matrix(baseline_tables),
        "codon_usage": lambda: _codon_usage_matrix(baseline_tables),
        "real_polynomial_control": lambda: _real_polynomial_control(fano_tables),
        "random_projection_control": lambda: _random_projection_control(
            config,
            dataset,
            fano_tables,
            baseline_tables,
        ),
    }
    feature_sets: dict[str, FeatureSet] = {}
    requested_names = list(config.features)
    if config.evaluation.run_ablations:
        for name in (
            "fanoseq_components",
            "fanoseq_products",
            "fanoseq_commutators",
            "fanoseq_associators",
            "fanoseq_fano_lines",
        ):
            if name not in requested_names:
                requested_names.append(name)

    for name in requested_names:
        if name not in builders:
            raise ValueError(f"Unsupported benchmark feature set: {name}")
        feature_set = builders[name]()
        feature_sets[name] = _align_feature_set(feature_set, dataset.sequence_ids)
    return feature_sets


def _build_fano_tables(
    config: BenchmarkConfig,
    dataset: BenchmarkDataset,
    cache_dir: Path,
) -> dict[str, pd.DataFrame]:
    needs_window = any(
        feature in set(config.features) | {"fanoseq_components"}
        for feature in FANO_FEATURES | CONTROL_FEATURES
    )
    needs_codon = dataset.seq_type == "dna" and (
        "codon_usage" in config.features or "fanoseq_fingerprints" in config.features
    )
    mode = "both" if needs_window and needs_codon else "codon" if needs_codon else "window"
    window_size = _effective_window_size(config.feature_extraction.window_size, dataset)
    run_config = RunConfig(
        input_path=config.dataset.fasta,
        seq_type=cast(SeqType, dataset.seq_type),
        mode=cast(Mode, mode),
        output_dir=cache_dir,
        window_size=window_size if mode in {"window", "both"} else None,
        step=config.feature_extraction.step,
        kmer_k=config.feature_extraction.kmer_k,
        max_ambiguous_fraction=config.feature_extraction.max_ambiguous_fraction,
        frame=config.feature_extraction.frame,
        codon_table=config.feature_extraction.codon_table,
        include_stop_codons=config.feature_extraction.include_stop_codons,
        codon_normalize=config.feature_extraction.codon_normalize,
        window_axis_scheme=config.feature_extraction.window_axis_scheme,
        codon_axis_scheme=config.feature_extraction.codon_axis_scheme,
        output_format="tsv",
    )
    outputs = run_analysis(run_config)
    tables = {Path(relative_path).stem: table for relative_path, table in outputs.items()}
    return tables


def _build_baseline_tables(
    config: BenchmarkConfig,
    dataset: BenchmarkDataset,
) -> dict[str, pd.DataFrame]:
    if dataset.seq_type == "dna":
        genetic_code = get_genetic_code(config.feature_extraction.codon_table)
        return build_baseline_tables(
            dataset.records,
            seq_type="dna",
            kmer_k=config.feature_extraction.kmer_k,
            genetic_code=genetic_code,
            frame=config.feature_extraction.frame,
        )
    return build_baseline_tables(
        dataset.records,
        seq_type="protein",
        kmer_k=min(config.feature_extraction.kmer_k, 2),
    )


def _fanoseq_components(tables: dict[str, pd.DataFrame]) -> FeatureSet:
    source = tables.get("window_octonions", pd.DataFrame())
    matrix = _summarize(
        source,
        [f"e{index}" for index in range(8)],
        prefix="component",
    )
    return FeatureSet(
        name="fanoseq_components",
        family="fanoseq",
        description="Raw FanoSeq window components e0...e7 summarized per sequence.",
        matrix=matrix,
        source_tables=("window_octonions",),
    )


def _fanoseq_products(tables: dict[str, pd.DataFrame]) -> FeatureSet:
    source = tables.get("octonion_products", pd.DataFrame())
    matrix = _summarize(
        source,
        [f"p{index}" for index in range(8)] + ["product_norm"],
        prefix="product",
    )
    return FeatureSet(
        name="fanoseq_products",
        family="fanoseq",
        description="Adjacent-window octonion product components p0...p7.",
        matrix=matrix,
        source_tables=("octonion_products",),
    )


def _fanoseq_commutators(tables: dict[str, pd.DataFrame]) -> FeatureSet:
    source = tables.get("octonion_products", pd.DataFrame())
    matrix = _summarize(
        source,
        ["commutator_score", "transition_score"],
        prefix="commutator",
    )
    return FeatureSet(
        name="fanoseq_commutators",
        family="fanoseq",
        description="Commutator and transition-score summaries for adjacent windows.",
        matrix=matrix,
        source_tables=("octonion_products",),
    )


def _fanoseq_associators(tables: dict[str, pd.DataFrame]) -> FeatureSet:
    source = tables.get("octonion_triplets", pd.DataFrame())
    matrix = _summarize(
        source,
        [f"a{index}" for index in range(8)] + ["associator_score"],
        prefix="associator",
    )
    return FeatureSet(
        name="fanoseq_associators",
        family="fanoseq",
        description="Associator component and score summaries for consecutive triplets.",
        matrix=matrix,
        source_tables=("octonion_triplets",),
    )


def _fanoseq_fano_lines(tables: dict[str, pd.DataFrame]) -> FeatureSet:
    source = tables.get("fano_line_features", pd.DataFrame())
    matrix = _numeric_by_sequence(source, prefix="fano_line")
    return FeatureSet(
        name="fanoseq_fano_lines",
        family="fanoseq",
        description="Fano-line attribution profile summaries.",
        matrix=matrix,
        source_tables=("fano_line_features",),
    )


def _fanoseq_fingerprints(tables: dict[str, pd.DataFrame]) -> FeatureSet:
    source = tables.get("sequence_fingerprints", pd.DataFrame(columns=["sequence_id"]))
    matrix = _numeric_by_sequence(source, prefix="fingerprint")
    return FeatureSet(
        name="fanoseq_fingerprints",
        family="fanoseq",
        description="Combined FanoSeq sequence fingerprints.",
        matrix=matrix,
        source_tables=("sequence_fingerprints",),
    )


def _nucleotide_composition(tables: dict[str, pd.DataFrame]) -> FeatureSet:
    source = tables.get("baseline_sequence_features", pd.DataFrame())
    matrix = _numeric_by_sequence(source, prefix="composition")
    return FeatureSet(
        name="nucleotide_composition",
        family="conventional",
        description="Standard composition, GC/AT skew, entropy, and length descriptors.",
        matrix=matrix,
        source_tables=("baseline_sequence_features",),
    )


def _kmer_matrix(tables: dict[str, pd.DataFrame]) -> FeatureSet:
    source = tables.get("baseline_kmer_feature_matrix", pd.DataFrame(columns=["sequence_id"]))
    matrix = _numeric_by_sequence(source, prefix="kmer")
    return FeatureSet(
        name="kmer",
        family="conventional",
        description="Standard fixed-length k-mer frequency vector.",
        matrix=matrix,
        source_tables=("baseline_kmer_feature_matrix",),
    )


def _fcgr_matrix(tables: dict[str, pd.DataFrame]) -> FeatureSet:
    source = tables.get("baseline_kmer_frequencies", pd.DataFrame())
    if source.empty or not {"sequence_id", "fcgr_x", "fcgr_y", "frequency"}.issubset(
        source.columns
    ):
        matrix = pd.DataFrame(columns=["sequence_id"])
    else:
        table = source.copy()
        table["fcgr_cell"] = (
            "fcgr_"
            + table["fcgr_x"].astype(int).astype(str)
            + "_"
            + table["fcgr_y"].astype(int).astype(str)
        )
        matrix = (
            table.pivot_table(
                index="sequence_id",
                columns="fcgr_cell",
                values="frequency",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
    return FeatureSet(
        name="fcgr",
        family="conventional",
        description="Flattened FCGR tensor/cell frequencies derived from k-mer coordinates.",
        matrix=matrix,
        source_tables=("baseline_kmer_frequencies",),
    )


def _codon_usage_matrix(tables: dict[str, pd.DataFrame]) -> FeatureSet:
    source = tables.get("baseline_codon_usage", pd.DataFrame())
    if source.empty:
        matrix = pd.DataFrame(columns=["sequence_id"])
    else:
        rows: list[dict[str, object]] = []
        for sequence_id, group in source.groupby("sequence_id", sort=False):
            row: dict[str, object] = {"sequence_id": sequence_id}
            for codon, codon_group in group.groupby("codon", sort=True):
                frequency = float(codon_group["frequency"].mean())
                row[f"codon_freq_{codon}"] = frequency
                row[f"codon_rscu_{codon}"] = float(codon_group["rscu"].mean())
            for aa, aa_group in group.groupby("amino_acid", sort=True):
                row[f"aa_comp_{aa}"] = float(aa_group["frequency"].sum())
            codon_frequency = group.groupby("codon")["frequency"].mean()
            row["codon_entropy"] = _entropy(codon_frequency.to_numpy(dtype=float), base=64)
            row["gc1"] = _position_gc(group, 0)
            row["gc2"] = _position_gc(group, 1)
            row["gc3"] = _position_gc(group, 2)
            rows.append(row)
        matrix = pd.DataFrame(rows).fillna(0.0)
    return FeatureSet(
        name="codon_usage",
        family="conventional",
        description=(
            "Codon frequencies, RSCU, GC1/GC2/GC3, codon entropy, "
            "and amino-acid composition."
        ),
        matrix=matrix,
        source_tables=("baseline_codon_usage",),
    )


def _real_polynomial_control(tables: dict[str, pd.DataFrame]) -> FeatureSet:
    components = _fanoseq_components(tables).matrix
    if components.empty:
        matrix = pd.DataFrame(columns=["sequence_id"])
    else:
        base_columns = [
            column
            for column in components.columns
            if column != "sequence_id" and column.endswith("_mean")
        ]
        if not base_columns:
            base_columns = [column for column in components.columns if column != "sequence_id"]
        matrix = components[["sequence_id", *base_columns]].copy()
        values = matrix[base_columns].to_numpy(dtype=float)
        for index, column in enumerate(base_columns):
            matrix[f"poly_{column}_square"] = values[:, index] ** 2
            matrix[f"poly_{column}_cube"] = values[:, index] ** 3
        for left_index, left in enumerate(base_columns):
            for right_index in range(left_index + 1, len(base_columns)):
                right = base_columns[right_index]
                matrix[f"pair_product_{left}__{right}"] = (
                    values[:, left_index] * values[:, right_index]
                )
                matrix[f"abs_diff_{left}__{right}"] = np.abs(
                    values[:, left_index] - values[:, right_index]
                )
    return FeatureSet(
        name="real_polynomial_control",
        family="control",
        description=(
            "Real-valued control from the same eight descriptors with ordinary products, "
            "absolute differences, squares, and cubes; no octonion multiplication."
        ),
        matrix=matrix,
        source_tables=("window_octonions",),
    )


def _random_projection_control(
    config: BenchmarkConfig,
    dataset: BenchmarkDataset,
    fano_tables: dict[str, pd.DataFrame],
    baseline_tables: dict[str, pd.DataFrame],
) -> FeatureSet:
    conventional = [
        _nucleotide_composition(baseline_tables),
        _kmer_matrix(baseline_tables),
        _fcgr_matrix(baseline_tables),
    ]
    if dataset.seq_type == "dna":
        conventional.append(_codon_usage_matrix(baseline_tables))
    source = _merge_feature_matrices(conventional)
    source = _align_matrix(source, dataset.sequence_ids)
    source_columns = [column for column in source.columns if column != "sequence_id"]
    if not source_columns:
        matrix = pd.DataFrame(columns=["sequence_id"])
    else:
        fano_dim = config.feature_extraction.random_projection_dim
        if fano_dim is None:
            fano_dim = max(1, len(_fanoseq_fingerprints(fano_tables).feature_columns))
        rng = np.random.default_rng(config.evaluation.random_seed)
        values = source[source_columns].to_numpy(dtype=float)
        projection = rng.normal(0.0, 1.0, size=(values.shape[1], int(fano_dim)))
        projection /= np.sqrt(max(values.shape[1], 1))
        projected = np.column_stack(
            [np.sum(values * projection[:, index], axis=1) for index in range(int(fano_dim))]
        )
        matrix = pd.DataFrame(
            projected,
            columns=[f"random_projection_{index}" for index in range(int(fano_dim))],
        )
        matrix.insert(0, "sequence_id", source["sequence_id"].to_numpy())
    return FeatureSet(
        name="random_projection_control",
        family="control",
        description="Random projection of conventional features matched to FanoSeq dimensionality.",
        matrix=matrix,
        source_tables=("baseline_sequence_features", "baseline_kmer_feature_matrix"),
    )


def _summarize(source: pd.DataFrame, columns: list[str], *, prefix: str) -> pd.DataFrame:
    if source.empty or "sequence_id" not in source.columns:
        return pd.DataFrame(columns=["sequence_id"])
    selected = [column for column in columns if column in source.columns]
    if not selected:
        return pd.DataFrame(columns=["sequence_id"])
    rows: list[dict[str, object]] = []
    for sequence_id, group in source.groupby("sequence_id", sort=False):
        row: dict[str, object] = {"sequence_id": sequence_id}
        for column in selected:
            values = pd.to_numeric(group[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            row[f"{prefix}_{column}_mean"] = float(np.mean(values)) if values.size else 0.0
            row[f"{prefix}_{column}_std"] = float(np.std(values)) if values.size else 0.0
            row[f"{prefix}_{column}_min"] = float(np.min(values)) if values.size else 0.0
            row[f"{prefix}_{column}_max"] = float(np.max(values)) if values.size else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _numeric_by_sequence(source: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    if source.empty or "sequence_id" not in source.columns:
        return pd.DataFrame(columns=["sequence_id"])
    numeric_columns = [
        column
        for column in source.select_dtypes(include=["number"]).columns
        if column not in {"frame", "position", "start", "end"}
    ]
    if not numeric_columns:
        return pd.DataFrame({"sequence_id": source["sequence_id"].astype(str).unique()})
    rows: list[dict[str, object]] = []
    for sequence_id, group in source.groupby("sequence_id", sort=False):
        row: dict[str, object] = {"sequence_id": sequence_id}
        if len(group) == 1:
            for column in numeric_columns:
                row[f"{prefix}_{column}"] = float(group[column].iloc[0])
        else:
            for column in numeric_columns:
                values = pd.to_numeric(group[column], errors="coerce").fillna(0.0).to_numpy(float)
                row[f"{prefix}_{column}_mean"] = float(np.mean(values))
                row[f"{prefix}_{column}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows)


def _merge_feature_matrices(feature_sets: list[FeatureSet]) -> pd.DataFrame:
    matrices = [feature_set.matrix for feature_set in feature_sets if not feature_set.matrix.empty]
    if not matrices:
        return pd.DataFrame(columns=["sequence_id"])
    merged = matrices[0].copy()
    for feature_set, matrix in zip(feature_sets[1:], matrices[1:]):
        renamed = matrix.rename(
            columns={
                column: f"{feature_set.name}__{column}"
                for column in matrix.columns
                if column != "sequence_id"
            }
        )
        merged = merged.merge(renamed, on="sequence_id", how="outer")
    return merged.fillna(0.0)


def _align_feature_set(feature_set: FeatureSet, sequence_ids: tuple[str, ...]) -> FeatureSet:
    return FeatureSet(
        name=feature_set.name,
        family=feature_set.family,
        description=feature_set.description,
        matrix=_align_matrix(feature_set.matrix, sequence_ids),
        source_tables=feature_set.source_tables,
    )


def _align_matrix(matrix: pd.DataFrame, sequence_ids: tuple[str, ...]) -> pd.DataFrame:
    if "sequence_id" not in matrix.columns:
        matrix = pd.DataFrame(columns=["sequence_id"])
    aligned = pd.DataFrame({"sequence_id": list(sequence_ids)}).merge(
        matrix,
        on="sequence_id",
        how="left",
    )
    for column in aligned.columns:
        if column != "sequence_id":
            aligned[column] = pd.to_numeric(aligned[column], errors="coerce").fillna(0.0)
    return aligned


def _effective_window_size(requested: int, dataset: BenchmarkDataset) -> int:
    min_length = min(len(sequence) for sequence in dataset.sequences.values())
    return max(1, min(requested, min_length))


def _entropy(values: np.ndarray, *, base: int) -> float:
    positive = values[values > 0]
    if positive.size == 0:
        return 0.0
    probabilities = positive / positive.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)) / np.log2(base))


def _position_gc(group: pd.DataFrame, position: int) -> float:
    values = []
    weights = []
    for _, row in group.iterrows():
        codon = str(row["codon"])
        if len(codon) <= position:
            continue
        values.append(1.0 if codon[position] in {"G", "C"} else 0.0)
        weights.append(float(row["frequency"]))
    total = sum(weights)
    if total <= 0:
        return 0.0
    return float(np.average(np.asarray(values), weights=np.asarray(weights)))
