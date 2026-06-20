"""FanoSeq analysis pipeline."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from fanoseq.codon_features import codon_entropy, encode_codon, iter_codons
from fanoseq.dna_features import encode_dna_window
from fanoseq.fano_attribution import fano_line_attribution
from fanoseq.fasta import read_fasta
from fanoseq.genetic_code import GeneticCode, all_standard_codons, get_genetic_code
from fanoseq.io import OutputFormat, write_outputs
from fanoseq.octonion import Octonion
from fanoseq.protein_features import encode_protein_window
from fanoseq.windows import iter_windows

SeqType = Literal["dna", "protein"]
Mode = Literal["window", "codon", "both"]

SCHEMA_VERSION = "0.2.0"

WINDOW_COLUMNS = [
    "sequence_id",
    "position",
    "start",
    "end",
    "window",
    "seq_type",
    "mono_entropy",
    "gc_content",
    "valid_fraction",
    "ambiguous_fraction",
    "e0",
    "e1",
    "e2",
    "e3",
    "e4",
    "e5",
    "e6",
    "e7",
]

PRODUCT_COLUMNS = [
    "sequence_id",
    "position",
    "window",
    "next_window",
    "p0",
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "p7",
    "product_norm",
    "commutator_score",
    "transition_score",
]

TRIPLET_COLUMNS = [
    "sequence_id",
    "position",
    "window_1",
    "window_2",
    "window_3",
    "a0",
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
    "a6",
    "a7",
    "associator_score",
]

CODON_COLUMNS = [
    "sequence_id",
    "frame",
    "codon_index",
    "start",
    "end",
    "codon",
    "amino_acid",
    "is_start",
    "is_stop",
    "valid_fraction",
    "ambiguous_fraction",
    "gc_content",
    "gc1",
    "gc2",
    "gc3",
    "ry_pos1",
    "ry_pos2",
    "ry_pos3",
    "sw_pos1",
    "sw_pos2",
    "sw_pos3",
    "mk_pos1",
    "mk_pos2",
    "mk_pos3",
    "codon_associator_score",
    "e0",
    "e1",
    "e2",
    "e3",
    "e4",
    "e5",
    "e6",
    "e7",
]

CODON_PRODUCT_COLUMNS = [
    "sequence_id",
    "frame",
    "position",
    "codon",
    "next_codon",
    "amino_acid",
    "next_amino_acid",
    "p0",
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "p7",
    "product_norm",
    "commutator_score",
    "transition_score",
]

CODON_USAGE_COLUMNS = [
    "sequence_id",
    "frame",
    "codon",
    "amino_acid",
    "is_stop",
    "count",
    "frequency",
    "synonymous_family_size",
    "rscu",
    "mean_e0",
    "mean_e1",
    "mean_e2",
    "mean_e3",
    "mean_e4",
    "mean_e5",
    "mean_e6",
    "mean_e7",
    "mean_codon_associator_score",
]

CODON_SUMMARY_COLUMNS = [
    "sequence_id",
    "frame",
    "n_valid_codons",
    "n_stop_codons",
    "stop_density",
    "codon_entropy",
    "gc1_mean",
    "gc2_mean",
    "gc3_mean",
    "mean_codon_transition_score",
    "max_codon_transition_score",
    "mean_codon_associator_score",
    "max_codon_associator_score",
]

WINDOW_SUMMARY_COLUMNS = [
    "sequence_id",
    "seq_type",
    "n_windows",
    "mean_transition_score",
    "max_transition_score",
    "mean_associator_score",
    "max_associator_score",
    "mean_e0",
    "mean_e1",
    "mean_e2",
    "mean_e3",
    "mean_e4",
    "mean_e5",
    "mean_e6",
    "mean_e7",
    "std_e0",
    "std_e1",
    "std_e2",
    "std_e3",
    "std_e4",
    "std_e5",
    "std_e6",
    "std_e7",
]


@dataclass(frozen=True)
class RunConfig:
    """Configuration for a FanoSeq run."""

    input_path: Path
    seq_type: SeqType
    mode: Mode
    output_dir: Path
    window_size: int | None = None
    step: int = 1
    kmer_k: int = 2
    epsilon: float = 1e-9
    max_ambiguous_fraction: float = 0.0
    frame: int | Literal["all"] = 0
    codon_table: str | int = "standard"
    include_partial_codons: bool = False
    include_stop_codons: bool = True
    codon_normalize: bool = False
    output_format: OutputFormat = "tsv"
    summary_only: bool = False
    top_k_transitions: int | None = None
    transition_threshold: float | None = None


def run_analysis(config: RunConfig) -> dict[str, pd.DataFrame]:
    """Run FanoSeq and write configured outputs."""
    _validate_config(config)
    records = read_fasta(config.input_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    fano_rows: list[dict[str, object]] = []

    if config.mode in {"window", "both"}:
        window_df = build_window_octonions(records, config)
        product_df, product_fano = build_window_products(window_df, config.seq_type)
        product_df, product_fano = _filter_product_events(product_df, product_fano, config, "window")
        triplet_df = build_window_triplets(window_df)
        window_summary_df = build_window_summary(window_df, product_df, triplet_df, config.seq_type)
        tables["window_sequence_summary"] = window_summary_df
        if not config.summary_only:
            tables["window_octonions"] = window_df
            tables["octonion_products"] = product_df
            tables["octonion_triplets"] = triplet_df
            fano_rows.extend(product_fano)

    if config.mode in {"codon", "both"}:
        genetic_code = get_genetic_code(config.codon_table)
        codon_df = build_codon_octonions(records, config, genetic_code)
        codon_product_df, codon_fano = build_codon_products(codon_df)
        codon_product_df, codon_fano = _filter_product_events(
            codon_product_df, codon_fano, config, "codon"
        )
        usage_df = build_codon_usage(codon_df, genetic_code, config)
        summary_df = build_codon_summary(codon_df, codon_product_df)
        tables["codon_usage_fano_features"] = usage_df
        tables["codon_usage_sequence_summary"] = summary_df
        if not config.summary_only:
            tables["codon_octonions"] = codon_df
            tables["codon_transition_products"] = codon_product_df
            fano_rows.extend(codon_fano)

    fano_df = pd.DataFrame(fano_rows)
    if not config.summary_only:
        if not fano_df.empty:
            tables["fano_interactions"] = fano_df
        else:
            tables["fano_interactions"] = _empty_fano_dataframe()

    written = write_outputs(
        tables,
        config.output_dir,
        config.output_format,
        manifest=_build_manifest(config),
    )
    return {relative_path: tables[stem] for stem, relative_path in written.items()}


def build_window_octonions(records: object, config: RunConfig) -> pd.DataFrame:
    """Build the window_octonions table."""
    rows: list[dict[str, object]] = []
    assert config.window_size is not None
    for record in records:
        for window in iter_windows(record.sequence, config.window_size, config.step):
            if config.seq_type == "dna":
                encoded = encode_dna_window(
                    window.sequence,
                    k=config.kmer_k,
                    epsilon=config.epsilon,
                    max_ambiguous_fraction=config.max_ambiguous_fraction,
                )
            else:
                encoded = encode_protein_window(
                    window.sequence,
                    k=config.kmer_k,
                    epsilon=config.epsilon,
                    max_ambiguous_fraction=config.max_ambiguous_fraction,
                )
            if encoded is None:
                continue
            octonion, metadata = encoded
            row = {
                "sequence_id": record.id,
                "position": window.position,
                "start": window.start,
                "end": window.end,
                "window": window.sequence,
                "seq_type": config.seq_type,
                **metadata,
            }
            row.update(_component_dict("e", octonion.components))
            rows.append(row)
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)


def build_window_products(
    window_df: pd.DataFrame, seq_type: str
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Build adjacent window product rows and Fano attribution rows."""
    rows: list[dict[str, object]] = []
    fano_rows: list[dict[str, object]] = []
    for _, group in window_df.groupby("sequence_id", sort=False):
        ordered = group.sort_values("position").reset_index(drop=True)
        for idx in range(len(ordered) - 1):
            left = ordered.iloc[idx]
            right = ordered.iloc[idx + 1]
            if int(right["position"]) != int(left["position"]) + 1:
                continue
            x = _row_octonion(left, "e")
            y = _row_octonion(right, "e")
            product = x * y
            commutator = x.commutator(y)
            commutator_score = commutator.norm()
            row = {
                "sequence_id": left["sequence_id"],
                "position": int(left["position"]),
                "window": left["window"],
                "next_window": right["window"],
                **_component_dict("p", product.components),
                "product_norm": product.norm(),
                "commutator_score": commutator_score,
                "transition_score": commutator_score,
            }
            rows.append(row)
            fano_rows.extend(
                fano_line_attribution(
                    x.components,
                    y.components,
                    sequence_id=str(left["sequence_id"]),
                    mode="window",
                    seq_type=seq_type,
                    frame="NA",
                    position=int(left["position"]),
                    left_object=str(left["window"]),
                    right_object=str(right["window"]),
                )
            )
    return pd.DataFrame(rows, columns=PRODUCT_COLUMNS), fano_rows


def build_window_triplets(window_df: pd.DataFrame) -> pd.DataFrame:
    """Build consecutive window triplet associator rows."""
    rows: list[dict[str, object]] = []
    for _, group in window_df.groupby("sequence_id", sort=False):
        ordered = group.sort_values("position").reset_index(drop=True)
        for idx in range(len(ordered) - 2):
            first = ordered.iloc[idx]
            second = ordered.iloc[idx + 1]
            third = ordered.iloc[idx + 2]
            if int(second["position"]) != int(first["position"]) + 1:
                continue
            if int(third["position"]) != int(second["position"]) + 1:
                continue
            x = _row_octonion(first, "e")
            y = _row_octonion(second, "e")
            z = _row_octonion(third, "e")
            associator = x.associator(y, z)
            row = {
                "sequence_id": first["sequence_id"],
                "position": int(first["position"]),
                "window_1": first["window"],
                "window_2": second["window"],
                "window_3": third["window"],
                **_component_dict("a", associator.components),
                "associator_score": associator.norm(),
            }
            rows.append(row)
    return pd.DataFrame(rows, columns=TRIPLET_COLUMNS)


def build_window_summary(
    window_df: pd.DataFrame,
    product_df: pd.DataFrame,
    triplet_df: pd.DataFrame,
    seq_type: str,
) -> pd.DataFrame:
    """Build compact per-sequence window trajectory fingerprints."""
    if window_df.empty:
        return pd.DataFrame(columns=WINDOW_SUMMARY_COLUMNS)

    rows: list[dict[str, object]] = []
    for sequence_id, group in window_df.groupby("sequence_id", sort=False):
        products = product_df[product_df["sequence_id"] == sequence_id]
        triplets = triplet_df[triplet_df["sequence_id"] == sequence_id]
        row: dict[str, object] = {
            "sequence_id": sequence_id,
            "seq_type": seq_type,
            "n_windows": len(group),
            "mean_transition_score": float(products["transition_score"].mean())
            if not products.empty
            else 0.0,
            "max_transition_score": float(products["transition_score"].max())
            if not products.empty
            else 0.0,
            "mean_associator_score": float(triplets["associator_score"].mean())
            if not triplets.empty
            else 0.0,
            "max_associator_score": float(triplets["associator_score"].max())
            if not triplets.empty
            else 0.0,
        }
        for index in range(8):
            row[f"mean_e{index}"] = float(group[f"e{index}"].mean())
            row[f"std_e{index}"] = float(group[f"e{index}"].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows, columns=WINDOW_SUMMARY_COLUMNS)


def build_codon_octonions(
    records: object, config: RunConfig, genetic_code: GeneticCode
) -> pd.DataFrame:
    """Build the codon_octonions table."""
    rows: list[dict[str, object]] = []
    frames = (0, 1, 2) if config.frame == "all" else (int(config.frame),)
    for record in records:
        for frame in frames:
            for codon_slice in iter_codons(record.sequence, frame, config.include_partial_codons):
                encoded = encode_codon(
                    codon_slice.codon,
                    genetic_code,
                    max_ambiguous_fraction=config.max_ambiguous_fraction,
                    include_stop_codons=config.include_stop_codons,
                    normalize=config.codon_normalize,
                )
                if encoded is None:
                    continue
                row = {
                    "sequence_id": record.id,
                    "frame": frame,
                    "codon_index": codon_slice.codon_index,
                    "start": codon_slice.start,
                    "end": codon_slice.end,
                    "codon": codon_slice.codon,
                    **encoded.metadata,
                }
                row.update(_component_dict("e", encoded.octonion.components))
                rows.append(row)
    return pd.DataFrame(rows, columns=CODON_COLUMNS)


def build_codon_products(codon_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Build adjacent codon products and Fano attribution rows."""
    rows: list[dict[str, object]] = []
    fano_rows: list[dict[str, object]] = []
    for _, group in codon_df.groupby(["sequence_id", "frame"], sort=False):
        ordered = group.sort_values("codon_index").reset_index(drop=True)
        for idx in range(len(ordered) - 1):
            left = ordered.iloc[idx]
            right = ordered.iloc[idx + 1]
            if int(right["codon_index"]) != int(left["codon_index"]) + 1:
                continue
            x = _row_octonion(left, "e")
            y = _row_octonion(right, "e")
            product = x * y
            commutator_score = x.commutator(y).norm()
            rows.append(
                {
                    "sequence_id": left["sequence_id"],
                    "frame": int(left["frame"]),
                    "position": int(left["codon_index"]),
                    "codon": left["codon"],
                    "next_codon": right["codon"],
                    "amino_acid": left["amino_acid"],
                    "next_amino_acid": right["amino_acid"],
                    **_component_dict("p", product.components),
                    "product_norm": product.norm(),
                    "commutator_score": commutator_score,
                    "transition_score": commutator_score,
                }
            )
            fano_rows.extend(
                fano_line_attribution(
                    x.components,
                    y.components,
                    sequence_id=str(left["sequence_id"]),
                    mode="codon",
                    seq_type="dna",
                    frame=int(left["frame"]),
                    position=int(left["codon_index"]),
                    left_object=str(left["codon"]),
                    right_object=str(right["codon"]),
                )
            )
    return pd.DataFrame(rows, columns=CODON_PRODUCT_COLUMNS), fano_rows


def build_codon_usage(
    codon_df: pd.DataFrame, genetic_code: GeneticCode, config: RunConfig
) -> pd.DataFrame:
    """Build codon usage and codon-octonion summary rows."""
    if codon_df.empty:
        return pd.DataFrame(columns=CODON_USAGE_COLUMNS)
    rows: list[dict[str, object]] = []
    codons64 = all_standard_codons()
    for (sequence_id, frame), group in codon_df.groupby(["sequence_id", "frame"], sort=False):
        counts = Counter(str(codon) for codon in group["codon"])
        total_valid = sum(counts.values())
        family_totals: dict[str, int] = {}
        for codon in codons64:
            aa = genetic_code.amino_acid(codon)
            family_totals[aa] = family_totals.get(aa, 0) + counts[codon]

        grouped_by_codon = {codon: values for codon, values in group.groupby("codon", sort=False)}
        for codon in codons64:
            aa = genetic_code.amino_acid(codon)
            family = genetic_code.synonymous_codons(aa)
            family_size = len(family)
            total_for_aa = family_totals.get(aa, 0)
            observed = counts[codon]
            expected = total_for_aa / family_size if family_size and total_for_aa else 0.0
            rscu = observed / expected if expected else 0.0

            if codon in grouped_by_codon:
                codon_group = grouped_by_codon[codon]
                mean_components = [float(codon_group[f"e{i}"].mean()) for i in range(8)]
                mean_assoc = float(codon_group["codon_associator_score"].mean())
            else:
                deterministic = encode_codon(
                    codon,
                    genetic_code,
                    max_ambiguous_fraction=0.0,
                    include_stop_codons=True,
                    normalize=config.codon_normalize,
                )
                assert deterministic is not None
                mean_components = deterministic.octonion.to_list()
                mean_assoc = float(deterministic.metadata["codon_associator_score"])

            row = {
                "sequence_id": sequence_id,
                "frame": int(frame),
                "codon": codon,
                "amino_acid": aa,
                "is_stop": genetic_code.is_stop(codon),
                "count": observed,
                "frequency": observed / total_valid if total_valid else 0.0,
                "synonymous_family_size": family_size,
                "rscu": rscu,
                "mean_codon_associator_score": mean_assoc,
            }
            row.update({f"mean_e{i}": mean_components[i] for i in range(8)})
            rows.append(row)
    return pd.DataFrame(rows, columns=CODON_USAGE_COLUMNS)


def build_codon_summary(codon_df: pd.DataFrame, codon_product_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-sequence/per-frame codon summary rows."""
    if codon_df.empty:
        return pd.DataFrame(columns=CODON_SUMMARY_COLUMNS)
    rows: list[dict[str, object]] = []
    for (sequence_id, frame), group in codon_df.groupby(["sequence_id", "frame"], sort=False):
        products = codon_product_df[
            (codon_product_df["sequence_id"] == sequence_id) & (codon_product_df["frame"] == frame)
        ]
        n_valid = len(group)
        n_stop = int(group["is_stop"].sum())
        row = {
            "sequence_id": sequence_id,
            "frame": int(frame),
            "n_valid_codons": n_valid,
            "n_stop_codons": n_stop,
            "stop_density": n_stop / n_valid if n_valid else 0.0,
            "codon_entropy": codon_entropy([str(codon) for codon in group["codon"]]),
            "gc1_mean": float(group["gc1"].mean()) if n_valid else np.nan,
            "gc2_mean": float(group["gc2"].mean()) if n_valid else np.nan,
            "gc3_mean": float(group["gc3"].mean()) if n_valid else np.nan,
            "mean_codon_transition_score": float(products["transition_score"].mean())
            if not products.empty
            else 0.0,
            "max_codon_transition_score": float(products["transition_score"].max())
            if not products.empty
            else 0.0,
            "mean_codon_associator_score": float(group["codon_associator_score"].mean())
            if n_valid
            else 0.0,
            "max_codon_associator_score": float(group["codon_associator_score"].max())
            if n_valid
            else 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=CODON_SUMMARY_COLUMNS)


def _validate_config(config: RunConfig) -> None:
    if config.seq_type not in {"dna", "protein"}:
        raise ValueError("--seq-type must be either 'dna' or 'protein'.")
    if config.mode not in {"window", "codon", "both"}:
        raise ValueError("--mode must be one of: window, codon, both.")
    if config.mode in {"codon", "both"} and config.seq_type != "dna":
        raise ValueError("Codon mode requires DNA input; use --seq-type dna.")
    if config.mode in {"window", "both"}:
        if config.window_size is None or config.window_size <= 0:
            raise ValueError("--window-size is required for window and both modes and must be > 0.")
    if config.step <= 0:
        raise ValueError("--step must be > 0.")
    if config.kmer_k <= 0:
        raise ValueError("--kmer-k must be > 0.")
    if config.max_ambiguous_fraction < 0 or config.max_ambiguous_fraction > 1:
        raise ValueError("--max-ambiguous-fraction must be between 0 and 1.")
    if config.frame != "all" and int(config.frame) not in {0, 1, 2}:
        raise ValueError("--frame must be 0, 1, 2, or all.")
    if config.output_format not in {"tsv", "parquet", "bundle"}:
        raise ValueError("--output-format must be one of: tsv, parquet, bundle.")
    if config.top_k_transitions is not None and config.top_k_transitions <= 0:
        raise ValueError("--top-k-transitions must be > 0 when provided.")
    if config.transition_threshold is not None and config.transition_threshold < 0:
        raise ValueError("--transition-threshold must be >= 0 when provided.")


def _component_dict(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {f"{prefix}{index}": float(values[index]) for index in range(8)}


def _row_octonion(row: pd.Series, prefix: str) -> Octonion:
    return Octonion([float(row[f"{prefix}{index}"]) for index in range(8)])


def _filter_product_events(
    product_df: pd.DataFrame,
    fano_rows: list[dict[str, object]],
    config: RunConfig,
    mode: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Apply transition threshold and top-k filtering to products and Fano rows."""
    if product_df.empty:
        return product_df, []

    filtered = product_df
    if config.transition_threshold is not None:
        filtered = filtered[filtered["transition_score"] >= config.transition_threshold]
    if config.top_k_transitions is not None and len(filtered) > config.top_k_transitions:
        filtered = filtered.nlargest(config.top_k_transitions, "transition_score")

    filtered = filtered.reset_index(drop=True)
    if mode == "window":
        event_keys = {
            (str(row["sequence_id"]), "NA", int(row["position"])) for _, row in filtered.iterrows()
        }
    else:
        event_keys = {
            (str(row["sequence_id"]), int(row["frame"]), int(row["position"]))
            for _, row in filtered.iterrows()
        }
    filtered_fano = [
        row
        for row in fano_rows
        if (str(row["sequence_id"]), row["frame"], int(row["position"])) in event_keys
    ]
    return filtered, filtered_fano


def _empty_fano_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sequence_id",
            "mode",
            "seq_type",
            "frame",
            "position",
            "left_object",
            "right_object",
            "fano_line",
            "axis_a",
            "axis_b",
            "axis_c",
            "axis_a_label",
            "axis_b_label",
            "axis_c_label",
            "pair_ab_to_c",
            "pair_bc_to_a",
            "pair_ca_to_b",
            "contribution_to_a",
            "contribution_to_b",
            "contribution_to_c",
            "line_contribution_norm",
        ]
    )


def _build_manifest(config: RunConfig) -> dict[str, object]:
    return {
        "format": "fanoseq-bundle",
        "schema_version": SCHEMA_VERSION,
        "input_path": str(config.input_path),
        "input_sha256": _sha256_file(config.input_path),
        "config": {
            "seq_type": config.seq_type,
            "mode": config.mode,
            "window_size": config.window_size,
            "step": config.step,
            "kmer_k": config.kmer_k,
            "epsilon": config.epsilon,
            "max_ambiguous_fraction": config.max_ambiguous_fraction,
            "frame": config.frame,
            "codon_table": config.codon_table,
            "include_partial_codons": config.include_partial_codons,
            "include_stop_codons": config.include_stop_codons,
            "codon_normalize": config.codon_normalize,
            "output_format": config.output_format,
            "summary_only": config.summary_only,
            "top_k_transitions": config.top_k_transitions,
            "transition_threshold": config.transition_threshold,
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
