"""Orchestration for complete pipeline, benchmark, and encoding-audit runs."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from fanoseq.benchmark import load_benchmark_config, run_benchmark
from fanoseq.encoding_audit import (
    EncodingAuditConfig,
    build_encoding_audit_tables,
    plot_encoding_audit_outputs,
)
from fanoseq.io import write_outputs
from fanoseq.pipeline import RunConfig, run_analysis
from fanoseq.plots import plot_multipanel


@dataclass(frozen=True)
class CompleteAnalysisConfig:
    """Configuration for a complete, consistently organized FanoSeq analysis."""

    input_path: Path
    benchmark_config: Path
    output_dir: Path
    seq_type: str = "dna"
    window_size: int = 10
    step: int = 1
    kmer_k: int = 2
    audit_permutation_samples: int = 16
    audit_max_perturbations: int = 200


def run_complete_analysis(config: CompleteAnalysisConfig) -> dict[str, Path]:
    """Run all three workflows and promote their main reports to the output root."""
    benchmark = load_benchmark_config(config.benchmark_config)
    input_path = config.input_path.resolve()
    benchmark_fasta = benchmark.dataset.fasta.resolve()
    if input_path != benchmark_fasta:
        raise ValueError(
            "The complete analysis requires --input to match dataset.fasta in "
            f"--benchmark-config. Input: {input_path}; benchmark FASTA: {benchmark_fasta}."
        )
    if config.seq_type != benchmark.dataset.seq_type:
        raise ValueError(
            "--seq-type must match dataset.seq_type in --benchmark-config "
            f"({benchmark.dataset.seq_type!r})."
        )

    root = config.output_dir
    pipeline_dir = root / "pipeline"
    benchmark_dir = root / "benchmark"
    audit_dir = root / "audit"
    root.mkdir(parents=True, exist_ok=True)

    mode = "both" if config.seq_type == "dna" else "window"
    run_analysis(
        RunConfig(
            input_path=config.input_path,
            seq_type=config.seq_type,  # type: ignore[arg-type]
            mode=mode,
            output_dir=pipeline_dir,
            window_size=config.window_size,
            step=config.step,
            kmer_k=config.kmer_k,
            frame="all" if config.seq_type == "dna" else 0,
        )
    )
    pipeline_window_plot = plot_multipanel(
        pipeline_dir,
        pipeline_dir / "pipeline_window_multipanel.png",
        mode="window",
    )
    pipeline_codon_plot: Path | None = None
    if config.seq_type == "dna":
        pipeline_codon_plot = plot_multipanel(
            pipeline_dir,
            pipeline_dir / "pipeline_codon_multipanel.png",
            mode="codon",
        )

    benchmark_outputs = run_benchmark(config.benchmark_config, benchmark_dir)

    audit_checks = (
        ("reverse-complement", "permutation", "collision", "mutation", "redundancy", "codon")
        if config.seq_type == "dna"
        else ("contracts", "permutation", "mutation", "redundancy")
    )
    axis_scheme = "dna-window-v1" if config.seq_type == "dna" else "protein-sequence-v1"
    audit_tables = build_encoding_audit_tables(
        EncodingAuditConfig(
            input_path=config.input_path,
            seq_type=config.seq_type,  # type: ignore[arg-type]
            axis_scheme_id=axis_scheme,
            checks=audit_checks,
            window_size=config.window_size,
            step=config.step,
            kmer_k=config.kmer_k,
            permutation_samples=config.audit_permutation_samples,
            max_perturbations=config.audit_max_perturbations,
        )
    )
    write_outputs(audit_tables, audit_dir, "tsv")
    audit_plots = plot_encoding_audit_outputs(audit_tables, audit_dir)
    audit_multipanel = audit_dir / "encoding_audit_multipanel.png"
    if audit_multipanel not in audit_plots:
        raise RuntimeError("The encoding audit did not produce its multipanel report.")

    promoted = {
        "pipeline_window_plot": _promote(pipeline_window_plot, root),
        "benchmark_plot": _promote(benchmark_outputs["plot"], root),
        "benchmark_report": _promote(benchmark_outputs["report"], root),
        "audit_plot": _promote(audit_multipanel, root),
    }
    if pipeline_codon_plot is not None:
        promoted["pipeline_codon_plot"] = _promote(pipeline_codon_plot, root)
    fingerprints = pipeline_dir / "sequence_fingerprints.tsv"
    if fingerprints.exists():
        promoted["sequence_fingerprints"] = _promote(fingerprints, root)

    manifest = root / "analysis_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "fanoseq-complete-analysis",
                "input_path": str(input_path),
                "benchmark_config": str(config.benchmark_config.resolve()),
                "seq_type": config.seq_type,
                "subdirectories": {
                    "pipeline": "pipeline",
                    "benchmark": "benchmark",
                    "audit": "audit",
                },
                "main_files": {
                    name: path.name for name, path in promoted.items()
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {**promoted, "manifest": manifest}


def _promote(source: Path, output_root: Path) -> Path:
    destination = output_root / source.name
    shutil.copy2(source, destination)
    return destination
