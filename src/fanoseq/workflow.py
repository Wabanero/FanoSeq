"""Orchestration for complete pipeline, benchmark, and encoding-audit runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd

import fanoseq
from fanoseq.axis_schemes import resolve_axis_scheme
from fanoseq.benchmark import load_benchmark_config
from fanoseq.benchmark.config import BenchmarkConfig
from fanoseq.benchmark.evaluation import run_benchmark_config
from fanoseq.encoding_audit import (
    FANO_CONVENTION_ID,
    EncodingAuditConfig,
    build_encoding_audit_tables,
    plot_encoding_audit_outputs,
)
from fanoseq.io import write_outputs
from fanoseq.pipeline import Mode, RunConfig, run_analysis
from fanoseq.plots import plot_multipanel

ANALYSIS_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class CompleteAnalysisConfig:
    """Configuration for a complete, consistently organized FanoSeq analysis."""

    input_path: Path
    benchmark_config: Path
    output_dir: Path
    seq_type: str | None = None
    window_size: int | None = None
    step: int | None = None
    kmer_k: int | None = None
    allow_config_override: bool = False
    audit_permutation_samples: int = 16
    audit_max_perturbations: int = 200


def run_complete_analysis(config: CompleteAnalysisConfig) -> dict[str, Path]:
    """Run all workflows from one resolved extraction plan and write a rich manifest."""
    benchmark, override_warnings = _resolve_analysis_config(config)
    extraction = benchmark.feature_extraction
    input_path = benchmark.dataset.fasta.resolve()
    seq_type = benchmark.dataset.seq_type

    root = config.output_dir
    pipeline_dir = root / "pipeline"
    benchmark_dir = root / "benchmark"
    audit_dir = root / "audit"
    root.mkdir(parents=True, exist_ok=True)

    window_scheme = resolve_axis_scheme(
        seq_type,
        "window",
        extraction.window_axis_scheme,
        require_runnable=True,
    )
    codon_scheme = (
        resolve_axis_scheme(
            "dna",
            "codon",
            extraction.codon_axis_scheme,
            require_runnable=True,
        )
        if seq_type == "dna"
        else None
    )
    resolved_plan = {
        "input_path": str(input_path),
        "seq_type": seq_type,
        "feature_extraction": asdict(extraction),
        "axis_schemes": {
            "window": window_scheme.scheme_id,
            "codon": codon_scheme.scheme_id if codon_scheme else None,
        },
        "fano_convention_id": FANO_CONVENTION_ID,
    }

    started_utc = datetime.now(timezone.utc)
    stage_runtimes: dict[str, float] = {}
    tracemalloc.start()

    mode = "both" if seq_type == "dna" else "window"
    stage_started = time.perf_counter()
    pipeline_tables = run_analysis(
        RunConfig(
            input_path=input_path,
            seq_type=seq_type,
            mode=cast(Mode, mode),
            output_dir=pipeline_dir,
            window_size=extraction.window_size,
            step=extraction.step,
            kmer_k=extraction.kmer_k,
            epsilon=extraction.epsilon,
            max_ambiguous_fraction=extraction.max_ambiguous_fraction,
            frame=extraction.frame,
            codon_table=extraction.codon_table,
            include_partial_codons=extraction.include_partial_codons,
            include_stop_codons=extraction.include_stop_codons,
            rscu_stop_policy=extraction.rscu_stop_policy,
            codon_normalize=extraction.codon_normalize,
            window_axis_scheme=window_scheme.scheme_id,
            codon_axis_scheme=codon_scheme.scheme_id if codon_scheme else None,
            output_format=benchmark.evaluation.output_format,
        )
    )
    pipeline_window_plot = plot_multipanel(
        pipeline_dir,
        pipeline_dir / "pipeline_window_multipanel.png",
        mode="window",
    )
    pipeline_codon_plot: Path | None = None
    if seq_type == "dna":
        pipeline_codon_plot = plot_multipanel(
            pipeline_dir,
            pipeline_dir / "pipeline_codon_multipanel.png",
            mode="codon",
        )
    stage_runtimes["pipeline"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    benchmark_outputs = run_benchmark_config(benchmark, benchmark_dir)
    stage_runtimes["benchmark"] = time.perf_counter() - stage_started

    audit_checks = (
        ("reverse-complement", "permutation", "collision", "mutation", "redundancy", "codon")
        if seq_type == "dna"
        else ("contracts", "permutation", "mutation", "redundancy")
    )
    stage_started = time.perf_counter()
    audit_tables = build_encoding_audit_tables(
        EncodingAuditConfig(
            input_path=input_path,
            seq_type=seq_type,
            axis_scheme_id=window_scheme.scheme_id,
            codon_axis_scheme_id=(
                codon_scheme.scheme_id if codon_scheme else "codon-product-v1"
            ),
            checks=audit_checks,
            window_size=extraction.window_size,
            step=extraction.step,
            kmer_k=extraction.kmer_k,
            epsilon=extraction.epsilon,
            max_ambiguous_fraction=extraction.max_ambiguous_fraction,
            codon_table=extraction.codon_table,
            output_format=benchmark.evaluation.output_format,
            random_seed=benchmark.evaluation.random_seed,
            permutation_samples=config.audit_permutation_samples,
            max_perturbations=config.audit_max_perturbations,
            normalize_codons=extraction.codon_normalize,
        )
    )
    write_outputs(audit_tables, audit_dir, benchmark.evaluation.output_format)
    audit_plots = plot_encoding_audit_outputs(audit_tables, audit_dir)
    audit_multipanel = audit_dir / "encoding_audit_multipanel.png"
    if audit_multipanel not in audit_plots:
        raise RuntimeError("The encoding audit did not produce its multipanel report.")
    stage_runtimes["audit"] = time.perf_counter() - stage_started

    promoted = {
        "pipeline_window_plot": _promote(pipeline_window_plot, root),
        "benchmark_plot": _promote(benchmark_outputs["plot"], root),
        "benchmark_report": _promote(benchmark_outputs["report"], root),
        "audit_plot": _promote(audit_multipanel, root),
    }
    if pipeline_codon_plot is not None:
        promoted["pipeline_codon_plot"] = _promote(pipeline_codon_plot, root)
    fingerprints = _find_table(pipeline_dir, "sequence_fingerprints")
    if fingerprints is not None:
        promoted["sequence_fingerprints"] = _promote(fingerprints, root)

    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    finished_utc = datetime.now(timezone.utc)
    warnings = [*override_warnings, *_benchmark_warnings(benchmark_outputs.get("manifest"))]
    manifest = root / "analysis_manifest.json"
    output_inventory = _output_inventory(root, exclude={manifest})
    fold_artifact = next(
        (
            artifact
            for artifact in output_inventory
            if artifact["path"].startswith("benchmark/benchmark_folds.")
        ),
        None,
    )
    manifest_payload = {
        "format": "fanoseq-complete-analysis",
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_id": hashlib.sha256(
            json.dumps(resolved_plan, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16],
        "started_utc": started_utc.isoformat(),
        "finished_utc": finished_utc.isoformat(),
        "fanoseq_version": fanoseq.__version__,
        "git_commit": _git_commit(),
        "resolved_analysis_plan": resolved_plan,
        "benchmark_configuration": benchmark.to_dict(),
        "input_hashes": {
            "fasta_sha256": _sha256(input_path),
            "metadata_sha256": _sha256(benchmark.dataset.table),
            "benchmark_config_sha256": _sha256(config.benchmark_config.resolve()),
        },
        "random_seeds": {
            "evaluation": benchmark.evaluation.random_seed,
            "null_models": benchmark.null_models.random_seed,
            "audit": benchmark.evaluation.random_seed,
        },
        "fold_assignments": fold_artifact,
        "unsafe_split_fallback": {
            "enabled": benchmark.evaluation.allow_unsafe_split_fallback,
            "used": any("UNSAFE split fallback used" in warning for warning in warnings),
        },
        "runtime_seconds": {
            **{name: round(value, 6) for name, value in stage_runtimes.items()},
            "total": round((finished_utc - started_utc).total_seconds(), 6),
        },
        "peak_traced_memory_bytes": int(peak_memory),
        "software": _software_versions(),
        "table_dimensions": {
            "pipeline": _in_memory_dimensions(pipeline_tables),
            "audit": _in_memory_dimensions(audit_tables),
            "written": {
                item["path"]: item["table_dimensions"]
                for item in output_inventory
                if item.get("table_dimensions") is not None
            },
        },
        "outputs": output_inventory,
        "subdirectories": {"pipeline": "pipeline", "benchmark": "benchmark", "audit": "audit"},
        "main_files": {name: path.name for name, path in promoted.items()},
        "warnings": warnings,
        "biological_evidence_status": (
            "not_established_by_software_execution; biological claims require registered, "
            "held-out empirical datasets and the documented statistical acceptance criteria"
        ),
    }
    manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    return {**promoted, "manifest": manifest}


def _resolve_analysis_config(
    config: CompleteAnalysisConfig,
) -> tuple[BenchmarkConfig, list[str]]:
    benchmark = load_benchmark_config(config.benchmark_config)
    input_path = config.input_path.resolve()
    requested_seq_type = config.seq_type.lower() if config.seq_type is not None else None
    requested = {
        "input_path": (str(benchmark.dataset.fasta.resolve()), str(input_path)),
        "seq_type": (benchmark.dataset.seq_type, requested_seq_type),
        "window_size": (benchmark.feature_extraction.window_size, config.window_size),
        "step": (benchmark.feature_extraction.step, config.step),
        "kmer_k": (benchmark.feature_extraction.kmer_k, config.kmer_k),
    }
    mismatches = {
        name: (current, supplied)
        for name, (current, supplied) in requested.items()
        if supplied is not None and supplied != current
    }
    if mismatches and not config.allow_config_override:
        details = "; ".join(
            f"{name}: manifest={current!r}, CLI={supplied!r}"
            for name, (current, supplied) in mismatches.items()
        )
        raise ValueError(
            "Complete analysis extraction settings must come from one resolved plan. "
            f"Conflicting overrides: {details}. Use --allow-config-override to make the "
            "override explicit and record it in the manifest."
        )
    if not mismatches:
        return benchmark, []

    dataset = replace(
        benchmark.dataset,
        fasta=input_path,
        seq_type=(requested_seq_type or benchmark.dataset.seq_type),  # type: ignore[arg-type]
    )
    extraction = replace(
        benchmark.feature_extraction,
        window_size=config.window_size or benchmark.feature_extraction.window_size,
        step=config.step or benchmark.feature_extraction.step,
        kmer_k=config.kmer_k or benchmark.feature_extraction.kmer_k,
    )
    warning = "Explicit configuration override: " + "; ".join(
        f"{name}={supplied!r} (manifest was {current!r})"
        for name, (current, supplied) in mismatches.items()
    )
    return replace(benchmark, dataset=dataset, feature_extraction=extraction), [warning]


def _promote(source: Path, output_root: Path) -> Path:
    destination = output_root / source.name
    shutil.copy2(source, destination)
    return destination


def _find_table(directory: Path, stem: str) -> Path | None:
    for suffix in (".tsv", ".parquet"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_inventory(root: Path, *, exclude: set[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path in exclude:
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "table_dimensions": _table_dimensions(path),
            }
        )
    return rows


def _table_dimensions(path: Path) -> dict[str, int] | None:
    if path.suffix == ".tsv":
        with path.open("r", encoding="utf-8") as handle:
            header = handle.readline().rstrip("\r\n")
            rows = sum(1 for _ in handle)
        return {"rows": rows, "columns": len(header.split("\t")) if header else 0}
    if path.suffix == ".parquet":
        table = pd.read_parquet(path)
        return {"rows": int(len(table)), "columns": int(len(table.columns))}
    return None


def _in_memory_dimensions(tables: dict[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    return {
        name: {"rows": int(len(table)), "columns": int(len(table.columns))}
        for name, table in tables.items()
    }


def _benchmark_warnings(manifest_path: Path | None) -> list[str]:
    if manifest_path is None or not manifest_path.exists():
        return ["Benchmark manifest was not available for complete-analysis provenance."]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = payload.get("split_assignments", {})
    if split.get("unsafe_fallback_used"):
        return [
            "UNSAFE split fallback used: " + "; ".join(split.get("fallback_reasons", []))
        ]
    return []


def _software_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "fanoseq": fanoseq.__version__,
    }
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "typer", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _git_commit() -> str:
    git_dir = Path(__file__).resolve().parents[2] / ".git"
    if git_dir.is_file():
        marker = git_dir.read_text(encoding="utf-8").strip()
        if marker.startswith("gitdir:"):
            git_dir = (git_dir.parent / marker.split(":", 1)[1].strip()).resolve()
    head = git_dir / "HEAD"
    if not head.exists():
        return "unknown"
    value = head.read_text(encoding="utf-8").strip()
    if not value.startswith("ref:"):
        return value
    reference = value.split(":", 1)[1].strip()
    ref_path = git_dir / reference
    if ref_path.exists():
        return ref_path.read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if packed.exists():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and line.endswith(f" {reference}"):
                return line.split(" ", 1)[0]
    return "unknown"
