"""Command-line interface for FanoSeq."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from fanoseq.algebra import (
    multiplication_table,
    structure_constants,
    validation_passed,
    validation_report,
)
from fanoseq.axis_schemes import (
    axis_scheme_tables,
    get_axis_scheme,
    list_axis_definitions,
    list_axis_schemes,
    validate_axis_scheme_definitions,
)
from fanoseq.baselines import build_baseline_tables
from fanoseq.distances import build_distance_matrix, build_neighbor_table
from fanoseq.encodings import (
    build_codon_embedding_initialization,
    iter_octonion_walks,
    list_encoding_specs,
)
from fanoseq.fasta import read_fasta
from fanoseq.fano_motifs import amino_acid_axis_map, dna_base_axis_map, fano_triad_counts
from fanoseq.fano_plane import (
    build_fano_line_features,
    build_fano_line_stability,
    fano_plane_tables,
    plot_fano_plane,
)
from fanoseq.genetic_code import get_genetic_code
from fanoseq.io import write_outputs
from fanoseq.matrix_genetics import build_matrix_genetics_tables
from fanoseq.pipeline import RunConfig, run_analysis
from fanoseq.plots import plot_multipanel
from fanoseq.tensor_export import read_table, write_tensor_npz

app = typer.Typer(help="FanoSeq: sequence trajectories in Fano-structured octonion space.")
console = Console()


@app.callback()
def main() -> None:
    """Sequence trajectories in Fano-structured octonion space."""


@app.command()
def run(
    input_path: Path = typer.Option(..., "--input", "-i", help="Path to FASTA file."),
    seq_type: str = typer.Option(..., "--seq-type", help="Either 'dna' or 'protein'."),
    mode: str = typer.Option("window", "--mode", help="Analysis mode: window, codon, or both."),
    window_size: Optional[int] = typer.Option(
        None, "--window-size", help="Sliding-window size. Required for window and both modes."
    ),
    step: int = typer.Option(1, "--step", help="Sliding-window step size."),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for TSV outputs."),
    kmer_k: int = typer.Option(2, "--kmer-k", help="k for k-mer entropy descriptors."),
    epsilon: float = typer.Option(1e-9, "--epsilon", help="Small value to avoid division by zero."),
    max_ambiguous_fraction: float = typer.Option(
        0.0,
        "--max-ambiguous-fraction",
        help="Maximum ambiguous-character fraction allowed in a unit.",
    ),
    frame: str = typer.Option("0", "--frame", help="Reading frame: 0, 1, 2, or all."),
    codon_table: str = typer.Option("standard", "--codon-table", help="Codon table name or ID."),
    include_partial_codons: bool = typer.Option(
        False,
        "--include-partial-codons",
        help="Include trailing partial codons by padding missing bases as ambiguous.",
    ),
    include_stop_codons: bool = typer.Option(
        True,
        "--include-stop-codons/--exclude-stop-codons",
        help="Include or exclude stop codons in codon mode.",
    ),
    codon_normalize: bool = typer.Option(
        False,
        "--codon-normalize",
        help="Normalize codon octonions to unit norm when possible.",
    ),
    axis_scheme: Optional[str] = typer.Option(
        None,
        "--axis-scheme",
        help="Shortcut for a single runnable scheme in window or codon mode.",
    ),
    window_axis_scheme: Optional[str] = typer.Option(
        None,
        "--window-axis-scheme",
        help="Runnable window axis scheme. Defaults to dna-window-v1 or protein-sequence-v1.",
    ),
    codon_axis_scheme: Optional[str] = typer.Option(
        None,
        "--codon-axis-scheme",
        help="Runnable codon axis scheme. Defaults to codon-product-v1.",
    ),
    output_format: str = typer.Option(
        "tsv",
        "--output-format",
        help="Output storage format: tsv, parquet, or bundle.",
    ),
    summary_only: bool = typer.Option(
        False,
        "--summary-only",
        help="Write compact summary/fingerprint tables instead of row-heavy trajectory tables.",
    ),
    top_k_transitions: Optional[int] = typer.Option(
        None,
        "--top-k-transitions",
        help="Store only the top K strongest transition products and matching Fano rows.",
    ),
    transition_threshold: Optional[float] = typer.Option(
        None,
        "--transition-threshold",
        help="Store only transition products with transition_score >= this value.",
    ),
) -> None:
    """Run FanoSeq on DNA or protein FASTA input."""
    try:
        mode_normalized = mode.lower()
        window_axis_scheme_effective = window_axis_scheme
        codon_axis_scheme_effective = codon_axis_scheme
        if axis_scheme is not None:
            if mode_normalized == "window":
                window_axis_scheme_effective = axis_scheme
            elif mode_normalized == "codon":
                codon_axis_scheme_effective = axis_scheme
            else:
                raise ValueError(
                    "--axis-scheme is ambiguous for --mode both; use "
                    "--window-axis-scheme and --codon-axis-scheme."
                )
        parsed_frame: int | str = "all" if frame == "all" else int(frame)
        config = RunConfig(
            input_path=input_path,
            seq_type=seq_type.lower(),  # type: ignore[arg-type]
            mode=mode_normalized,  # type: ignore[arg-type]
            output_dir=output_dir,
            window_size=window_size,
            step=step,
            kmer_k=kmer_k,
            epsilon=epsilon,
            max_ambiguous_fraction=max_ambiguous_fraction,
            frame=parsed_frame,  # type: ignore[arg-type]
            codon_table=codon_table,
            include_partial_codons=include_partial_codons,
            include_stop_codons=include_stop_codons,
            codon_normalize=codon_normalize,
            window_axis_scheme=window_axis_scheme_effective,
            codon_axis_scheme=codon_axis_scheme_effective,
            output_format=output_format.lower(),  # type: ignore[arg-type]
            summary_only=summary_only,
            top_k_transitions=top_k_transitions,
            transition_threshold=transition_threshold,
        )
        outputs = run_analysis(config)
    except Exception as exc:
        console.print(f"[red]FanoSeq error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]FanoSeq wrote {len(outputs)} {output_format.lower()} table(s) to {output_dir}[/green]"
    )


@app.command("list-encodings")
def list_encodings() -> None:
    """List implemented octonion encoding schemes."""
    table = list_encoding_specs()
    console.print(table.to_string(index=False))


@app.command("list-axis-schemes")
def list_axis_schemes_command() -> None:
    """List versioned biological/computational axis schemes."""
    table = list_axis_schemes()
    console.print(table.to_string(index=False))


@app.command("list-axis-definitions")
def list_axis_definitions_command(
    scheme_id: Optional[str] = typer.Option(
        None,
        "--scheme-id",
        help="Optional axis scheme id. Defaults to all schemes.",
    ),
) -> None:
    """List concrete axis formulas, inputs, scaling, and missing-data policies."""
    try:
        table = list_axis_definitions(scheme_id)
    except Exception as exc:
        console.print(f"[red]FanoSeq error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(table.to_string(index=False))


@app.command("describe-axis-scheme")
def describe_axis_scheme(
    scheme_id: str = typer.Argument(..., help="Axis scheme id, for example dna-window-v1."),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Optional directory for axis-scheme tables."
    ),
    output_format: str = typer.Option(
        "tsv", "--output-format", help="Output storage format when --output-dir is supplied."
    ),
) -> None:
    """Describe one axis scheme and its Fano-line semantics."""
    try:
        scheme = get_axis_scheme(scheme_id)
        tables = axis_scheme_tables(scheme_id)
        console.print(
            f"[bold]{scheme.scheme_id}[/bold] ({scheme.status}, {scheme.representation})\n"
            f"{scheme.recommended_use}\n\n"
            f"[bold]Axes[/bold]\n{tables['axis_scheme_axes'].to_string(index=False)}\n\n"
            f"[bold]Fano lines[/bold]\n"
            f"{tables['axis_scheme_fano_lines'][['fano_line', 'line_label']].to_string(index=False)}\n\n"
            f"[bold]Validation[/bold]\n"
            f"{tables['axis_scheme_validation'][['check_id', 'passed']].to_string(index=False)}"
        )
        if output_dir is not None:
            written = write_outputs(
                tables,
                output_dir,
                output_format.lower(),  # type: ignore[arg-type]
                manifest={
                    "format": "fanoseq-axis-scheme",
                    "scheme_id": scheme.scheme_id,
                    "schema_version": "0.5.0",
                },
            )
            console.print(
                f"[green]FanoSeq wrote {len(written)} axis-scheme table(s) to {output_dir}[/green]"
            )
    except Exception as exc:
        console.print(f"[red]FanoSeq error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("validate-axis-schemes")
def validate_axis_schemes(
    scheme_id: Optional[str] = typer.Option(
        None,
        "--scheme-id",
        help="Optional axis scheme id. Defaults to all schemes.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Optional directory for axis-scheme validation tables.",
    ),
    output_format: str = typer.Option(
        "tsv",
        "--output-format",
        help="Output storage format when --output-dir is supplied.",
    ),
) -> None:
    """Validate completeness and run-readiness of registered axis definitions."""
    try:
        validation = validate_axis_scheme_definitions(scheme_id)
        if output_dir is not None:
            written = write_outputs(
                {"axis_scheme_validation": validation},
                output_dir,
                output_format.lower(),  # type: ignore[arg-type]
                manifest={
                    "format": "fanoseq-axis-scheme-validation",
                    "scheme_id": scheme_id or "all",
                    "schema_version": "0.6.0",
                },
            )
            console.print(
                f"[green]FanoSeq wrote {len(written)} axis validation table(s) to {output_dir}[/green]"
            )
        console.print(validation.to_string(index=False))
        if not validation["passed"].all():
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]FanoSeq error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("fano-plane")
def fano_plane_command(
    axis_scheme_id: Optional[str] = typer.Option(
        None,
        "--axis-scheme",
        help="Optional axis scheme id for biological/computational labels.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Optional directory for Fano-plane point, line, incidence, and pair tables.",
    ),
    output_format: str = typer.Option(
        "tsv",
        "--output-format",
        help="Output storage format when --output-dir is supplied.",
    ),
) -> None:
    """Describe the Fano plane as an explicit object."""
    try:
        tables = fano_plane_tables(axis_scheme_id)
        console.print(
            f"[bold]Fano plane[/bold] ({axis_scheme_id or 'basis'})\n\n"
            f"[bold]Lines[/bold]\n"
            f"{tables['fano_plane_lines'][['fano_line', 'line_label', 'orientation_rule']].to_string(index=False)}\n\n"
            f"[bold]Points[/bold]\n"
            f"{tables['fano_plane_points'][['symbol', 'axis_label', 'degree']].to_string(index=False)}"
        )
        if output_dir is not None:
            written = write_outputs(
                tables,
                output_dir,
                output_format.lower(),  # type: ignore[arg-type]
                manifest={
                    "format": "fanoseq-fano-plane",
                    "axis_scheme_id": axis_scheme_id or "basis",
                    "schema_version": "0.7.0",
                },
            )
            console.print(
                f"[green]FanoSeq wrote {len(written)} Fano-plane table(s) to {output_dir}[/green]"
            )
    except Exception as exc:
        console.print(f"[red]FanoSeq error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("plot-fano-plane")
def plot_fano_plane_command(
    output_path: Path = typer.Option(..., "--output", help="Output PNG path."),
    axis_scheme_id: Optional[str] = typer.Option(
        None,
        "--axis-scheme",
        help="Optional axis scheme id for labels.",
    ),
    size: int = typer.Option(1200, "--size", help="Image size in pixels."),
) -> None:
    """Draw a true seven-point/seven-line Fano-plane PNG."""
    try:
        path = plot_fano_plane(output_path, axis_scheme_id=axis_scheme_id, size=size)
    except Exception as exc:
        console.print(f"[red]FanoSeq error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]FanoSeq wrote Fano-plane plot to {path}[/green]")


@app.command("fano-features")
def fano_features_command(
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for Fano-line features."),
    interactions_path: Optional[Path] = typer.Option(
        None,
        "--interactions",
        help="Path to fano_interactions TSV/CSV/Parquet.",
    ),
    input_dir: Optional[Path] = typer.Option(
        None,
        "--input-dir",
        help="FanoSeq output directory containing fano_interactions.",
    ),
    output_format: str = typer.Option(
        "tsv",
        "--output-format",
        help="Output storage format: tsv, parquet, or bundle.",
    ),
) -> None:
    """Build Fano-line profile features from attribution rows."""
    try:
        interactions = _read_fano_interactions(interactions_path, input_dir)
        features = build_fano_line_features(interactions)
        written = write_outputs(
            {"fano_line_features": features},
            output_dir,
            output_format.lower(),  # type: ignore[arg-type]
            manifest={
                "format": "fanoseq-fano-features",
                "schema_version": "0.7.0",
                "interactions_path": str(interactions_path) if interactions_path else None,
                "input_dir": str(input_dir) if input_dir else None,
            },
        )
    except Exception as exc:
        console.print(f"[red]FanoSeq error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]FanoSeq wrote {len(written)} Fano feature table(s) to {output_dir}[/green]")


@app.command("fano-stability")
def fano_stability_command(
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for Fano-line stability."),
    interactions_path: Optional[Path] = typer.Option(
        None,
        "--interactions",
        help="Path to fano_interactions TSV/CSV/Parquet.",
    ),
    input_dir: Optional[Path] = typer.Option(
        None,
        "--input-dir",
        help="FanoSeq output directory containing fano_interactions.",
    ),
    n_bootstrap: int = typer.Option(100, "--n-bootstrap", help="Bootstrap resamples per group."),
    random_seed: int = typer.Option(0, "--random-seed", help="Deterministic bootstrap seed."),
    output_format: str = typer.Option(
        "tsv",
        "--output-format",
        help="Output storage format: tsv, parquet, or bundle.",
    ),
) -> None:
    """Bootstrap dominant Fano-line profile stability."""
    try:
        interactions = _read_fano_interactions(interactions_path, input_dir)
        stability = build_fano_line_stability(
            interactions,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        written = write_outputs(
            {"fano_line_stability": stability},
            output_dir,
            output_format.lower(),  # type: ignore[arg-type]
            manifest={
                "format": "fanoseq-fano-stability",
                "schema_version": "0.7.0",
                "interactions_path": str(interactions_path) if interactions_path else None,
                "input_dir": str(input_dir) if input_dir else None,
                "n_bootstrap": n_bootstrap,
                "random_seed": random_seed,
            },
        )
    except Exception as exc:
        console.print(f"[red]FanoSeq error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]FanoSeq wrote {len(written)} Fano stability table(s) to {output_dir}[/green]"
    )


@app.command("validate-basis")
def validate_basis(
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Optional directory for validation tables."
    ),
    output_format: str = typer.Option(
        "tsv", "--output-format", help="Output storage format when --output-dir is supplied."
    ),
) -> None:
    """Validate the FanoSeq octonion basis convention and algebra representations."""
    try:
        report = validation_report()
        passed = validation_passed(report)
        if output_dir is not None:
            constants = structure_constants()
            nonzero = np.argwhere(constants != 0.0)
            structure_rows = [
                {
                    "left": int(left),
                    "right": int(right),
                    "basis": int(basis),
                    "coefficient": float(constants[left, right, basis]),
                }
                for left, right, basis in nonzero
            ]
            written = write_outputs(
                {
                    "basis_validation": report,
                    "basis_multiplication_table": multiplication_table(),
                    "structure_constants": pd.DataFrame(structure_rows),
                },
                output_dir,
                output_format.lower(),  # type: ignore[arg-type]
                manifest={
                    "format": "fanoseq-basis-validation",
                    "schema_version": "0.4.0",
                    "passed": passed,
                },
            )
            console.print(
                f"[green]FanoSeq wrote {len(written)} basis validation table(s) to {output_dir}[/green]"
            )
        failed = report[~report["passed"]]
        if passed:
            console.print(f"[green]Basis validation passed ({len(report)} checks).[/green]")
        else:
            console.print(f"[red]Basis validation failed ({len(failed)} failing checks).[/red]")
            console.print(failed.to_string(index=False))
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]FanoSeq error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("baselines")
def baselines(
    input_path: Path = typer.Option(..., "--input", "-i", help="Path to DNA or protein FASTA file."),
    seq_type: str = typer.Option(..., "--seq-type", help="Either 'dna' or 'protein'."),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for baseline tables."),
    kmer_k: int = typer.Option(4, "--kmer-k", help="k for k-mer baseline features."),
    frame: str = typer.Option("0", "--frame", help="DNA reading frame: 0, 1, 2, or all."),
    codon_table: str = typer.Option("standard", "--codon-table", help="Codon table name or ID."),
    output_format: str = typer.Option(
        "tsv", "--output-format", help="Output storage format: tsv, parquet, or bundle."
    ),
) -> None:
    """Write mature baseline features for benchmark comparisons."""
    try:
        seq_type_normalized = seq_type.lower()
        parsed_frame: int | str = "all" if frame == "all" else int(frame)
        genetic_code = get_genetic_code(codon_table) if seq_type_normalized == "dna" else None
        tables = build_baseline_tables(
            read_fasta(input_path),
            seq_type=seq_type_normalized,  # type: ignore[arg-type]
            kmer_k=kmer_k,
            genetic_code=genetic_code,
            frame=parsed_frame,  # type: ignore[arg-type]
        )
        written = write_outputs(
            tables,
            output_dir,
            output_format.lower(),  # type: ignore[arg-type]
            manifest={
                "format": "fanoseq-baselines",
                "input_path": str(input_path),
                "seq_type": seq_type_normalized,
                "kmer_k": kmer_k,
                "frame": parsed_frame,
                "codon_table": codon_table if seq_type_normalized == "dna" else "NA",
                "schema_version": "0.4.0",
            },
        )
    except Exception as exc:
        console.print(f"[red]FanoSeq error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]FanoSeq wrote {len(written)} baseline table(s) to {output_dir}[/green]")


@app.command("matrix-genetics")
def matrix_genetics(
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for matrix-genetics tables."),
    codon_table: str = typer.Option("standard", "--codon-table", help="Codon table name or ID."),
    output_format: str = typer.Option(
        "tsv", "--output-format", help="Output storage format: tsv, parquet, or bundle."
    ),
) -> None:
    """Write codon-matrix, degeneracy, Hadamard, dyadic-shift, and GF(8)-label tables."""
    try:
        genetic_code = get_genetic_code(codon_table)
        tables = build_matrix_genetics_tables(genetic_code)
        written = write_outputs(
            tables,
            output_dir,
            output_format.lower(),  # type: ignore[arg-type]
            manifest={
                "format": "fanoseq-matrix-genetics",
                "codon_table": genetic_code.name,
                "schema_version": "0.3.0",
            },
        )
    except Exception as exc:
        console.print(f"[red]FanoSeq error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]FanoSeq wrote {len(written)} matrix-genetics table(s) to {output_dir}[/green]")


@app.command("codon-embedding-init")
def codon_embedding_init(
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for codon embedding table."),
    codon_table: str = typer.Option("standard", "--codon-table", help="Codon table name or ID."),
    output_format: str = typer.Option(
        "tsv", "--output-format", help="Output storage format: tsv, parquet, or bundle."
    ),
) -> None:
    """Write a 64x8 codon-octonion embedding initializer."""
    try:
        genetic_code = get_genetic_code(codon_table)
        table = build_codon_embedding_initialization(genetic_code)
        written = write_outputs(
            {"codon_embedding_init": table},
            output_dir,
            output_format.lower(),  # type: ignore[arg-type]
            manifest={
                "format": "fanoseq-codon-embedding-init",
                "codon_table": genetic_code.name,
                "schema_version": "0.3.0",
            },
        )
    except Exception as exc:
        console.print(f"[red]FanoSeq error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]FanoSeq wrote {len(written)} codon embedding table(s) to {output_dir}[/green]")


@app.command("octonion-walk")
def octonion_walk(
    input_path: Path = typer.Option(..., "--input", "-i", help="Path to DNA FASTA file."),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for octonion-walk table."),
    k: int = typer.Option(6, "--k", help="DNA k-mer length."),
    step: int = typer.Option(1, "--step", help="Sliding step size."),
    normalize: bool = typer.Option(False, "--normalize", help="Normalize each k-mer walk octonion."),
    output_format: str = typer.Option(
        "tsv", "--output-format", help="Output storage format: tsv, parquet, or bundle."
    ),
) -> None:
    """Write order-sensitive octonion-walk k-mer encodings."""
    try:
        rows: list[dict[str, object]] = []
        for record in read_fasta(input_path):
            for row in iter_octonion_walks(record.sequence, k=k, step=step, normalize=normalize):
                rows.append({"sequence_id": record.id, **row})
        table = pd.DataFrame(rows)
        written = write_outputs(
            {"octonion_walks": table},
            output_dir,
            output_format.lower(),  # type: ignore[arg-type]
            manifest={
                "format": "fanoseq-octonion-walk",
                "input_path": str(input_path),
                "k": k,
                "step": step,
                "normalize": normalize,
                "schema_version": "0.3.0",
            },
        )
    except Exception as exc:
        console.print(f"[red]FanoSeq error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]FanoSeq wrote {len(written)} octonion-walk table(s) to {output_dir}[/green]")


@app.command("export-tensor")
def export_tensor(
    table_path: Path = typer.Option(..., "--table", help="Path to a FanoSeq TSV/CSV/Parquet table."),
    output_path: Path = typer.Option(..., "--output", help="Output .npz archive path."),
    group_column: str = typer.Option("sequence_id", "--group-column", help="Sequence/group column."),
    component_prefix: str = typer.Option("e", "--component-prefix", help="Component column prefix."),
    order_column: str = typer.Option("position", "--order-column", help="Position/order column."),
) -> None:
    """Export a component table as an AI-ready [N, 8, L] NPZ tensor."""
    try:
        table = read_table(table_path)
        path = write_tensor_npz(
            table,
            output_path,
            group_column=group_column,
            component_prefix=component_prefix,
            order_column=order_column,
        )
    except Exception as exc:
        console.print(f"[red]FanoSeq error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]FanoSeq wrote tensor archive to {path}[/green]")


@app.command("distances")
def distances(
    fingerprints_path: Path = typer.Option(
        ..., "--fingerprints", help="Path to sequence_fingerprints TSV/CSV/Parquet."
    ),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for distance outputs."),
    metric: str = typer.Option("cosine", "--metric", help="euclidean, cosine, correlation, or manhattan."),
    k: int = typer.Option(5, "--k", help="Nearest neighbors to keep."),
    standardize: bool = typer.Option(
        True, "--standardize/--no-standardize", help="Standardize numeric features first."
    ),
    output_format: str = typer.Option(
        "tsv", "--output-format", help="Output storage format: tsv, parquet, or bundle."
    ),
) -> None:
    """Build distance and nearest-neighbor tables from sequence fingerprints."""
    try:
        fingerprints = read_table(fingerprints_path)
        matrix = build_distance_matrix(
            fingerprints,
            metric=metric,  # type: ignore[arg-type]
            standardize=standardize,
        )
        matrix.index.name = "sequence_id"
        matrix = matrix.reset_index()
        neighbors = build_neighbor_table(
            fingerprints,
            metric=metric,  # type: ignore[arg-type]
            k=k,
            standardize=standardize,
        )
        written = write_outputs(
            {"distance_matrix": matrix, "nearest_neighbors": neighbors},
            output_dir,
            output_format.lower(),  # type: ignore[arg-type]
            manifest={
                "format": "fanoseq-distances",
                "fingerprints_path": str(fingerprints_path),
                "metric": metric,
                "standardize": standardize,
                "schema_version": "0.3.0",
            },
        )
    except Exception as exc:
        console.print(f"[red]FanoSeq error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]FanoSeq wrote {len(written)} distance table(s) to {output_dir}[/green]")


@app.command("plot-multipanel")
def plot_multipanel_command(
    input_dir: Path = typer.Option(
        ..., "--input-dir", help="Directory containing FanoSeq output tables."
    ),
    output_path: Path = typer.Option(..., "--output", help="Output PNG path."),
    mode: str = typer.Option("auto", "--mode", help="Plot mode: auto, window, or codon."),
    sequence_id: Optional[str] = typer.Option(
        None, "--sequence-id", help="Sequence ID to plot. Defaults to the first sequence."
    ),
    frame: Optional[int] = typer.Option(None, "--frame", help="Codon frame for codon plots."),
    max_points: int = typer.Option(
        500, "--max-points", help="Maximum trajectory points to draw in line panels."
    ),
) -> None:
    """Create a PNG multipanel summary from FanoSeq output tables."""
    try:
        path = plot_multipanel(
            input_dir,
            output_path,
            mode=mode.lower(),  # type: ignore[arg-type]
            sequence_id=sequence_id,
            frame=frame,
            max_points=max_points,
        )
    except Exception as exc:
        console.print(f"[red]FanoSeq error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]FanoSeq wrote multipanel plot to {path}[/green]")


@app.command("fano-triads")
def fano_triads(
    input_path: Path = typer.Option(..., "--input", "-i", help="Path to DNA or protein FASTA file."),
    seq_type: str = typer.Option(..., "--seq-type", help="Either 'dna' or 'protein'."),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for Fano-triad counts."),
    stride: int = typer.Option(1, "--stride", help="Sliding triad stride."),
    output_format: str = typer.Option(
        "tsv", "--output-format", help="Output storage format: tsv, parquet, or bundle."
    ),
) -> None:
    """Count symbol triads whose mapped axes fall on Fano-plane lines."""
    try:
        if seq_type.lower() == "dna":
            axis_map = dna_base_axis_map()
        elif seq_type.lower() == "protein":
            axis_map = amino_acid_axis_map()
        else:
            raise ValueError("--seq-type must be either 'dna' or 'protein'.")
        rows: list[pd.DataFrame] = []
        for record in read_fasta(input_path):
            counts = fano_triad_counts(record.sequence.upper(), axis_map, stride=stride)
            if counts.empty:
                continue
            counts.insert(0, "sequence_id", record.id)
            rows.append(counts)
        table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        written = write_outputs(
            {"fano_triad_counts": table},
            output_dir,
            output_format.lower(),  # type: ignore[arg-type]
            manifest={
                "format": "fanoseq-fano-triads",
                "input_path": str(input_path),
                "seq_type": seq_type.lower(),
                "stride": stride,
                "schema_version": "0.3.0",
            },
        )
    except Exception as exc:
        console.print(f"[red]FanoSeq error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]FanoSeq wrote {len(written)} Fano-triad table(s) to {output_dir}[/green]")


def _read_fano_interactions(
    interactions_path: Path | None,
    input_dir: Path | None,
) -> pd.DataFrame:
    if interactions_path is not None:
        return read_table(interactions_path)
    if input_dir is None:
        raise ValueError("Provide either --interactions or --input-dir.")
    for candidate in (
        input_dir / "fano_interactions.tsv",
        input_dir / "fano_interactions.parquet",
    ):
        if candidate.exists():
            return read_table(candidate)
    raise FileNotFoundError(f"No fano_interactions table found in {input_dir}.")


if __name__ == "__main__":
    app()
