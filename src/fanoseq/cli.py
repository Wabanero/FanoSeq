"""Command-line interface for FanoSeq."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console

from fanoseq.distances import build_distance_matrix, build_neighbor_table
from fanoseq.encodings import (
    build_codon_embedding_initialization,
    iter_octonion_walks,
    list_encoding_specs,
)
from fanoseq.fasta import read_fasta
from fanoseq.fano_motifs import amino_acid_axis_map, dna_base_axis_map, fano_triad_counts
from fanoseq.genetic_code import get_genetic_code
from fanoseq.io import write_outputs
from fanoseq.matrix_genetics import build_matrix_genetics_tables
from fanoseq.pipeline import RunConfig, run_analysis
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
        parsed_frame: int | str = "all" if frame == "all" else int(frame)
        config = RunConfig(
            input_path=input_path,
            seq_type=seq_type.lower(),  # type: ignore[arg-type]
            mode=mode.lower(),  # type: ignore[arg-type]
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


if __name__ == "__main__":
    app()
