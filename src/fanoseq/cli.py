"""Command-line interface for FanoSeq."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from fanoseq.pipeline import RunConfig, run_analysis

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


if __name__ == "__main__":
    app()
