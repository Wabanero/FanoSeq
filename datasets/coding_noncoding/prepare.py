"""Prepare length/GC/chromosome-matched coding and noncoding sequence windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

REQUIRED_COLUMNS = ("sequence_id", "label", "chromosome")
ALLOWED_LABELS = {"coding", "noncoding"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, type=Path, help="Curated GENCODE-derived FASTA.")
    parser.add_argument("--metadata", required=True, type=Path, help="Curated metadata TSV.")
    parser.add_argument("--output-dir", default=Path("prepared"), type=Path)
    parser.add_argument("--min-length", default=200, type=int)
    parser.add_argument("--max-length", default=2000, type=int)
    parser.add_argument("--length-bin", default=100, type=int)
    parser.add_argument("--gc-bin-width", default=0.05, type=float)
    parser.add_argument("--max-per-class-per-bin", default=100, type=int)
    parser.add_argument("--random-seed", default=42, type=int)
    args = parser.parse_args()
    if args.min_length <= 0 or args.max_length < args.min_length:
        raise SystemExit("Require 0 < --min-length <= --max-length.")
    if args.length_bin <= 0 or not 0.0 < args.gc_bin_width <= 1.0:
        raise SystemExit("Require --length-bin > 0 and 0 < --gc-bin-width <= 1.")

    rows = _read_tsv(args.metadata)
    _validate_metadata(rows)
    sequences = _read_fasta(args.fasta)
    eligible: list[dict[str, object]] = []
    rejection_counts: Counter[str] = Counter()
    for row in rows:
        sequence_id = row["sequence_id"]
        sequence = sequences.get(sequence_id)
        if sequence is None:
            rejection_counts["metadata_id_absent_from_fasta"] += 1
            continue
        sequence = sequence.upper()
        if set(sequence) - set("ACGT"):
            rejection_counts["ambiguous_sequence"] += 1
            continue
        if not args.min_length <= len(sequence) <= args.max_length:
            rejection_counts["length_filter"] += 1
            continue
        gc_fraction = (sequence.count("G") + sequence.count("C")) / len(sequence)
        eligible.append(
            {
                **row,
                "sequence": sequence,
                "length": len(sequence),
                "gc_fraction": gc_fraction,
                "length_bin": len(sequence) // args.length_bin,
                "gc_bin": int(gc_fraction / args.gc_bin_width),
            }
        )
    selected = _matched_sample(
        eligible,
        max_per_class=args.max_per_class_per_bin,
        random_seed=args.random_seed,
    )
    if not selected:
        raise SystemExit(
            "No chromosome/length/GC stratum contains both coding and noncoding records."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_fasta = args.output_dir / "sequences.fasta"
    out_metadata = args.output_dir / "metadata.tsv"
    _write_fasta(out_fasta, selected)
    _write_metadata(out_metadata, selected)
    _write_hashes(args.output_dir / "input_hashes.tsv", out_fasta, out_metadata)
    (args.output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "source_fasta": str(args.fasta.resolve()),
                "source_fasta_sha256": _sha256(args.fasta),
                "source_metadata": str(args.metadata.resolve()),
                "source_metadata_sha256": _sha256(args.metadata),
                "filters": {
                    "min_length": args.min_length,
                    "max_length": args.max_length,
                    "ambiguity_policy": "ACGT-only",
                },
                "matching": {
                    "group": "chromosome",
                    "length_bin": args.length_bin,
                    "gc_bin_width": args.gc_bin_width,
                    "max_per_class_per_bin": args.max_per_class_per_bin,
                    "random_seed": args.random_seed,
                },
                "source_rows": len(rows),
                "eligible_rows": len(eligible),
                "prepared_rows": len(selected),
                "prepared_class_counts": dict(
                    sorted(Counter(str(row["label"]) for row in selected).items())
                ),
                "rejection_counts": dict(sorted(rejection_counts.items())),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _matched_sample(
    rows: list[dict[str, object]],
    *,
    max_per_class: int,
    random_seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(random_seed)
    strata: dict[tuple[str, int, int], dict[str, list[dict[str, object]]]] = {}
    for row in rows:
        key = (str(row["chromosome"]), int(row["length_bin"]), int(row["gc_bin"]))
        strata.setdefault(key, {label: [] for label in sorted(ALLOWED_LABELS)})[
            str(row["label"])
        ].append(row)
    selected: list[dict[str, object]] = []
    for key in sorted(strata):
        by_label = strata[key]
        n = min(len(by_label["coding"]), len(by_label["noncoding"]), max_per_class)
        if n == 0:
            continue
        for label in sorted(ALLOWED_LABELS):
            candidates = sorted(by_label[label], key=lambda row: str(row["sequence_id"]))
            selected.extend(rng.sample(candidates, n))
    return sorted(selected, key=lambda row: str(row["sequence_id"]))


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise SystemExit("Metadata TSV is empty.")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise SystemExit(f"Metadata TSV is missing columns: {', '.join(missing)}")
        return [dict(row) for row in reader]


def _validate_metadata(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise SystemExit("Metadata TSV contains no rows.")
    seen: set[str] = set()
    for row in rows:
        sequence_id = row["sequence_id"]
        if sequence_id in seen:
            raise SystemExit(f"Duplicate sequence_id: {sequence_id}")
        seen.add(sequence_id)
        if row["label"] not in ALLOWED_LABELS:
            raise SystemExit(f"Unsupported label for {sequence_id}: {row['label']}")
        if not row["chromosome"]:
            raise SystemExit(f"Missing chromosome for {sequence_id}")


def _read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(chunks)
                current_id = line[1:].split()[0]
                if current_id in sequences:
                    raise SystemExit(f"Duplicate FASTA sequence_id: {current_id}")
                chunks = []
            elif current_id is None:
                raise SystemExit("Sequence text appears before the first FASTA header.")
            else:
                chunks.append(line)
    if current_id is not None:
        sequences[current_id] = "".join(chunks)
    if not sequences:
        raise SystemExit("FASTA contains no records.")
    return sequences


def _write_fasta(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for row in rows:
            handle.write(f">{row['sequence_id']}\n{row['sequence']}\n")


def _write_metadata(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "sequence_id",
        "label",
        "chromosome",
        "length",
        "gc_fraction",
        "length_bin",
        "gc_bin",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def _write_hashes(path: Path, fasta: Path, metadata: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "sha256", "path"))
        writer.writerows(
            (
                ("fasta_sha256", _sha256(fasta), str(fasta)),
                ("metadata_sha256", _sha256(metadata), str(metadata)),
            )
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
