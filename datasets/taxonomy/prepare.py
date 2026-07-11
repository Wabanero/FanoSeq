"""Normalize a curated taxonomy FASTA and metadata table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

REQUIRED_COLUMNS = ("sequence_id", "taxon_label", "heldout_taxon")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, type=Path, help="Curated FASTA input.")
    parser.add_argument("--metadata", required=True, type=Path, help="Curated metadata TSV.")
    parser.add_argument("--output-dir", default=Path("prepared"), type=Path)
    args = parser.parse_args()

    rows = _read_tsv(args.metadata)
    _validate_metadata(rows)
    fasta_ids = _fasta_ids(args.fasta)
    missing = sorted({row["sequence_id"] for row in rows} - fasta_ids)
    if missing:
        raise SystemExit(f"Metadata IDs absent from FASTA: {', '.join(missing[:5])}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_fasta = args.output_dir / "sequences.fasta"
    out_metadata = args.output_dir / "metadata.tsv"
    shutil.copyfile(args.fasta, out_fasta)
    shutil.copyfile(args.metadata, out_metadata)
    _write_hashes(args.output_dir / "input_hashes.tsv", out_fasta, out_metadata)


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
        for column in REQUIRED_COLUMNS[1:]:
            if not row[column]:
                raise SystemExit(f"Missing {column} for {sequence_id}")


def _fasta_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                ids.add(line[1:].strip().split()[0])
    if not ids:
        raise SystemExit("FASTA contains no records.")
    return ids


def _write_hashes(path: Path, fasta: Path, metadata: Path) -> None:
    rows = [
        ("fasta_sha256", _sha256(fasta), str(fasta)),
        ("metadata_sha256", _sha256(metadata), str(metadata)),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("key", "sha256", "path"))
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
