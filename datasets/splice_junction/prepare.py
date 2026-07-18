"""Download and prepare the public UCI splice-junction dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/69/"
    "molecular%2Bbiology%2Bsplice%2Bjunction%2Bgene%2Bsequences.zip"
)
SOURCE_SHA256 = "3e7ce5dcbeec8c221f57dda495611b9d6ec9525551f445419f5c74cc38067e4e"
LABELS = {
    "EI": "exon_intron_donor",
    "IE": "intron_exon_acceptor",
    "N": "neither",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Existing official UCI ZIP archive.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "prepared",
    )
    parser.add_argument("--max-per-class", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_per_class <= 0:
        raise SystemExit("--max-per-class must be > 0")

    raw_dir = Path(__file__).resolve().parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = args.archive or raw_dir / "uci-splice-junction.zip"
    if args.archive is None and not archive.exists():
        urllib.request.urlretrieve(SOURCE_URL, archive)
    archive_hash = _sha256(archive)
    if archive_hash != SOURCE_SHA256:
        raise SystemExit(
            f"Archive SHA256 mismatch: expected {SOURCE_SHA256}, observed {archive_hash}"
        )

    rows = _read_archive(archive)
    source_counts = Counter(row["source_label"] for row in rows)
    rows = [row for row in rows if set(row["sequence"]) <= set("ACGT")]
    rows = _deduplicate_sequences(rows)
    prepared = _balanced_sample(rows, args.max_per_class, args.random_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = args.output_dir / "sequences.fasta"
    metadata_path = args.output_dir / "metadata.tsv"
    _write_prepared(prepared, fasta_path, metadata_path)
    _write_hashes(args.output_dir / "input_hashes.tsv", archive, fasta_path, metadata_path)
    (args.output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "source_url": SOURCE_URL,
                "source_archive_sha256": archive_hash,
                "source_class_counts": dict(sorted(source_counts.items())),
                "prepared_class_counts": dict(
                    sorted(Counter(row["label"] for row in prepared).items())
                ),
                "prepared_records": len(prepared),
                "max_per_class": args.max_per_class,
                "random_seed": args.random_seed,
                "filters": ["ACGT-only", "exact-sequence deduplication"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _read_archive(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        with archive.open("splice.data") as binary_handle:
            lines = (line.decode("ascii") for line in binary_handle)
            reader = csv.reader(lines)
            rows: list[dict[str, str]] = []
            for source_label, instance_name, sequence in reader:
                source_label = source_label.strip()
                instance_name = instance_name.strip()
                sequence = sequence.strip().upper()
                source_gene = re.split(r"-(?:DONOR|ACCEPTOR|NEG)-", instance_name)[0]
                rows.append(
                    {
                        "source_label": source_label,
                        "label": LABELS[source_label],
                        "source_instance": instance_name,
                        "source_gene": source_gene,
                        "sequence": sequence,
                    }
                )
    return rows


def _deduplicate_sequences(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    retained: dict[str, dict[str, str]] = {}
    for row in sorted(rows, key=lambda item: (item["source_instance"], item["source_label"])):
        retained.setdefault(row["sequence"], row)
    return list(retained.values())


def _balanced_sample(
    rows: list[dict[str, str]], max_per_class: int, random_seed: int
) -> list[dict[str, str]]:
    rng = random.Random(random_seed)
    sampled: list[dict[str, str]] = []
    for source_label in LABELS:
        candidates = [row for row in rows if row["source_label"] == source_label]
        if len(candidates) < max_per_class:
            raise SystemExit(
                f"Class {source_label} has only {len(candidates)} usable unique records"
            )
        sampled.extend(rng.sample(candidates, max_per_class))
    return sorted(sampled, key=lambda row: (row["source_label"], row["source_instance"]))


def _write_prepared(
    rows: list[dict[str, str]], fasta_path: Path, metadata_path: Path
) -> None:
    with fasta_path.open("w", encoding="ascii", newline="\n") as fasta:
        for index, row in enumerate(rows, start=1):
            row["sequence_id"] = f"uci_splice_{index:04d}"
            fasta.write(f">{row['sequence_id']}\n{row['sequence']}\n")
    with metadata_path.open("w", encoding="utf-8", newline="") as metadata:
        columns = ["sequence_id", "label", "source_gene", "source_instance", "length"]
        writer = csv.DictWriter(metadata, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sequence_id": row["sequence_id"],
                    "label": row["label"],
                    "source_gene": row["source_gene"],
                    "source_instance": row["source_instance"],
                    "length": len(row["sequence"]),
                }
            )


def _write_hashes(path: Path, archive: Path, fasta: Path, metadata: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("file", "sha256"))
        for file_path in (archive, fasta, metadata):
            writer.writerow((file_path.name, _sha256(file_path)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
