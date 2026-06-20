"""FASTA input utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FastaRecord:
    """A minimal FASTA record."""

    id: str
    description: str
    sequence: str


def read_fasta(path: str | Path) -> list[FastaRecord]:
    """Read a FASTA file into records using a small dependency-light parser."""
    fasta_path = Path(path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file does not exist: {fasta_path}")

    records: list[FastaRecord] = []
    current_header: str | None = None
    chunks: list[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    records.append(_build_record(current_header, chunks))
                current_header = line[1:].strip()
                if not current_header:
                    raise ValueError(f"Empty FASTA header at line {line_number}.")
                chunks = []
            else:
                if current_header is None:
                    raise ValueError(
                        f"Found sequence text before the first FASTA header at line {line_number}."
                    )
                chunks.append(line)

    if current_header is not None:
        records.append(_build_record(current_header, chunks))

    if not records:
        raise ValueError(f"No FASTA records found in {fasta_path}.")
    return records


def _build_record(header: str, chunks: list[str]) -> FastaRecord:
    description = header
    record_id = header.split()[0]
    sequence = "".join(chunks).replace(" ", "").replace("\t", "")
    if not sequence:
        raise ValueError(f"FASTA record {record_id!r} has no sequence.")
    return FastaRecord(id=record_id, description=description, sequence=sequence)

