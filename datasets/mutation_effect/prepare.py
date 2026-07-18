"""Generate controlled mutation-effect benchmark inputs from a curated CDS FASTA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

GENETIC_CODE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}
TRANSITIONS = {"A": "G", "G": "A", "C": "T", "T": "C"}
TRANSVERSIONS = {"A": "C", "C": "A", "G": "T", "T": "G"}
CONSERVATIVE_GROUPS = [
    {"A", "G", "P", "S", "T"},
    {"D", "E", "N", "Q"},
    {"H", "K", "R"},
    {"I", "L", "M", "V"},
    {"F", "W", "Y"},
    {"C"},
]
CODONS_BY_AA: dict[str, list[str]] = {}
for _codon, _aa in GENETIC_CODE.items():
    CODONS_BY_AA.setdefault(_aa, []).append(_codon)


@dataclass(frozen=True)
class FastaRecord:
    sequence_id: str
    sequence: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cds-fasta", required=True, type=Path, help="Curated in-frame CDS FASTA.")
    parser.add_argument("--output-dir", default=Path("prepared"), type=Path)
    parser.add_argument("--min-length", default=300, type=int)
    parser.add_argument("--max-length", default=3000, type=int)
    parser.add_argument("--max-records", default=500, type=int)
    parser.add_argument("--random-seed", default=42, type=int)
    args = parser.parse_args()

    records = [
        record
        for record in _read_fasta(args.cds_fasta)
        if _is_usable_cds(record.sequence, min_length=args.min_length)
    ][: args.max_records]
    if not records:
        raise SystemExit("No usable CDS records found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_fasta = args.output_dir / "sequences.fasta"
    out_metadata = args.output_dir / "metadata.tsv"

    metadata_rows: list[dict[str, object]] = []
    rng = random.Random(args.random_seed)
    with out_fasta.open("w", encoding="utf-8") as fasta_handle:
        for record in records:
            base = _clean_cds(record.sequence, max_length=args.max_length)
            for perturbation_class, sequence, edit_count, preserves_translation, preserves_codons in _perturbations(base, rng):
                sequence_id = f"{record.sequence_id}|{perturbation_class}"
                fasta_handle.write(f">{sequence_id}\n{_wrap(sequence)}\n")
                metadata_rows.append(
                    {
                        "sequence_id": sequence_id,
                        "perturbation_class": perturbation_class,
                        "parent_cds_id": record.sequence_id,
                        "edit_count": edit_count,
                        "length": len(sequence),
                        "preserves_translation": preserves_translation,
                        "preserves_codon_counts": preserves_codons,
                    }
                )
    _write_metadata(out_metadata, metadata_rows)
    _write_hashes(args.output_dir / "input_hashes.tsv", out_fasta, out_metadata)
    (args.output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "source_cds_fasta": str(args.cds_fasta.resolve()),
                "source_cds_sha256": _sha256(args.cds_fasta),
                "random_seed": args.random_seed,
                "min_length": args.min_length,
                "max_length": args.max_length,
                "max_records": args.max_records,
                "parent_records": len(records),
                "prepared_records": len(metadata_rows),
                "controls": ["codon_order_shuffle", "synonymous_recoding"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _read_fasta(path: Path) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    current_id: str | None = None
    chunks: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append(FastaRecord(current_id, "".join(chunks)))
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        records.append(FastaRecord(current_id, "".join(chunks)))
    return records


def _is_usable_cds(sequence: str, *, min_length: int) -> bool:
    cleaned = "".join(base for base in sequence.upper() if base in {"A", "C", "G", "T"})
    return len(cleaned) >= min_length and len(cleaned) >= 3


def _clean_cds(sequence: str, *, max_length: int) -> str:
    cleaned = "".join(base for base in sequence.upper() if base in {"A", "C", "G", "T"})
    cleaned = cleaned[:max_length]
    return cleaned[: len(cleaned) - (len(cleaned) % 3)]


def _perturbations(
    sequence: str, rng: random.Random
) -> list[tuple[str, str, int, bool, bool]]:
    codon_shuffled = _codon_order_shuffle(sequence, rng)
    synonymous_recoded = _synonymous_recode(sequence)
    candidates = [
        ("reference", sequence, 0, True, True),
        ("synonymous_substitution", _synonymous(sequence), 1, True, False),
        ("synonymous_recoding", synonymous_recoded, _codon_edit_count(sequence, synonymous_recoded), True, False),
        ("codon_order_shuffle", codon_shuffled, _codon_edit_count(sequence, codon_shuffled), False, True),
        ("transition", _single_base(sequence, TRANSITIONS), 1, False, False),
        ("transversion", _single_base(sequence, TRANSVERSIONS), 1, False, False),
        ("conservative_amino_acid_change", _amino_acid_change(sequence, conservative=True), 1, False, False),
        ("radical_amino_acid_change", _amino_acid_change(sequence, conservative=False), 1, False, False),
        ("premature_stop", _premature_stop(sequence), 1, False, False),
        ("frameshift", _frameshift(sequence), 1, False, False),
    ]
    return [
        (label, mutated, edit_count, preserves_translation, preserves_codons)
        for label, mutated, edit_count, preserves_translation, preserves_codons in candidates
        if (
            mutated is not None
            and mutated
            and (label == "reference" or mutated != sequence)
        )
    ]


def _synonymous(sequence: str) -> str | None:
    for start, codon in _codons(sequence):
        aa = GENETIC_CODE.get(codon)
        synonyms = [candidate for candidate in CODONS_BY_AA.get(aa or "", []) if candidate != codon]
        if aa not in {None, "*"} and synonyms:
            return _replace(sequence, start, synonyms[0])
    return None


def _synonymous_recode(sequence: str) -> str | None:
    recoded: list[str] = []
    changed = False
    for _, codon in _codons(sequence):
        aa = GENETIC_CODE.get(codon)
        alternatives = [candidate for candidate in CODONS_BY_AA.get(aa or "", []) if candidate != codon]
        if aa not in {None, "*"} and alternatives:
            recoded.append(sorted(alternatives)[0])
            changed = True
        else:
            recoded.append(codon)
    result = "".join(recoded)
    return result if changed else None


def _codon_order_shuffle(sequence: str, rng: random.Random) -> str | None:
    codons = [codon for _, codon in _codons(sequence)]
    shuffled = codons.copy()
    rng.shuffle(shuffled)
    result = "".join(shuffled)
    return result if result != sequence else None


def _codon_edit_count(original: str, changed: str | None) -> int:
    if changed is None:
        return 0
    return sum(left != right for left, right in zip(_codons(original), _codons(changed)))


def _single_base(sequence: str, mapping: dict[str, str]) -> str | None:
    for index, base in enumerate(sequence):
        replacement = mapping.get(base)
        if replacement is not None:
            return f"{sequence[:index]}{replacement}{sequence[index + 1:]}"
    return None


def _amino_acid_change(sequence: str, *, conservative: bool) -> str | None:
    for start, codon in _codons(sequence):
        aa = GENETIC_CODE.get(codon)
        if aa in {None, "*"}:
            continue
        target = _replacement_aa(aa, conservative=conservative)
        if target == aa:
            continue
        target_codons = CODONS_BY_AA.get(target, [])
        if target_codons:
            return _replace(sequence, start, target_codons[0])
    return None


def _replacement_aa(aa: str, *, conservative: bool) -> str:
    group = next((item for item in CONSERVATIVE_GROUPS if aa in item), {aa})
    if conservative:
        choices = sorted(group - {aa})
        return choices[0] if choices else aa
    for candidate in sorted(set(CODONS_BY_AA) - {"*"} - group):
        return candidate
    return aa


def _premature_stop(sequence: str) -> str | None:
    for start, codon in _codons(sequence):
        if GENETIC_CODE.get(codon) not in {None, "*"}:
            return _replace(sequence, start, "TAA")
    return None


def _frameshift(sequence: str) -> str | None:
    if len(sequence) <= 4:
        return None
    index = 3 if len(sequence) > 6 else 1
    return f"{sequence[:index]}{sequence[index + 1:]}"


def _codons(sequence: str) -> list[tuple[int, str]]:
    return [(index, sequence[index : index + 3]) for index in range(0, len(sequence) - 2, 3)]


def _replace(sequence: str, start: int, codon: str) -> str:
    return f"{sequence[:start]}{codon}{sequence[start + 3:]}"


def _wrap(sequence: str, width: int = 80) -> str:
    return "\n".join(sequence[index : index + width] for index in range(0, len(sequence), width))


def _write_metadata(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sequence_id",
                "perturbation_class",
                "parent_cds_id",
                "edit_count",
                "length",
                "preserves_translation",
                "preserves_codon_counts",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


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
