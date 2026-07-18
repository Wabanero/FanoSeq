"""Genetic-code helpers with an explicit standard-code fallback."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

BASES = "TCAG"

STANDARD_CODE = {
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

STANDARD_START_CODONS = {"ATG"}
STANDARD_STOP_CODONS = {"TAA", "TAG", "TGA"}


@dataclass(frozen=True)
class GeneticCode:
    """Codon table used for codon annotation and RSCU."""

    name: str
    codon_to_aa: dict[str, str]
    start_codons: set[str]
    stop_codons: set[str]

    def amino_acid(self, codon: str) -> str:
        """Return the amino-acid symbol, '*' for stop, or 'X' if unknown."""
        return self.codon_to_aa.get(codon.upper(), "X")

    def is_start(self, codon: str) -> bool:
        """Return True if codon is annotated as a start codon."""
        return codon.upper() in self.start_codons

    def is_stop(self, codon: str) -> bool:
        """Return True if codon is annotated as a stop codon."""
        return codon.upper() in self.stop_codons

    def synonymous_codons(self, amino_acid: str) -> list[str]:
        """Return codons in the synonymous family for an amino acid."""
        aa = amino_acid.upper()
        return sorted(codon for codon, value in self.codon_to_aa.items() if value == aa)


def get_genetic_code(table: str | int = "standard") -> GeneticCode:
    """Return a requested genetic code without silently substituting another table."""
    try:
        from Bio.Data import CodonTable
    except ModuleNotFoundError as exc:
        if str(table).lower().replace("_", " ") not in {"standard", "standard code", "1"}:
            raise ValueError(
                f"Genetic code {table!r} requires Biopython; only the built-in standard "
                "table is available."
            ) from exc
        return _standard_genetic_code()

    try:
        if isinstance(table, int) or str(table).isdigit():
            bio_table = CodonTable.unambiguous_dna_by_id[int(table)]
        else:
            table_key = str(table).lower().replace("_", " ")
            if table_key in {"standard", "standard code"}:
                bio_table = CodonTable.unambiguous_dna_by_id[1]
            else:
                matches = {
                    name.lower(): candidate
                    for candidate in CodonTable.unambiguous_dna_by_id.values()
                    for name in candidate.names
                    if name is not None
                }
                bio_table = matches[table_key]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unknown or unsupported genetic code table: {table!r}.") from exc
    codon_to_aa = {codon: aa for codon, aa in bio_table.forward_table.items()}
    for codon in bio_table.stop_codons:
        codon_to_aa[codon] = "*"
    return GeneticCode(
        name=bio_table.names[0],
        codon_to_aa=codon_to_aa,
        start_codons=set(bio_table.start_codons),
        stop_codons=set(bio_table.stop_codons),
    )


def _standard_genetic_code() -> GeneticCode:
    """Return the bundled NCBI table 1 used only when Biopython is unavailable."""
    return GeneticCode(
        name="standard",
        codon_to_aa=dict(STANDARD_CODE),
        start_codons=set(STANDARD_START_CODONS),
        stop_codons=set(STANDARD_STOP_CODONS),
    )


def all_standard_codons() -> list[str]:
    """Return the 64 DNA codons in lexical A/C/G/T order."""
    return ["".join(codon) for codon in product("ACGT", repeat=3)]
