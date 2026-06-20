from fanoseq.baselines import (
    build_baseline_tables,
    build_dna_baseline_tables,
    count_kmers,
    fcgr_coordinates,
)
from fanoseq.fasta import FastaRecord
from fanoseq.genetic_code import get_genetic_code


def test_count_kmers_and_fcgr_coordinates() -> None:
    counts = count_kmers("ACGTAC", 2, "ACGT")
    assert counts["AC"] == 2
    assert counts["CG"] == 1
    assert fcgr_coordinates("AC") == (0, 1)
    assert fcgr_coordinates("GT") == (3, 2)


def test_dna_baselines_include_sequence_kmer_and_codon_tables() -> None:
    records = [FastaRecord(id="s1", description="s1", sequence="ATGATATGA")]
    tables = build_dna_baseline_tables(records, 2, get_genetic_code("standard"), frame=0)
    assert {
        "baseline_sequence_features",
        "baseline_kmer_frequencies",
        "baseline_kmer_feature_matrix",
        "baseline_codon_usage",
    } == set(tables)
    sequence = tables["baseline_sequence_features"].iloc[0]
    assert sequence["sequence_id"] == "s1"
    assert sequence["valid_length"] == 9
    assert "kmer_AT" in tables["baseline_kmer_feature_matrix"].columns
    codon_usage = tables["baseline_codon_usage"]
    atg = codon_usage[codon_usage["codon"] == "ATG"].iloc[0]
    assert atg["count"] == 1
    assert atg["amino_acid"] == "M"


def test_protein_baselines_include_residue_composition() -> None:
    records = [FastaRecord(id="p1", description="p1", sequence="ACDEFG")]
    tables = build_baseline_tables(records, "protein", kmer_k=2)
    assert "baseline_residue_composition" in tables
    assert "baseline_kmer_feature_matrix" in tables
    residue = tables["baseline_residue_composition"]
    assert residue[residue["residue"] == "A"]["count"].iloc[0] == 1
