from fanoseq.genetic_code import get_genetic_code
from fanoseq.matrix_genetics import (
    build_codon_matrix_entries,
    build_dyadic_shift_summary,
    build_hadamard_spectrum,
    build_matrix_genetics_tables,
    build_root_degeneracy,
    codon_to_matrix_position,
)


def test_canonical_matrix_has_64_codon_entries() -> None:
    code = get_genetic_code("standard")
    entries = build_codon_matrix_entries(code)
    assert len(entries) == 64
    assert set(entries["row"]) == set(range(8))
    assert set(entries["column"]) == set(range(8))
    assert codon_to_matrix_position("AAA") == (0, 0)


def test_root_degeneracy_flags_strong_roots() -> None:
    roots = build_root_degeneracy(get_genetic_code("standard"))
    gc_root = roots[roots["root"] == "GC"].iloc[0]
    assert bool(gc_root["root_is_strong"]) is True
    assert gc_root["unique_amino_acids"] == "A"


def test_hadamard_and_dyadic_shift_tables_have_expected_sizes() -> None:
    code = get_genetic_code("standard")
    spectrum = build_hadamard_spectrum(code)
    shifts = build_dyadic_shift_summary(code)
    assert len(spectrum) == 64
    assert len(shifts) == 63
    assert {"basis_index", "coefficient", "energy"}.issubset(spectrum.columns)
    assert {"shift", "same_amino_acid_fraction"}.issubset(shifts.columns)


def test_matrix_genetics_table_bundle_names() -> None:
    tables = build_matrix_genetics_tables(get_genetic_code("standard"))
    assert {
        "codon_matrix_entries",
        "codon_degeneracy_roots",
        "codon_hadamard_spectrum",
        "codon_dyadic_shifts",
        "gf8_codon_labels",
    } == set(tables)
