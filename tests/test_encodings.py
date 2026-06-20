import numpy as np

from fanoseq.encodings import (
    build_codon_embedding_initialization,
    encode_dna_base_context,
    encode_gf8_base,
    encode_octonion_walk,
    encode_protein_groups,
    list_encoding_specs,
)
from fanoseq.genetic_code import get_genetic_code


def test_encoding_registry_lists_core_encodings() -> None:
    specs = list_encoding_specs()
    assert {"dna-base-context", "gf8-base", "octonion-walk", "codon-embedding-init"}.issubset(
        set(specs["name"])
    )


def test_dna_base_context_uses_current_and_previous_base_channels() -> None:
    table = encode_dna_base_context("AC")
    assert len(table) == 2
    assert table.loc[0, "e0"] == 1.0
    assert table.loc[1, "e1"] == 1.0
    assert table.loc[1, "e4"] == 1.0


def test_gf8_base_and_octonion_walk_are_deterministic() -> None:
    encoded = encode_gf8_base("A")
    assert encoded.components[0] == 1.0
    assert encoded.components[1] == 1.0
    walk = encode_octonion_walk("AC")
    assert walk is not None
    assert np.allclose(walk.components, [0, 0, 0, 1, 0, 0, 0, 0])


def test_codon_embedding_initializer_covers_all_codons() -> None:
    table = build_codon_embedding_initialization(get_genetic_code("standard"))
    assert len(table) == 64
    assert {"codon", "amino_acid", "e0", "e7"}.issubset(table.columns)
    assert bool(table.loc[table["codon"] == "ATG", "is_start"].iloc[0]) is True


def test_protein_group_encoding_uses_interpretable_axes() -> None:
    table = encode_protein_groups("ACDP")
    assert list(table["group_label"]) == [
        "small_flexible",
        "sulfur_unique",
        "negative",
        "kink_rigid",
    ]
    assert table.loc[0, "e1"] == 1.0
    assert table.loc[3, "e0"] == 1.0
