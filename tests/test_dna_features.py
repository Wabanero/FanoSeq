import numpy as np

from fanoseq.dna_features import (
    DNA_ALPHABET,
    encode_dna_window,
    reverse_complement,
    reverse_complement_similarity,
    shannon_entropy,
)


def test_reverse_complement() -> None:
    assert reverse_complement("ACGTTA") == "TAACGT"


def test_entropy_range() -> None:
    entropy = shannon_entropy("ACGTACGT", DNA_ALPHABET)
    assert 0 <= entropy <= 1


def test_encoding_components_and_scalar_mass() -> None:
    encoded = encode_dna_window("ACGTACGTAC")
    assert encoded is not None
    octonion, metadata = encoded
    assert len(octonion.components) == 8
    assert octonion.components[0] == metadata["valid_fraction"]
    assert "mono_entropy" in metadata
    assert not np.isclose(octonion.components[0], metadata["mono_entropy"])


def test_known_simple_sequence_behavior() -> None:
    low = encode_dna_window("AAAAAAAAAA")
    high = encode_dna_window("ACGTACGTAC")
    repeat = encode_dna_window("ATATATATAT")
    varied = encode_dna_window("ACGTAGCTTA")
    assert low is not None and high is not None and repeat is not None and varied is not None
    assert low[1]["mono_entropy"] < high[1]["mono_entropy"]
    assert repeat[1]["mono_entropy"] > low[1]["mono_entropy"]
    assert repeat[0].components[6] < varied[0].components[6]


def test_reverse_complement_symmetry_range() -> None:
    score = 2 * reverse_complement_similarity("ACGTACGT") - 1
    assert -1 <= score <= 1

