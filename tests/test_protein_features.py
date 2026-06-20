import numpy as np

from fanoseq.protein_features import encode_protein_window


def test_encoding_components_and_scalar_mass() -> None:
    encoded = encode_protein_window("ACDEFGHIKL")
    assert encoded is not None
    octonion, metadata = encoded
    assert len(octonion.components) == 8
    assert octonion.components[0] == metadata["valid_fraction"]
    assert "mono_entropy" in metadata
    assert not np.isclose(octonion.components[0], metadata["mono_entropy"])


def test_hydrophobic_sequence_scores_higher_than_charged_sequence() -> None:
    hydrophobic = encode_protein_window("LLLLVVVVII")
    charged = encode_protein_window("DDDDEEEEKK")
    assert hydrophobic is not None and charged is not None
    assert hydrophobic[0].components[1] > charged[0].components[1]


def test_low_complexity_sequence_has_higher_repeat_score() -> None:
    repeat = encode_protein_window("AAAAAAAAAA")
    varied = encode_protein_window("ACDEFGHIKL")
    assert repeat is not None and varied is not None
    assert repeat[0].components[7] > varied[0].components[7]

