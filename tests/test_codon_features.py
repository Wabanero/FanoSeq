import numpy as np

from fanoseq.codon_features import base_position_octonion, encode_codon, iter_codons
from fanoseq.fasta import FastaRecord
from fanoseq.genetic_code import get_genetic_code
from fanoseq.pipeline import (
    RunConfig,
    build_codon_octonions,
    build_codon_usage,
)


def test_codon_octonion_is_ordered_product() -> None:
    code = get_genetic_code("standard")
    atg = encode_codon("ATG", code)
    gta = encode_codon("GTA", code)
    assert atg is not None and gta is not None
    simple_sum = (
        base_position_octonion("A", 1)
        + base_position_octonion("T", 2)
        + base_position_octonion("G", 3)
    )
    assert len(atg.octonion.components) == 8
    assert not np.allclose(atg.octonion.components, simple_sum.components)
    assert not np.allclose(atg.octonion.components, gta.octonion.components)
    assert atg.metadata["codon_associator_score"] >= 0


def test_standard_codon_annotations() -> None:
    code = get_genetic_code("standard")
    atg = encode_codon("ATG", code)
    assert atg is not None
    assert atg.metadata["amino_acid"] == "M"
    assert atg.metadata["is_start"] is True
    for stop in ("TAA", "TAG", "TGA"):
        encoded = encode_codon(stop, code)
        assert encoded is not None
        assert encoded.metadata["is_stop"] is True


def test_rscu_values_and_frame_all() -> None:
    code = get_genetic_code("standard")
    records = [FastaRecord(id="cds", description="cds", sequence="ATGATGATA")]
    config = RunConfig(
        input_path="unused.fasta",  # type: ignore[arg-type]
        seq_type="dna",
        mode="codon",
        output_dir="unused",  # type: ignore[arg-type]
        frame="all",
    )
    codon_df = build_codon_octonions(records, config, code)
    assert set(codon_df["frame"]) == {0, 1, 2}
    usage = build_codon_usage(codon_df[codon_df["frame"] == 0], code, config)
    atg = usage[(usage["codon"] == "ATG") & (usage["frame"] == 0)].iloc[0]
    ata = usage[(usage["codon"] == "ATA") & (usage["frame"] == 0)].iloc[0]
    assert atg["count"] == 2
    assert atg["rscu"] == 1
    assert ata["count"] == 1
    assert ata["rscu"] == 3


def test_trailing_incomplete_codons_discarded_when_false() -> None:
    codons = list(iter_codons("ATGA", frame=0, include_partial_codons=False))
    assert [codon.codon for codon in codons] == ["ATG"]

