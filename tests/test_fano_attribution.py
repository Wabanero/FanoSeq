import numpy as np

from fanoseq.fano_attribution import fano_line_attribution


def test_fano_line_formula_for_selected_lines() -> None:
    x = np.zeros(8)
    y = np.zeros(8)
    x[1] = 2
    x[2] = 3
    x[4] = 5
    y[1] = 7
    y[2] = 11
    y[4] = 13

    rows = fano_line_attribution(
        x,
        y,
        sequence_id="s",
        mode="window",
        seq_type="dna",
        frame="NA",
        position=0,
        left_object="left",
        right_object="right",
    )
    assert len(rows) == 7
    line_123 = next(row for row in rows if row["fano_line"] == "(1,2,3)")
    line_145 = next(row for row in rows if row["fano_line"] == "(1,4,5)")
    assert line_123["pair_ab_to_c"] == x[1] * y[2] - x[2] * y[1]
    assert line_145["pair_ab_to_c"] == x[1] * y[4] - x[4] * y[1]
    assert all(row["line_contribution_norm"] >= 0 for row in rows)


def test_axis_labels_for_modes() -> None:
    x = np.ones(8)
    y = np.arange(8, dtype=float)
    dna = fano_line_attribution(
        x,
        y,
        sequence_id="s",
        mode="window",
        seq_type="dna",
        frame="NA",
        position=0,
        left_object="l",
        right_object="r",
    )
    protein = fano_line_attribution(
        x,
        y,
        sequence_id="s",
        mode="window",
        seq_type="protein",
        frame="NA",
        position=0,
        left_object="l",
        right_object="r",
    )
    codon = fano_line_attribution(
        x,
        y,
        sequence_id="s",
        mode="codon",
        seq_type="dna",
        frame=0,
        position=0,
        left_object="ATG",
        right_object="GCC",
    )
    assert dna[0]["axis_a_label"] == "purine/pyrimidine balance"
    assert protein[0]["axis_a_label"] == "hydrophobicity"
    assert codon[0]["axis_a_label"] == "base RY property"


def test_scalar_and_same_axis_terms_are_not_included() -> None:
    scalar_x = np.zeros(8)
    scalar_y = np.zeros(8)
    scalar_x[0] = 10
    scalar_y[1] = 3
    scalar_rows = fano_line_attribution(
        scalar_x,
        scalar_y,
        sequence_id="s",
        mode="window",
        seq_type="dna",
        frame="NA",
        position=0,
        left_object="l",
        right_object="r",
    )
    assert all(row["line_contribution_norm"] == 0 for row in scalar_rows)

    same_axis_x = np.zeros(8)
    same_axis_y = np.zeros(8)
    same_axis_x[1] = 4
    same_axis_y[1] = 5
    same_axis_rows = fano_line_attribution(
        same_axis_x,
        same_axis_y,
        sequence_id="s",
        mode="window",
        seq_type="dna",
        frame="NA",
        position=0,
        left_object="l",
        right_object="r",
    )
    assert all(row["line_contribution_norm"] == 0 for row in same_axis_rows)

