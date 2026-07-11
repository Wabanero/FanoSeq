from pathlib import Path

import pandas as pd

from fanoseq.fano_plane import (
    FANO_LINE_KEYS,
    FanoPlane,
    build_fano_line_features,
    build_fano_line_stability,
    fano_plane_tables,
    plot_fano_plane,
)


def _example_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sequence_id": "s1",
                "mode": "window",
                "seq_type": "dna",
                "axis_scheme_id": "dna-window-v1",
                "frame": "NA",
                "fano_line": "(1,2,3)",
                "line_label": "base chemistry triad",
                "line_contribution_norm": 3.0,
            },
            {
                "sequence_id": "s1",
                "mode": "window",
                "seq_type": "dna",
                "axis_scheme_id": "dna-window-v1",
                "frame": "NA",
                "fano_line": "(1,2,3)",
                "line_label": "base chemistry triad",
                "line_contribution_norm": 1.0,
            },
            {
                "sequence_id": "s1",
                "mode": "window",
                "seq_type": "dna",
                "axis_scheme_id": "dna-window-v1",
                "frame": "NA",
                "fano_line": "(2,4,6)",
                "line_label": "GC-skew-complexity triad",
                "line_contribution_norm": 2.0,
            },
        ]
    )


def test_fano_plane_tables_with_axis_scheme_labels() -> None:
    tables = fano_plane_tables("dna-window-v1")
    assert {
        "fano_plane_points",
        "fano_plane_lines",
        "fano_plane_incidence",
        "fano_plane_pairs",
    } == set(tables)
    assert len(tables["fano_plane_points"]) == 7
    assert len(tables["fano_plane_lines"]) == 7
    assert len(tables["fano_plane_incidence"]) == 21
    assert "base chemistry triad" in set(tables["fano_plane_lines"]["line_label"])


def test_fano_plane_pair_table_encodes_oriented_products() -> None:
    pair_table = FanoPlane("dna-window-v1").pair_table()
    ordered = pair_table[pair_table["is_ordered_product"].astype(bool)]
    assert len(ordered) == 42
    e1e2 = ordered[(ordered["left_axis"] == 1) & (ordered["right_axis"] == 2)].iloc[0]
    e2e1 = ordered[(ordered["left_axis"] == 2) & (ordered["right_axis"] == 1)].iloc[0]
    assert e1e2["output_axis"] == 3
    assert e1e2["product_sign"] == 1
    assert e2e1["output_axis"] == 3
    assert e2e1["product_sign"] == -1


def test_fano_line_features_profile_and_axis_loads() -> None:
    features = build_fano_line_features(_example_interactions())
    assert len(features) == 1
    row = features.iloc[0]
    assert row["dominant_fano_line"] == "(1,2,3)"
    assert row["dominant_fano_line_label"] == "base chemistry triad"
    assert row["fano_line_1_2_3_share"] == 4.0 / 6.0
    assert row["fano_line_2_4_6_share"] == 2.0 / 6.0
    assert row["axis_e1_incident_share"] == 4.0 / 6.0
    assert row["axis_e2_incident_share"] == 1.0
    assert set(FANO_LINE_KEYS).issuperset({"(1,2,3)", "(2,4,6)"})


def test_fano_line_stability_is_deterministic() -> None:
    left = build_fano_line_stability(_example_interactions(), n_bootstrap=25, random_seed=7)
    right = build_fano_line_stability(_example_interactions(), n_bootstrap=25, random_seed=7)
    pd.testing.assert_frame_equal(left, right)
    assert left["dominant_line_stability"].between(0.0, 1.0).all()
    assert left["mean_profile_cosine_to_full"].between(0.0, 1.0).all()


def test_plot_fano_plane_writes_png(tmp_path: Path) -> None:
    output = plot_fano_plane(tmp_path / "fano_plane.png", axis_scheme_id="dna-window-v1", size=700)
    assert output.exists()
    assert output.stat().st_size > 1000
