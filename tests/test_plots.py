from pathlib import Path

import pandas as pd

from fanoseq.plots import plot_multipanel


def test_plot_multipanel_window_outputs_png(tmp_path: Path) -> None:
    output_dir = tmp_path / "fanoseq_out"
    output_dir.mkdir()
    pd.DataFrame(
        {
            "sequence_id": ["s1", "s1", "s1"],
            "position": [0, 1, 2],
            "gc_content": [0.4, 0.5, 0.6],
            "mono_entropy": [0.8, 0.9, 1.0],
            "valid_fraction": [1.0, 1.0, 1.0],
            **{f"e{i}": [float(i), float(i + 0.1), float(i + 0.2)] for i in range(8)},
        }
    ).to_csv(output_dir / "window_octonions.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "sequence_id": ["s1", "s1"],
            "position": [0, 1],
            "transition_score": [0.2, 0.5],
        }
    ).to_csv(output_dir / "octonion_products.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "sequence_id": ["s1"],
            "position": [0],
            "associator_score": [0.3],
        }
    ).to_csv(output_dir / "octonion_triplets.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "sequence_id": ["s1", "s1"],
            "mode": ["window", "window"],
            "position": [0, 1],
            "fano_line": ["(1,2,3)", "(1,2,3)"],
            "line_contribution_norm": [0.1, 0.4],
        }
    ).to_csv(output_dir / "fano_interactions.tsv", sep="\t", index=False)

    output = plot_multipanel(output_dir, tmp_path / "multipanel.png")

    assert output.exists()
    assert output.stat().st_size > 0
