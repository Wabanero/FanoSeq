from pathlib import Path

import numpy as np
import pandas as pd

from fanoseq.distances import build_distance_matrix, build_neighbor_table
from fanoseq.fano_motifs import amino_acid_axis_map, fano_triad_counts
from fanoseq.fingerprints import build_sequence_fingerprints
from fanoseq.tensor_export import table_to_component_tensor, write_tensor_npz


def test_sequence_fingerprints_merge_window_and_codon_summaries() -> None:
    tables = {
        "window_sequence_summary": pd.DataFrame(
            {
                "sequence_id": ["s1"],
                "seq_type": ["dna"],
                "n_windows": [3],
                "mean_e0": [1.0],
            }
        ),
        "codon_usage_sequence_summary": pd.DataFrame(
            {
                "sequence_id": ["s1", "s1"],
                "frame": [0, 1],
                "n_valid_codons": [4, 2],
                "codon_entropy": [0.5, 0.25],
            }
        ),
    }
    fingerprints = build_sequence_fingerprints(tables)
    assert len(fingerprints) == 1
    assert "window_n_windows" in fingerprints.columns
    assert "codon_n_valid_codons_mean" in fingerprints.columns


def test_distance_and_neighbor_tables() -> None:
    fingerprints = pd.DataFrame(
        {
            "sequence_id": ["a", "b", "c"],
            "f1": [0.0, 1.0, 2.0],
            "f2": [0.0, 1.0, 4.0],
        }
    )
    matrix = build_distance_matrix(fingerprints, metric="euclidean", standardize=False)
    neighbors = build_neighbor_table(fingerprints, metric="euclidean", k=1, standardize=False)
    assert matrix.shape == (3, 3)
    assert np.isclose(matrix.loc["a", "a"], 0.0)
    assert len(neighbors) == 3


def test_tensor_export_shapes_and_writes_npz(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "sequence_id": ["s1", "s1", "s2"],
            "position": [0, 1, 0],
            **{f"e{i}": [float(i), float(i + 1), float(i + 2)] for i in range(8)},
        }
    )
    tensor, sequence_ids, lengths = table_to_component_tensor(table)
    assert tensor.shape == (2, 8, 2)
    assert sequence_ids == ["s1", "s2"]
    assert lengths.tolist() == [2, 1]
    path = write_tensor_npz(table, tmp_path / "x.npz")
    loaded = np.load(path, allow_pickle=True)
    assert loaded["x"].shape == (2, 8, 2)


def test_fano_triad_counts_identifies_line_triples() -> None:
    counts = fano_triad_counts("AVL", amino_acid_axis_map())
    assert len(counts) == 1
    assert bool(counts.loc[0, "is_fano_line"]) is False
    line_counts = fano_triad_counts("AVF", amino_acid_axis_map())
    assert bool(line_counts.loc[0, "is_fano_line"]) is True
