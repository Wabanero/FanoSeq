import pandas as pd
import pytest

from fanoseq.biological_context import (
    align_tracks_to_windows,
    validate_genomic_context,
)


def test_genomic_context_and_tracks_align_on_exact_windows() -> None:
    context = pd.DataFrame(
        {
            "sequence_id": ["s1"],
            "genome_build": ["GRCh38"],
            "chromosome": ["chr1"],
            "start": [100],
            "end": [120],
            "strand": ["+"],
            "feature_type": ["promoter"],
        }
    )
    assert validate_genomic_context(context).loc[0, "feature_type"] == "promoter"
    windows = pd.DataFrame(
        {"sequence_id": ["s1", "s1"], "start": [0, 10], "end": [10, 20]}
    )
    tracks = pd.DataFrame(
        {
            "sequence_id": ["s1", "s1", "s1", "s1"],
            "start": [0, 0, 10, 10],
            "end": [10, 10, 20, 20],
            "track_name": ["accessibility", "methylation", "accessibility", "methylation"],
            "value": [1.0, 0.2, 2.0, 0.4],
        }
    )
    aligned = align_tracks_to_windows(windows, tracks)
    assert aligned["accessibility"].tolist() == [1.0, 2.0]
    assert aligned["methylation"].tolist() == [0.2, 0.4]


def test_track_alignment_fails_on_missing_window_values() -> None:
    windows = pd.DataFrame(
        {"sequence_id": ["s1", "s1"], "start": [0, 10], "end": [10, 20]}
    )
    tracks = pd.DataFrame(
        {
            "sequence_id": ["s1"],
            "start": [0],
            "end": [10],
            "track_name": ["accessibility"],
            "value": [1.0],
        }
    )
    with pytest.raises(ValueError, match="lack synchronized track values"):
        align_tracks_to_windows(windows, tracks)
