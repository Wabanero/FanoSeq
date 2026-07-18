from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fanoseq.benchmark.homology import HomologyGroupConfig, prepare_homology_groups


def test_homology_groups_are_complete_and_provenanced(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.tsv"
    cluster_path = tmp_path / "clusters.tsv"
    output_path = tmp_path / "grouped.tsv"
    pd.DataFrame({"sequence_id": ["s1", "s2", "s3"], "label": [0, 0, 1]}).to_csv(
        metadata_path, sep="\t", index=False
    )
    pd.DataFrame(
        {
            "member": ["s1", "s2", "s3"],
            "cluster": ["c1", "c1", "c2"],
        }
    ).to_csv(cluster_path, sep="\t", index=False)

    grouped_path, manifest_path = prepare_homology_groups(
        HomologyGroupConfig(
            metadata_path=metadata_path,
            cluster_table_path=cluster_path,
            output_path=output_path,
            member_column="member",
            cluster_column="cluster",
            clustering_tool="mmseqs2",
            clustering_tool_version="test-version",
            minimum_identity=0.8,
            minimum_coverage=0.8,
        )
    )

    grouped = pd.read_csv(grouped_path, sep="\t")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert grouped["homology_cluster"].tolist() == ["c1", "c1", "c2"]
    assert manifest["clustering"]["tool"] == "mmseqs2"
    assert manifest["n_groups"] == 2
    assert len(manifest["output_sha256"]) == 64


def test_homology_groups_fail_closed_on_missing_assignment(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.tsv"
    cluster_path = tmp_path / "clusters.tsv"
    pd.DataFrame({"sequence_id": ["s1", "s2"]}).to_csv(
        metadata_path, sep="\t", index=False
    )
    pd.DataFrame({"sequence_id": ["s1"], "homology_cluster": ["c1"]}).to_csv(
        cluster_path, sep="\t", index=False
    )
    with pytest.raises(ValueError, match="Missing homology-cluster assignments"):
        prepare_homology_groups(
            HomologyGroupConfig(
                metadata_path=metadata_path,
                cluster_table_path=cluster_path,
                output_path=tmp_path / "out.tsv",
            )
        )
