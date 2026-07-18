from __future__ import annotations

from pathlib import Path

import yaml

from fanoseq.benchmark.config import load_benchmark_config

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
REQUIRED_CONTROLS = {
    "kmer",
    "fcgr",
    "fanoseq_components",
    "fanoseq_products",
    "fanoseq_commutators",
    "fanoseq_associators",
    "fanoseq_fano_lines",
    "real_polynomial_control",
    "antisymmetric_control",
    "randomized_fano_structure",
}


def test_dataset_registry_manifests_are_parseable() -> None:
    registry = _load_yaml(DATASETS / "registry.yaml")
    studies = registry["studies"]
    assert {study["dataset_id"] for study in studies} == {
        "coding-noncoding-v1",
        "taxonomy-v1",
        "mutation-effect-v1",
        "uci-splice-junction-v1",
    }

    for study in studies:
        manifest_path = DATASETS / study["path"]
        benchmark_path = DATASETS / study["benchmark_config"]
        manifest = _load_yaml(manifest_path)
        config = load_benchmark_config(benchmark_path)

        assert manifest["dataset_id"] == study["dataset_id"]
        assert manifest["inputs"]["fasta_sha256"]
        assert manifest["inputs"]["metadata_sha256"]
        assert manifest["split"]["seed"] == config.evaluation.random_seed
        assert manifest["split"]["group_column"] == config.dataset.group_column
        assert REQUIRED_CONTROLS.issubset(set(config.features))
        assert (manifest_path.parent / "prepare.py").exists()
        assert (manifest_path.parent / "splits.tsv").exists()


def test_methods_report_contains_decisive_comparison() -> None:
    report = (ROOT / "docs" / "methods_report.md").read_text(encoding="utf-8")
    assert "raw descriptors" in report
    assert "ordinary polynomial interactions" in report
    assert "ordinary antisymmetric interactions" in report
    assert "randomized Fano-like structure" in report
    assert "fixed FanoSeq" in report


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded
