from pathlib import Path

from fanoseq.workflow import CompleteAnalysisConfig, run_complete_analysis


def test_complete_analysis_organizes_subdirectories_and_main_files(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "complete"
    outputs = run_complete_analysis(
        CompleteAnalysisConfig(
            input_path=project_root / "examples" / "benchmark_sequences.fasta",
            benchmark_config=project_root / "examples" / "benchmark.yaml",
            output_dir=output_dir,
            seq_type="dna",
            window_size=9,
            step=3,
            audit_permutation_samples=1,
            audit_max_perturbations=8,
        )
    )

    assert {"pipeline", "benchmark", "audit"}.issubset(
        path.name for path in output_dir.iterdir() if path.is_dir()
    )
    assert outputs["manifest"] == output_dir / "analysis_manifest.json"
    assert outputs["pipeline_window_plot"].stat().st_size > 0
    assert outputs["benchmark_plot"].stat().st_size > 0
    assert outputs["audit_plot"].stat().st_size > 0
    assert outputs["sequence_fingerprints"].exists()
