"""Configuration objects for reproducible FanoSeq benchmarks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

BenchmarkTask = Literal[
    "classification",
    "binary_classification",
    "multiclass_classification",
    "regression",
    "clustering",
]
SplitStrategy = Literal["stratified", "group", "stratified_group", "kfold"]
SequenceType = Literal["dna", "protein"]
OutputFormat = Literal["tsv", "parquet"]

DEFAULT_FEATURES = (
    "fanoseq_components",
    "fanoseq_products",
    "fanoseq_commutators",
    "fanoseq_associators",
    "fanoseq_fano_lines",
    "fanoseq_fingerprints",
    "nucleotide_composition",
    "kmer",
    "fcgr",
    "codon_usage",
    "real_polynomial_control",
    "random_projection_control",
)


@dataclass(frozen=True)
class DatasetConfig:
    """Input metadata and FASTA settings."""

    table: Path
    fasta: Path
    id_column: str = "sequence_id"
    target_column: str | None = "label"
    group_column: str | None = None
    task: BenchmarkTask = "classification"
    seq_type: SequenceType = "dna"
    parent_column: str | None = None


@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Feature-extraction controls reused from the main FanoSeq pipeline."""

    window_size: int = 12
    step: int = 1
    kmer_k: int = 3
    frame: int | Literal["all"] = 0
    codon_table: str = "standard"
    max_ambiguous_fraction: float = 0.0
    include_stop_codons: bool = True
    codon_normalize: bool = False
    window_axis_scheme: str | None = None
    codon_axis_scheme: str | None = None
    random_projection_dim: int | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    """Cross-validation, model, and statistical-test settings."""

    outer_folds: int = 5
    inner_folds: int = 3
    repeats: int = 1
    random_seed: int = 42
    split_strategy: SplitStrategy = "stratified_group"
    primary_metric: str = "balanced_accuracy"
    models: tuple[str, ...] = ()
    output_format: OutputFormat = "tsv"
    sequence_similarity_audit: bool = False
    sequence_similarity_threshold: float = 0.95
    paired_permutation_rounds: int = 999
    run_ablations: bool = True
    feature_selection: bool = False
    n_jobs: int = 1


@dataclass(frozen=True)
class NullModelConfig:
    """Optional null models and representation perturbations."""

    sequence_nulls: tuple[str, ...] = ()
    representation_nulls: tuple[str, ...] = ()
    label_permutations: int = 0
    random_seed: int | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    """Resolved benchmark configuration."""

    dataset: DatasetConfig
    features: tuple[str, ...] = DEFAULT_FEATURES
    feature_extraction: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    null_models: NullModelConfig = field(default_factory=NullModelConfig)
    config_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable resolved configuration."""
        payload = asdict(self)
        payload["dataset"]["table"] = str(self.dataset.table)
        payload["dataset"]["fasta"] = str(self.dataset.fasta)
        payload["features"] = list(self.features)
        payload["evaluation"]["models"] = list(self.evaluation.models)
        payload["null_models"]["sequence_nulls"] = list(self.null_models.sequence_nulls)
        payload["null_models"]["representation_nulls"] = list(
            self.null_models.representation_nulls
        )
        payload["config_path"] = str(self.config_path) if self.config_path else None
        return payload


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load a YAML or JSON benchmark manifest."""
    config_path = Path(path)
    raw = _load_mapping(config_path)
    base_dir = config_path.parent
    config = parse_benchmark_config(raw, base_dir=base_dir, config_path=config_path)
    validate_benchmark_config(config)
    return config


def parse_benchmark_config(
    raw: dict[str, Any],
    *,
    base_dir: Path | None = None,
    config_path: Path | None = None,
) -> BenchmarkConfig:
    """Parse a benchmark mapping into dataclasses."""
    base = base_dir or Path.cwd()
    dataset_raw = _require_mapping(raw, "dataset")
    evaluation_raw = dict(raw.get("evaluation", {}))
    extraction_raw = dict(raw.get("feature_extraction", raw.get("extraction", {})))
    null_raw = dict(raw.get("null_models", {}))
    features_raw = raw.get("features", DEFAULT_FEATURES)
    if not isinstance(features_raw, list | tuple):
        raise ValueError("features must be a list of feature-set names.")

    dataset = DatasetConfig(
        table=_resolve_path(dataset_raw.get("table"), base, "dataset.table"),
        fasta=_resolve_path(dataset_raw.get("fasta"), base, "dataset.fasta"),
        id_column=str(dataset_raw.get("id_column", "sequence_id")),
        target_column=_optional_str(dataset_raw.get("target_column", "label")),
        group_column=_optional_str(dataset_raw.get("group_column")),
        task=_task(str(dataset_raw.get("task", "classification"))),
        seq_type=_seq_type(str(dataset_raw.get("seq_type", "dna"))),
        parent_column=_optional_str(dataset_raw.get("parent_column")),
    )
    extraction = FeatureExtractionConfig(
        window_size=int(extraction_raw.get("window_size", 12)),
        step=int(extraction_raw.get("step", 1)),
        kmer_k=int(extraction_raw.get("kmer_k", 3)),
        frame=_frame(extraction_raw.get("frame", 0)),
        codon_table=str(extraction_raw.get("codon_table", "standard")),
        max_ambiguous_fraction=float(extraction_raw.get("max_ambiguous_fraction", 0.0)),
        include_stop_codons=bool(extraction_raw.get("include_stop_codons", True)),
        codon_normalize=bool(extraction_raw.get("codon_normalize", False)),
        window_axis_scheme=_optional_str(extraction_raw.get("window_axis_scheme")),
        codon_axis_scheme=_optional_str(extraction_raw.get("codon_axis_scheme")),
        random_projection_dim=_optional_int(extraction_raw.get("random_projection_dim")),
    )
    evaluation = EvaluationConfig(
        outer_folds=int(evaluation_raw.get("outer_folds", 5)),
        inner_folds=int(evaluation_raw.get("inner_folds", 3)),
        repeats=int(evaluation_raw.get("repeats", 1)),
        random_seed=int(evaluation_raw.get("random_seed", 42)),
        split_strategy=_split_strategy(
            str(evaluation_raw.get("split_strategy", "stratified_group"))
        ),
        primary_metric=str(evaluation_raw.get("primary_metric", "balanced_accuracy")),
        models=tuple(str(item) for item in evaluation_raw.get("models", ())),
        output_format=_output_format(str(evaluation_raw.get("output_format", "tsv"))),
        sequence_similarity_audit=bool(
            evaluation_raw.get("sequence_similarity_audit", False)
        ),
        sequence_similarity_threshold=float(
            evaluation_raw.get("sequence_similarity_threshold", 0.95)
        ),
        paired_permutation_rounds=int(evaluation_raw.get("paired_permutation_rounds", 999)),
        run_ablations=bool(evaluation_raw.get("run_ablations", True)),
        feature_selection=bool(evaluation_raw.get("feature_selection", False)),
        n_jobs=int(evaluation_raw.get("n_jobs", 1)),
    )
    null_models = NullModelConfig(
        sequence_nulls=tuple(str(item) for item in null_raw.get("sequence_nulls", ())),
        representation_nulls=tuple(str(item) for item in null_raw.get("representation_nulls", ())),
        label_permutations=int(null_raw.get("label_permutations", 0)),
        random_seed=_optional_int(null_raw.get("random_seed")),
    )
    return BenchmarkConfig(
        dataset=dataset,
        features=tuple(str(item) for item in features_raw),
        feature_extraction=extraction,
        evaluation=evaluation,
        null_models=null_models,
        config_path=config_path,
    )


def validate_benchmark_config(config: BenchmarkConfig) -> None:
    """Validate resolved config values that do not require reading the dataset."""
    if config.evaluation.outer_folds < 2:
        raise ValueError("evaluation.outer_folds must be at least 2.")
    if config.evaluation.inner_folds < 2:
        raise ValueError("evaluation.inner_folds must be at least 2.")
    if config.evaluation.repeats < 1:
        raise ValueError("evaluation.repeats must be at least 1.")
    if config.feature_extraction.window_size <= 0:
        raise ValueError("feature_extraction.window_size must be > 0.")
    if config.feature_extraction.step <= 0:
        raise ValueError("feature_extraction.step must be > 0.")
    if config.feature_extraction.kmer_k <= 0:
        raise ValueError("feature_extraction.kmer_k must be > 0.")
    if not config.features:
        raise ValueError("At least one feature set must be requested.")
    if config.dataset.task != "clustering" and not config.dataset.target_column:
        raise ValueError("Supervised benchmark tasks require dataset.target_column.")


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark config does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        loaded = json.loads(text)
    else:
        loaded = _load_yaml(text)
    if not isinstance(loaded, dict):
        raise ValueError("Benchmark config must be a mapping at the top level.")
    return loaded


def _load_yaml(text: str) -> Any:
    try:
        import yaml

        return yaml.safe_load(text)
    except ModuleNotFoundError:
        return _load_simple_yaml(text)


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by bundled benchmark examples."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]
    last_key_at_indent: dict[int, str] = {}
    for raw_line in text.splitlines():
        line_without_comment = raw_line.split("#", 1)[0].rstrip()
        if not line_without_comment.strip():
            continue
        indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))
        line = line_without_comment.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]
        if line.startswith("- "):
            if not isinstance(container, list):
                parent = stack[-2][1]
                key = last_key_at_indent[stack[-1][0]]
                new_list: list[Any] = []
                if not isinstance(parent, dict):
                    raise ValueError("Unsupported YAML nesting.")
                parent[key] = new_list
                stack[-1] = (stack[-1][0], new_list)
                container = new_list
            container.append(_parse_scalar(line[2:].strip()))
            continue
        if ":" not in line or not isinstance(container, dict):
            raise ValueError("Unsupported YAML syntax; install PyYAML for full YAML support.")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        last_key_at_indent[indent] = key
        if value:
            container[key] = _parse_scalar(value)
        else:
            nested: dict[str, Any] = {}
            container[key] = nested
            stack.append((indent, nested))
    return root


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return dict(value)


def _resolve_path(value: Any, base: Path, field_name: str) -> Path:
    if value is None:
        raise ValueError(f"{field_name} is required.")
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _task(value: str) -> BenchmarkTask:
    normalized = value.lower()
    aliases = {
        "binary": "classification",
        "multiclass": "multiclass_classification",
        "unsupervised": "clustering",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {
        "classification",
        "binary_classification",
        "multiclass_classification",
        "regression",
        "clustering",
    }
    if normalized not in allowed:
        raise ValueError(f"Unsupported benchmark task: {value}")
    return cast(BenchmarkTask, normalized)


def _seq_type(value: str) -> SequenceType:
    normalized = value.lower()
    if normalized not in {"dna", "protein"}:
        raise ValueError("dataset.seq_type must be either 'dna' or 'protein'.")
    return cast(SequenceType, normalized)


def _split_strategy(value: str) -> SplitStrategy:
    normalized = value.lower()
    if normalized not in {"stratified", "group", "stratified_group", "kfold"}:
        raise ValueError(f"Unsupported split strategy: {value}")
    return cast(SplitStrategy, normalized)


def _output_format(value: str) -> OutputFormat:
    normalized = value.lower()
    if normalized not in {"tsv", "parquet"}:
        raise ValueError("evaluation.output_format must be 'tsv' or 'parquet'.")
    return cast(OutputFormat, normalized)


def _frame(value: Any) -> int | Literal["all"]:
    if str(value).lower() == "all":
        return "all"
    frame = int(value)
    if frame not in {0, 1, 2}:
        raise ValueError("feature_extraction.frame must be 0, 1, 2, or all.")
    return frame
