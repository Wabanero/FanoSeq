"""Cross-validation split utilities and leakage audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from fanoseq.benchmark.config import EvaluationConfig
from fanoseq.benchmark.datasets import BenchmarkDataset


@dataclass(frozen=True)
class FoldSpec:
    """One outer split with explicit train/test indices."""

    repeat: int
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray

    @property
    def split_id(self) -> str:
        """Return stable split identifier."""
        return f"r{self.repeat}_f{self.fold}"


def make_outer_folds(dataset: BenchmarkDataset, config: EvaluationConfig) -> list[FoldSpec]:
    """Create deterministic outer folds for the configured task and split strategy."""
    return _make_folds(
        y=dataset.y,
        groups=dataset.groups,
        task=dataset.task,
        requested_folds=config.outer_folds,
        repeats=config.repeats,
        split_strategy=config.split_strategy,
        random_seed=config.random_seed,
    )


def make_inner_splits(
    y_train: np.ndarray | None,
    groups_train: np.ndarray | None,
    *,
    task: str,
    requested_folds: int,
    split_strategy: str,
    random_seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create local train/validation index pairs for nested tuning."""
    folds = _make_folds(
        y=y_train,
        groups=groups_train,
        task=task,
        requested_folds=requested_folds,
        repeats=1,
        split_strategy=split_strategy,
        random_seed=random_seed,
    )
    return [(fold.train_indices, fold.test_indices) for fold in folds]


def fold_assignment_table(dataset: BenchmarkDataset, folds: Iterable[FoldSpec]) -> pd.DataFrame:
    """Return one row per sequence and outer fold assignment."""
    rows: list[dict[str, object]] = []
    labels = dataset.y
    for fold in folds:
        train_set = set(int(index) for index in fold.train_indices)
        test_set = set(int(index) for index in fold.test_indices)
        for index, sequence_id in enumerate(dataset.sequence_ids):
            if index in train_set:
                split = "train"
            elif index in test_set:
                split = "test"
            else:
                continue
            row: dict[str, object] = {
                "repeat": fold.repeat,
                "fold": fold.fold,
                "split_id": fold.split_id,
                "sequence_id": sequence_id,
                "split": split,
            }
            if labels is not None:
                row["target"] = labels[index]
            if dataset.groups is not None:
                row["group"] = dataset.groups[index]
            rows.append(row)
    return pd.DataFrame(rows)


def leakage_audit_table(
    dataset: BenchmarkDataset,
    folds: Iterable[FoldSpec],
    *,
    sequence_similarity: bool,
    similarity_threshold: float,
) -> pd.DataFrame:
    """Audit group overlap and optional sequence-similarity leakage per fold."""
    rows: list[dict[str, object]] = []
    for fold in folds:
        train_groups = _values_for_indices(dataset.groups, fold.train_indices)
        test_groups = _values_for_indices(dataset.groups, fold.test_indices)
        overlapping_groups = sorted(train_groups & test_groups)
        row: dict[str, object] = {
            "split_id": fold.split_id,
            "repeat": fold.repeat,
            "fold": fold.fold,
            "n_train": int(len(fold.train_indices)),
            "n_test": int(len(fold.test_indices)),
            "group_leakage_detected": bool(overlapping_groups),
            "overlapping_groups": ",".join(overlapping_groups),
            "sequence_similarity_audit": sequence_similarity,
            "max_train_test_identity": np.nan,
            "similarity_threshold": similarity_threshold,
            "sequence_similarity_leakage_detected": False,
            "most_similar_pair": "NA",
        }
        if sequence_similarity:
            identity, pair = _max_train_test_identity(dataset, fold)
            row["max_train_test_identity"] = identity
            row["sequence_similarity_leakage_detected"] = bool(identity >= similarity_threshold)
            row["most_similar_pair"] = pair
        rows.append(row)
    return pd.DataFrame(rows)


def _make_folds(
    *,
    y: np.ndarray | None,
    groups: np.ndarray | None,
    task: str,
    requested_folds: int,
    repeats: int,
    split_strategy: str,
    random_seed: int,
) -> list[FoldSpec]:
    n_samples = len(y) if y is not None else len(groups) if groups is not None else 0
    if n_samples == 0:
        raise ValueError("Cannot create folds for an empty dataset.")
    n_splits = _effective_n_splits(
        n_samples=n_samples,
        y=y,
        groups=groups,
        task=task,
        requested=requested_folds,
        split_strategy=split_strategy,
    )
    if n_splits < 2:
        raise ValueError("At least two folds are required for benchmark evaluation.")

    indices = np.arange(n_samples)
    folds: list[FoldSpec] = []
    for repeat in range(repeats):
        seed = random_seed + repeat
        splitter = _splitter(
            n_splits=n_splits,
            y=y,
            groups=groups,
            task=task,
            split_strategy=split_strategy,
            random_seed=seed,
        )
        try:
            split_iter = splitter.split(indices, y, groups)
            local_splits = list(split_iter)
        except ValueError:
            fallback = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            local_splits = list(fallback.split(indices))
        for fold_number, (train_idx, test_idx) in enumerate(local_splits):
            folds.append(
                FoldSpec(
                    repeat=repeat,
                    fold=fold_number,
                    train_indices=np.asarray(train_idx, dtype=int),
                    test_indices=np.asarray(test_idx, dtype=int),
                )
            )
    return folds


def _effective_n_splits(
    *,
    n_samples: int,
    y: np.ndarray | None,
    groups: np.ndarray | None,
    task: str,
    requested: int,
    split_strategy: str,
) -> int:
    candidates = [requested, n_samples]
    if groups is not None and split_strategy in {"group", "stratified_group"}:
        candidates.append(len(np.unique(groups)))
    if _is_classification(task) and y is not None and split_strategy in {
        "stratified",
        "stratified_group",
    }:
        _, counts = np.unique(y, return_counts=True)
        candidates.append(int(counts.min()))
    return max(1, min(candidates))


def _splitter(
    *,
    n_splits: int,
    y: np.ndarray | None,
    groups: np.ndarray | None,
    task: str,
    split_strategy: str,
    random_seed: int,
) -> KFold | StratifiedKFold | GroupKFold | StratifiedGroupKFold:
    if split_strategy == "stratified_group" and groups is not None and _is_classification(task):
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    if split_strategy in {"group", "stratified_group"} and groups is not None:
        return GroupKFold(n_splits=n_splits)
    if (
        split_strategy in {"stratified", "stratified_group"}
        and y is not None
        and _is_classification(task)
    ):
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)


def _is_classification(task: str) -> bool:
    return task in {"classification", "binary_classification", "multiclass_classification"}


def _values_for_indices(values: np.ndarray | None, indices: np.ndarray) -> set[str]:
    if values is None:
        return set()
    return {str(values[index]) for index in indices}


def _max_train_test_identity(dataset: BenchmarkDataset, fold: FoldSpec) -> tuple[float, str]:
    best_identity = 0.0
    best_pair = "NA"
    for train_index in fold.train_indices:
        train_id = dataset.sequence_ids[int(train_index)]
        train_sequence = dataset.sequences[train_id]
        for test_index in fold.test_indices:
            test_id = dataset.sequence_ids[int(test_index)]
            test_sequence = dataset.sequences[test_id]
            identity = _identity(train_sequence, test_sequence)
            if identity > best_identity:
                best_identity = identity
                best_pair = f"{train_id}|{test_id}"
    return best_identity, best_pair


def _identity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    matches = sum(1 for index in range(length) if left[index] == right[index])
    return matches / max(len(left), len(right))
