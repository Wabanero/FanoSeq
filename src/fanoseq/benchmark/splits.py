"""Cross-validation split utilities and leakage audits."""

from __future__ import annotations

import warnings
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


class BenchmarkSplitError(ValueError):
    """Raised when the requested scientific split cannot be constructed safely."""


@dataclass(frozen=True)
class FoldSpec:
    """One outer split with explicit train/test indices."""

    repeat: int
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    unsafe_fallback: bool = False
    fallback_reason: str | None = None

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
        allow_unsafe_split_fallback=config.allow_unsafe_split_fallback,
    )


def make_inner_splits(
    y_train: np.ndarray | None,
    groups_train: np.ndarray | None,
    *,
    task: str,
    requested_folds: int,
    split_strategy: str,
    random_seed: int,
    allow_unsafe_split_fallback: bool = False,
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
        allow_unsafe_split_fallback=allow_unsafe_split_fallback,
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
                "unsafe_split_fallback": fold.unsafe_fallback,
                "split_fallback_reason": fold.fallback_reason or "",
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
            "unsafe_split_fallback": fold.unsafe_fallback,
            "split_fallback_reason": fold.fallback_reason or "",
            "sequence_similarity_audit": sequence_similarity,
            "max_train_test_positional_identity": np.nan,
            "similarity_threshold": similarity_threshold,
            "sequence_similarity_leakage_detected": False,
            "most_similar_pair": "NA",
        }
        if sequence_similarity:
            identity, pair = _max_train_test_positional_identity(dataset, fold)
            row["max_train_test_positional_identity"] = identity
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
    allow_unsafe_split_fallback: bool,
) -> list[FoldSpec]:
    n_samples = len(y) if y is not None else len(groups) if groups is not None else 0
    if n_samples == 0:
        raise ValueError("Cannot create folds for an empty dataset.")
    problem = _split_request_problem(
        n_samples=n_samples,
        y=y,
        groups=groups,
        task=task,
        requested_folds=requested_folds,
        split_strategy=split_strategy,
    )
    fallback_reason: str | None = None
    if problem is not None:
        if not allow_unsafe_split_fallback:
            raise BenchmarkSplitError(problem)
        fallback_reason = problem
        warnings.warn(
            f"UNSAFE split fallback enabled: {problem} Falling back to shuffled KFold; "
            "the resulting estimates may contain group or class leakage.",
            RuntimeWarning,
            stacklevel=2,
        )
    n_splits = min(requested_folds, n_samples) if fallback_reason else requested_folds
    if n_splits < 2:
        raise BenchmarkSplitError("At least two folds are required for benchmark evaluation.")

    indices = np.arange(n_samples)
    folds: list[FoldSpec] = []
    for repeat in range(repeats):
        seed = random_seed + repeat
        try:
            splitter = _splitter(
                n_splits=n_splits,
                y=y,
                groups=groups,
                task=task,
                split_strategy=split_strategy,
                random_seed=seed,
            )
            if fallback_reason is not None:
                splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split_iter = splitter.split(indices, y, groups)
            local_splits = list(split_iter)
        except ValueError as exc:
            if not allow_unsafe_split_fallback:
                raise BenchmarkSplitError(
                    f"Unable to construct requested {split_strategy!r} split: {exc}"
                ) from exc
            fallback_reason = (
                fallback_reason
                or f"Unable to construct requested {split_strategy!r} split: {exc}"
            )
            warnings.warn(
                f"UNSAFE split fallback enabled: {fallback_reason} Falling back to shuffled "
                "KFold; the resulting estimates may contain group or class leakage.",
                RuntimeWarning,
                stacklevel=2,
            )
            fallback = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            local_splits = list(fallback.split(indices))
        for fold_number, (train_idx, test_idx) in enumerate(local_splits):
            if len(train_idx) == 0 or len(test_idx) == 0:
                raise BenchmarkSplitError("A generated fold contains an empty train or test set.")
            if fallback_reason is None and split_strategy in {"group", "stratified_group"}:
                train_groups = _values_for_indices(groups, np.asarray(train_idx, dtype=int))
                test_groups = _values_for_indices(groups, np.asarray(test_idx, dtype=int))
                if train_groups & test_groups:
                    raise BenchmarkSplitError(
                        "The grouped splitter produced overlapping train/test groups."
                    )
            folds.append(
                FoldSpec(
                    repeat=repeat,
                    fold=fold_number,
                    train_indices=np.asarray(train_idx, dtype=int),
                    test_indices=np.asarray(test_idx, dtype=int),
                    unsafe_fallback=fallback_reason is not None,
                    fallback_reason=fallback_reason,
                )
            )
    return folds


def _split_request_problem(
    *,
    n_samples: int,
    y: np.ndarray | None,
    groups: np.ndarray | None,
    task: str,
    requested_folds: int,
    split_strategy: str,
) -> str | None:
    if requested_folds < 2:
        return "At least two folds are required for benchmark evaluation."
    if requested_folds > n_samples:
        return (
            f"Requested {requested_folds} folds for only {n_samples} samples; "
            "fold count is never reduced silently."
        )
    grouped = split_strategy in {"group", "stratified_group"}
    if grouped and groups is None:
        return (
            f"Split strategy {split_strategy!r} requires dataset groups, but no group column "
            "was configured or loaded."
        )
    if grouped and groups is not None:
        n_groups = len(np.unique(groups))
        if requested_folds > n_groups:
            return (
                f"Requested {requested_folds} grouped folds but only {n_groups} unique groups "
                "are available; fold count is never reduced silently."
            )
    if _is_classification(task) and y is not None and split_strategy in {
        "stratified",
        "stratified_group",
    }:
        _, counts = np.unique(y, return_counts=True)
        if counts.size < 2:
            return "Stratified classification requires at least two target classes."
        if split_strategy == "stratified" and int(counts.min()) < requested_folds:
            return (
                f"Requested {requested_folds} stratified folds but the smallest class has "
                f"only {int(counts.min())} samples."
            )
        if split_strategy == "stratified_group" and groups is not None:
            for label in np.unique(y):
                label_group_count = len(np.unique(groups[y == label]))
                if label_group_count < 2:
                    return (
                        f"Class {label!r} occurs in only {label_group_count} unique group(s); "
                        "a leakage-safe stratified grouped split is impossible."
                    )
    return None


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


def _max_train_test_positional_identity(
    dataset: BenchmarkDataset,
    fold: FoldSpec,
) -> tuple[float, str]:
    best_identity = 0.0
    best_pair = "NA"
    for train_index in fold.train_indices:
        train_id = dataset.sequence_ids[int(train_index)]
        train_sequence = dataset.sequences[train_id]
        for test_index in fold.test_indices:
            test_id = dataset.sequence_ids[int(test_index)]
            test_sequence = dataset.sequences[test_id]
            identity = positional_identity(train_sequence, test_sequence)
            if identity > best_identity:
                best_identity = identity
                best_pair = f"{train_id}|{test_id}"
    return best_identity, best_pair


def positional_identity(left: str, right: str) -> float:
    """Return ungapped, position-by-position identity normalized by longer length.

    This is a lightweight leakage audit, not an alignment or homology measure.
    """
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    matches = sum(1 for index in range(length) if left[index] == right[index])
    return matches / max(len(left), len(right))
