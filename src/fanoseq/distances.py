"""Distance matrices over FanoSeq sequence fingerprints."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

DistanceMetric = Literal["euclidean", "cosine", "correlation", "manhattan"]


def build_distance_matrix(
    fingerprints: pd.DataFrame,
    id_column: str = "sequence_id",
    metric: DistanceMetric = "cosine",
    standardize: bool = True,
) -> pd.DataFrame:
    """Build a square pairwise distance matrix from numeric fingerprint columns."""
    if id_column not in fingerprints.columns:
        raise ValueError(f"Fingerprint table must contain {id_column!r}.")
    feature_columns = [
        column
        for column in fingerprints.select_dtypes(include=["number"]).columns
        if column != id_column
    ]
    if not feature_columns:
        raise ValueError("Fingerprint table has no numeric feature columns.")
    values = fingerprints[feature_columns].fillna(0.0).to_numpy(dtype=float)
    if standardize and values.shape[0] > 1:
        values = _standardize(values)
    distances = _pairwise_distances(values, metric)
    ids = fingerprints[id_column].astype(str).tolist()
    return pd.DataFrame(distances, index=ids, columns=ids)


def build_neighbor_table(
    fingerprints: pd.DataFrame,
    id_column: str = "sequence_id",
    metric: DistanceMetric = "cosine",
    k: int = 5,
    standardize: bool = True,
) -> pd.DataFrame:
    """Return a long nearest-neighbor table from fingerprints."""
    if k <= 0:
        raise ValueError("k must be > 0.")
    matrix = build_distance_matrix(
        fingerprints,
        id_column=id_column,
        metric=metric,
        standardize=standardize,
    )
    rows: list[dict[str, object]] = []
    for sequence_id in matrix.index:
        ordered = matrix.loc[sequence_id].sort_values()
        ordered = ordered[ordered.index != sequence_id].head(k)
        for rank, (neighbor_id, distance) in enumerate(ordered.items(), start=1):
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "neighbor_id": neighbor_id,
                    "rank": rank,
                    "distance": float(distance),
                    "metric": metric,
                }
            )
    return pd.DataFrame(rows)


def _standardize(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0.0] = 1.0
    return (values - means) / stds


def _pairwise_distances(values: np.ndarray, metric: DistanceMetric) -> np.ndarray:
    if metric == "euclidean":
        diff = values[:, None, :] - values[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2))
    if metric == "manhattan":
        return np.sum(np.abs(values[:, None, :] - values[None, :, :]), axis=2)
    if metric == "cosine":
        return _cosine_distances(values)
    if metric == "correlation":
        centered = values - values.mean(axis=1, keepdims=True)
        return _pairwise_distances(centered, "cosine")
    raise ValueError("metric must be one of: euclidean, cosine, correlation, manhattan.")


def _cosine_distances(values: np.ndarray) -> np.ndarray:
    n_rows = values.shape[0]
    distances = np.zeros((n_rows, n_rows), dtype=float)
    norms = np.sqrt(np.sum(values * values, axis=1))
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            denom = norms[i] * norms[j]
            if denom == 0.0:
                distance = 1.0
            else:
                similarity = float(np.sum(values[i] * values[j]) / denom)
                distance = 1.0 - similarity
            distances[i, j] = distance
            distances[j, i] = distance
    return distances
