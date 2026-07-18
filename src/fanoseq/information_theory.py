"""Transparent empirical information-theory diagnostics for feature audits."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import log2, sqrt
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def empirical_entropy(values: Iterable[object]) -> float:
    """Return plug-in Shannon entropy in bits for discrete observations."""
    observations = [_hashable(value) for value in values]
    if not observations:
        return 0.0
    counts = Counter(observations)
    total = float(len(observations))
    return -sum((count / total) * log2(count / total) for count in counts.values())


def mutual_information(x: Sequence[object], y: Sequence[object]) -> float:
    """Return empirical pairwise mutual information in bits."""
    _require_same_length(x, y)
    return empirical_entropy(x) + empirical_entropy(y) - empirical_entropy(zip(x, y))


def conditional_mutual_information(
    x: Sequence[object],
    y: Sequence[object],
    z: Sequence[object],
) -> float:
    """Return I(X;Y|Z) in bits using the empirical plug-in estimator."""
    _require_same_length(x, y, z)
    return (
        empirical_entropy(zip(x, z))
        + empirical_entropy(zip(y, z))
        - empirical_entropy(z)
        - empirical_entropy(zip(x, y, z))
    )


def interaction_information(
    x: Sequence[object],
    y: Sequence[object],
    z: Sequence[object],
) -> float:
    """Return signed three-way interaction information I(X;Y)-I(X;Y|Z)."""
    return mutual_information(x, y) - conditional_mutual_information(x, y, z)


def total_correlation(*variables: Sequence[object]) -> float:
    """Return multi-information/total correlation in bits for discrete variables."""
    if len(variables) < 2:
        raise ValueError("total_correlation requires at least two variables.")
    _require_same_length(*variables)
    joint = zip(*variables)
    return sum(empirical_entropy(variable) for variable in variables) - empirical_entropy(joint)


def quantile_discretize(values: Sequence[float], n_bins: int = 5) -> np.ndarray:
    """Discretize numeric observations into deterministic empirical-quantile bins."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    series = pd.Series(values, dtype=float)
    if series.isna().any():
        raise ValueError("quantile_discretize does not accept missing values.")
    if series.nunique() <= 1:
        return np.zeros(len(series), dtype=int)
    binned = pd.qcut(
        series,
        q=min(n_bins, int(series.nunique())),
        labels=False,
        duplicates="drop",
    )
    return binned.to_numpy(dtype=int)


def information_audit_table(
    table: pd.DataFrame,
    columns: Sequence[str],
    *,
    numeric_bins: int = 5,
    include_three_way: bool = True,
) -> pd.DataFrame:
    """Build pairwise and higher-order information diagnostics.

    Numeric inputs are quantile-discretized. The output records estimator and binning so
    values are not mistaken for invariant continuous-information estimates.
    """
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"Missing information-audit columns: {', '.join(missing)}.")
    discrete = {
        column: _discrete_column(table[column], numeric_bins=numeric_bins)
        for column in columns
    }
    rows: list[dict[str, object]] = []
    for left, right in combinations(columns, 2):
        left_values = discrete[left]
        right_values = discrete[right]
        mi = mutual_information(left_values, right_values)
        denominator = sqrt(
            empirical_entropy(left_values) * empirical_entropy(right_values)
        )
        rows.append(
            {
                "order": 2,
                "variables": f"{left}|{right}",
                "measure": "mutual_information",
                "value_bits": mi,
                "normalized_value": mi / denominator if denominator else 0.0,
                "estimator": "empirical_plugin",
                "discretization": f"quantile_bins={numeric_bins}",
            }
        )
    if include_three_way:
        for first, second, third in combinations(columns, 3):
            values = (discrete[first], discrete[second], discrete[third])
            rows.extend(
                [
                    {
                        "order": 3,
                        "variables": f"{first}|{second}|{third}",
                        "measure": "interaction_information",
                        "value_bits": interaction_information(*values),
                        "normalized_value": np.nan,
                        "estimator": "empirical_plugin",
                        "discretization": f"quantile_bins={numeric_bins}",
                    },
                    {
                        "order": 3,
                        "variables": f"{first}|{second}|{third}",
                        "measure": "total_correlation",
                        "value_bits": total_correlation(*values),
                        "normalized_value": np.nan,
                        "estimator": "empirical_plugin",
                        "discretization": f"quantile_bins={numeric_bins}",
                    },
                ]
            )
    return pd.DataFrame(rows)


def _discrete_column(series: pd.Series, *, numeric_bins: int) -> list[object]:
    if series.isna().any():
        raise ValueError(f"Information audit column {series.name!r} contains missing values.")
    if pd.api.types.is_numeric_dtype(series):
        return quantile_discretize(series.astype(float).tolist(), numeric_bins).tolist()
    return series.astype(str).tolist()


def _hashable(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


def _require_same_length(*variables: Sequence[object]) -> None:
    lengths = {len(variable) for variable in variables}
    if len(lengths) > 1:
        raise ValueError("Information-theory variables must have the same length.")
