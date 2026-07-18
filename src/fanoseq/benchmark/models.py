"""Compact model registry for benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR


@dataclass(frozen=True)
class ModelSpec:
    """A benchmark model and its compact hyperparameter grid."""

    name: str
    estimator: object
    param_grid: dict[str, list[Any]]


class TrainingFoldFeatureFilter(TransformerMixin, BaseEstimator):
    """Remove degenerate columns using training-fold data only.

    Constant, near-zero-variance, exact duplicate, and perfectly correlated columns are
    removed during ``fit``. Keeping this transformer inside the estimator pipeline avoids
    learning feature-quality decisions from validation or test rows.
    """

    def __init__(self, variance_threshold: float = 1e-12, correlation_threshold: float = 1.0):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold

    def fit(self, x: Any, y: Any = None) -> "TrainingFoldFeatureFilter":
        values = np.asarray(x, dtype=float)
        if values.ndim != 2:
            raise ValueError("TrainingFoldFeatureFilter expects a two-dimensional matrix.")
        reasons: dict[int, str] = {}
        variances = np.var(values, axis=0)
        for index, variance in enumerate(variances):
            if variance == 0.0:
                reasons[index] = "constant"
            elif variance <= self.variance_threshold:
                reasons[index] = "near_zero_variance"

        candidates = [index for index in range(values.shape[1]) if index not in reasons]
        retained: list[int] = []
        raw_signatures: dict[bytes, int] = {}
        correlation_signatures: dict[bytes, int] = {}
        for index in candidates:
            column = np.ascontiguousarray(values[:, index])
            raw_signature = column.tobytes()
            duplicate = raw_signatures.get(raw_signature)
            if duplicate is not None:
                reasons[index] = f"exact_duplicate_of_{duplicate}"
                continue
            centered = column - float(np.mean(column))
            normalized = centered / float(np.linalg.norm(centered))
            nonzero = np.flatnonzero(np.abs(normalized) > 1e-14)
            if nonzero.size and normalized[int(nonzero[0])] < 0:
                normalized = -normalized
            normalized = np.round(normalized, decimals=12)
            normalized[np.abs(normalized) < 1e-14] = 0.0
            correlation_signature = normalized.tobytes()
            correlated = correlation_signatures.get(correlation_signature)
            if correlated is None and self.correlation_threshold < 1.0 and retained:
                retained_values = values[:, retained]
                correlations = np.abs(
                    np.asarray(normalized) @ _normalized_columns(retained_values)
                )
                matches = np.flatnonzero(correlations >= self.correlation_threshold)
                if matches.size:
                    correlated = retained[int(matches[0])]
            if correlated is not None:
                reasons[index] = f"perfectly_correlated_with_{correlated}"
                continue
            retained.append(index)
            raw_signatures[raw_signature] = index
            correlation_signatures[correlation_signature] = index

        if not retained and values.shape[1] > 0:
            retained = [0]
            reasons.pop(0, None)
        self.n_features_in_ = values.shape[1]
        self.keep_indices_ = np.asarray(retained, dtype=int)
        self.removed_features_ = reasons
        return self

    def transform(self, x: Any) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        return values[:, self.keep_indices_]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            names = np.asarray([f"x{index}" for index in range(self.n_features_in_)], dtype=object)
        else:
            names = np.asarray(input_features, dtype=object)
        return names[self.keep_indices_]


def _normalized_columns(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values, axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0, keepdims=True)
    return centered / np.where(norms == 0.0, 1.0, norms)


def default_model_specs(
    task: str,
    *,
    random_seed: int,
    requested: tuple[str, ...] = (),
) -> list[ModelSpec]:
    """Return default compact model set for a benchmark task."""
    if task in {"classification", "binary_classification", "multiclass_classification"}:
        specs = _classification_specs(random_seed)
    elif task == "regression":
        specs = _regression_specs(random_seed)
    elif task == "clustering":
        return []
    else:
        raise ValueError(f"Unsupported task: {task}")
    if requested:
        requested_set = set(requested)
        unknown = requested_set - {spec.name for spec in specs}
        if unknown:
            raise ValueError(f"Unsupported model(s): {', '.join(sorted(unknown))}.")
        specs = [spec for spec in specs if spec.name in requested_set]
    return specs


def make_pipeline(spec: ModelSpec, *, feature_selection: bool = False) -> Pipeline:
    """Build a leakage-safe estimator pipeline fitted only inside CV folds."""
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("feature_quality", TrainingFoldFeatureFilter()),
        ("scaler", StandardScaler()),
    ]
    if feature_selection:
        from sklearn.feature_selection import VarianceThreshold

        steps.append(("feature_selection", VarianceThreshold()))
    steps.append(("model", spec.estimator))
    return Pipeline(steps)


def scoring_name(metric: str, task: str) -> str:
    """Map benchmark metric names to scikit-learn scoring names."""
    normalized = metric.lower()
    if task in {"classification", "binary_classification", "multiclass_classification"}:
        mapping = {
            "balanced_accuracy": "balanced_accuracy",
            "macro_f1": "f1_macro",
            "f1_macro": "f1_macro",
            "roc_auc": "roc_auc_ovr",
            "accuracy": "accuracy",
        }
        return mapping.get(normalized, "balanced_accuracy")
    if task == "regression":
        mapping = {
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
        }
        return mapping.get(normalized, "r2")
    return "adjusted_rand_score"


def higher_is_better(metric: str) -> bool:
    """Return whether larger values are better for a metric."""
    return metric.lower() not in {"rmse", "mae", "mean_absolute_error"}


def adapt_param_grid(spec: ModelSpec, *, n_train: int) -> dict[str, list[Any]]:
    """Drop KNN neighbor settings that exceed the available training rows."""
    grid = {key: list(values) for key, values in spec.param_grid.items()}
    if spec.name == "knn" and "model__n_neighbors" in grid:
        grid["model__n_neighbors"] = [
            value for value in grid["model__n_neighbors"] if int(value) <= n_train
        ] or [1]
    return grid


def _classification_specs(random_seed: int) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="logistic_regression",
            estimator=OneVsRestClassifier(
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_seed,
                )
            ),
            param_grid={"model__estimator__C": [0.1, 1.0, 10.0]},
        ),
        ModelSpec(
            name="linear_svm",
            estimator=LinearSVC(
                class_weight="balanced",
                max_iter=5000,
                random_state=random_seed,
            ),
            param_grid={"model__C": [0.1, 1.0, 10.0]},
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                n_estimators=80,
                class_weight="balanced",
                random_state=random_seed,
            ),
            param_grid={"model__max_depth": [None, 5]},
        ),
        ModelSpec(
            name="gradient_boosting",
            estimator=GradientBoostingClassifier(random_state=random_seed),
            param_grid={"model__learning_rate": [0.05, 0.1], "model__n_estimators": [50]},
        ),
        ModelSpec(
            name="knn",
            estimator=KNeighborsClassifier(),
            param_grid={"model__n_neighbors": [1, 3, 5]},
        ),
    ]


def _regression_specs(random_seed: int) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="ridge_regression",
            estimator=Ridge(random_state=random_seed),
            param_grid={"model__alpha": [0.1, 1.0, 10.0]},
        ),
        ModelSpec(
            name="linear_svm",
            estimator=LinearSVR(max_iter=5000, random_state=random_seed),
            param_grid={"model__C": [0.1, 1.0, 10.0]},
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestRegressor(n_estimators=80, random_state=random_seed),
            param_grid={"model__max_depth": [None, 5]},
        ),
        ModelSpec(
            name="gradient_boosting",
            estimator=GradientBoostingRegressor(random_state=random_seed),
            param_grid={"model__learning_rate": [0.05, 0.1], "model__n_estimators": [50]},
        ),
        ModelSpec(
            name="knn",
            estimator=KNeighborsRegressor(),
            param_grid={"model__n_neighbors": [1, 3, 5]},
        ),
    ]
