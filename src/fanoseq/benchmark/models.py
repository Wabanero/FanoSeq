"""Compact model registry for benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR


@dataclass(frozen=True)
class ModelSpec:
    """A benchmark model and its compact hyperparameter grid."""

    name: str
    estimator: object
    param_grid: dict[str, list[Any]]


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
            estimator=LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=random_seed,
            ),
            param_grid={"model__C": [0.1, 1.0, 10.0]},
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
