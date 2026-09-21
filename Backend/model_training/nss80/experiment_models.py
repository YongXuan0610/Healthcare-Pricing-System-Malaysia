"""Leakage-safe NSS 80 model screening and grouped-CV tuning.

This module never reads the final test target and never overwrites deployed
artifacts. It writes development-only experiment results beneath
``Backend/reports/nss80/experiments_v2`` for an explicit selection decision.

Run from ``Backend``::

    .\.venv\Scripts\python.exe -m model_training.nss80.experiment_models
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    load_and_prepare_dataset,
)
from .evaluate import regression_metrics


RANDOM_STATE = 42
BACKEND_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = BACKEND_ROOT / "datasets" / "nss80" / "raw"
OUTPUT_DIR = BACKEND_ROOT / "reports" / "nss80" / "experiments_v2"

ENGINEERED_NUMERIC_FEATURES = [*NUMERIC_FEATURES, "service_count"]
ENGINEERED_CATEGORICAL_FEATURES = [
    *CATEGORICAL_FEATURES,
    "age_group",
    "length_of_stay_category",
    "institution_ward",
    "surgery_length_of_stay",
    "treatment_institution",
]
ENGINEERED_FEATURES = ENGINEERED_NUMERIC_FEATURES + ENGINEERED_CATEGORICAL_FEATURES


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create pre-specified, interpretable features from prediction-time inputs."""

    output = frame[FEATURES].copy()
    output["age_group"] = pd.cut(
        output["age_years"],
        bins=[-1, 4, 17, 44, 59, 74, np.inf],
        labels=["0-4", "5-17", "18-44", "45-59", "60-74", "75+"],
    ).astype(str)
    output["length_of_stay_category"] = pd.cut(
        output["length_of_stay_days"],
        bins=[0, 1, 3, 7, 14, 30, np.inf],
        labels=["1", "2-3", "4-7", "8-14", "15-30", "31+"],
    ).astype(str)
    output["institution_ward"] = (
        output["medical_institution"].astype(str)
        + "__"
        + output["ward_type"].astype(str)
    )
    surgery_group = np.where(
        output["surgery"].eq("not_received"), "no_surgery", "surgery"
    )
    output["surgery_length_of_stay"] = (
        pd.Series(surgery_group, index=output.index).astype(str)
        + "__"
        + output["length_of_stay_category"].astype(str)
    )
    output["treatment_institution"] = (
        output["treatment_system"].astype(str)
        + "__"
        + output["medical_institution"].astype(str)
    )
    service_columns = ["surgery", "medicine", "imaging", "other_diagnostics"]
    output["service_count"] = (
        output[service_columns].ne("not_received").sum(axis=1).astype(float)
    )
    return output[ENGINEERED_FEATURES]


def _normalized_weights(series: pd.Series) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    return values / float(values.mean())


def _preprocessor(*, engineered: bool) -> ColumnTransformer:
    numeric = ENGINEERED_NUMERIC_FEATURES if engineered else NUMERIC_FEATURES
    categorical = (
        ENGINEERED_CATEGORICAL_FEATURES if engineered else CATEGORICAL_FEATURES
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        verbose_feature_names_out=False,
    )


def _prepare_X(frame: pd.DataFrame, engineered: bool) -> pd.DataFrame:
    return engineer_features(frame) if engineered else frame[FEATURES].copy()


@dataclass(frozen=True)
class ModelSpec:
    name: str
    target_transform: str
    engineered: bool
    family: str
    params: dict[str, Any]


def _fit_model(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
) -> Any:
    X_model = _prepare_X(X, spec.engineered)
    y_model = np.log1p(y) if spec.target_transform == "log1p" else y

    if spec.family == "dummy":
        estimator = DummyRegressor(strategy="median")
        estimator.fit(X_model, y_model, sample_weight=weights)
        return estimator

    if spec.family == "hist_gradient_boosting":
        categorical = (
            ENGINEERED_CATEGORICAL_FEATURES
            if spec.engineered
            else CATEGORICAL_FEATURES
        )
        numeric = ENGINEERED_NUMERIC_FEATURES if spec.engineered else NUMERIC_FEATURES
        estimator = Pipeline(
            [
                ("preprocessor", _preprocessor(engineered=spec.engineered)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        categorical_features=[False] * len(numeric)
                        + [True] * len(categorical),
                        early_stopping=True,
                        validation_fraction=0.10,
                        n_iter_no_change=25,
                        random_state=RANDOM_STATE,
                        **spec.params,
                    ),
                ),
            ]
        )
        estimator.fit(X_model, y_model, model__sample_weight=weights)
        return estimator

    if spec.family == "random_forest":
        estimator = Pipeline(
            [
                ("preprocessor", _preprocessor(engineered=spec.engineered)),
                (
                    "model",
                    RandomForestRegressor(
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        **spec.params,
                    ),
                ),
            ]
        )
        estimator.fit(X_model, y_model, model__sample_weight=weights)
        return estimator

    if spec.family == "gradient_boosting":
        estimator = Pipeline(
            [
                ("preprocessor", _preprocessor(engineered=spec.engineered)),
                (
                    "model",
                    GradientBoostingRegressor(
                        random_state=RANDOM_STATE,
                        **spec.params,
                    ),
                ),
            ]
        )
        estimator.fit(X_model, y_model, model__sample_weight=weights)
        return estimator

    if spec.family == "catboost":
        categorical = (
            ENGINEERED_CATEGORICAL_FEATURES
            if spec.engineered
            else CATEGORICAL_FEATURES
        )
        estimator = CatBoostRegressor(
            random_seed=RANDOM_STATE,
            allow_writing_files=False,
            verbose=False,
            thread_count=-1,
            cat_features=categorical,
            **spec.params,
        )
        estimator.fit(X_model, y_model, sample_weight=weights)
        return estimator

    raise ValueError(f"Unknown model family: {spec.family}")


def _predict(spec: ModelSpec, estimator: Any, X: pd.DataFrame) -> np.ndarray:
    prediction = np.asarray(
        estimator.predict(_prepare_X(X, spec.engineered)), dtype=float
    )
    if spec.target_transform == "log1p":
        prediction = np.expm1(prediction)
    return np.maximum(0.0, prediction)


def _evaluate_spec(
    spec: ModelSpec,
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> tuple[dict[str, Any], Any]:
    started = time.perf_counter()
    estimator = _fit_model(
        spec,
        train[FEATURES],
        train[TARGET],
        _normalized_weights(train["survey_multiplier"]),
    )
    prediction = _predict(spec, estimator, valid[FEATURES])
    metrics = regression_metrics(valid[TARGET].to_numpy(), prediction)
    weighted = regression_metrics(
        valid[TARGET].to_numpy(),
        prediction,
        sample_weight=_normalized_weights(valid["survey_multiplier"]),
    )
    row: dict[str, Any] = {
        "model": spec.name,
        "family": spec.family,
        "target_transform": spec.target_transform,
        "feature_set": "engineered" if spec.engineered else "original",
        "selection_data": "FSU-grouped screening validation",
        **metrics,
        **{f"weighted_{key}": value for key, value in weighted.items()},
        "fit_seconds": time.perf_counter() - started,
        "parameters": json.dumps(spec.params, sort_keys=True),
    }
    return row, estimator


def _cross_validate(
    spec: ModelSpec,
    frame: pd.DataFrame,
    folds: int = 3,
) -> dict[str, Any]:
    fold_rows: list[dict[str, float]] = []
    started = time.perf_counter()
    splitter = GroupKFold(n_splits=folds)
    for fold, (train_index, valid_index) in enumerate(
        splitter.split(frame[FEATURES], frame[TARGET], frame["fsu_id"]), start=1
    ):
        train = frame.iloc[train_index]
        valid = frame.iloc[valid_index]
        estimator = _fit_model(
            spec,
            train[FEATURES],
            train[TARGET],
            _normalized_weights(train["survey_multiplier"]),
        )
        prediction = _predict(spec, estimator, valid[FEATURES])
        metrics = regression_metrics(valid[TARGET].to_numpy(), prediction)
        weighted = regression_metrics(
            valid[TARGET].to_numpy(),
            prediction,
            sample_weight=_normalized_weights(valid["survey_multiplier"]),
        )
        fold_rows.append(
            {**metrics, **{f"weighted_{key}": value for key, value in weighted.items()}}
        )
        print(
            f"{spec.name} CV fold {fold}: MAE={metrics['mae']:.2f}, "
            f"RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}",
            flush=True,
        )

    row: dict[str, Any] = {
        "model": spec.name,
        "family": spec.family,
        "target_transform": spec.target_transform,
        "feature_set": "engineered" if spec.engineered else "original",
        "parameters": json.dumps(spec.params, sort_keys=True),
        "cv_folds": folds,
        "fit_seconds": time.perf_counter() - started,
    }
    for metric in fold_rows[0]:
        values = [fold[metric] for fold in fold_rows]
        row[f"cv_{metric}_mean"] = float(np.mean(values))
        row[f"cv_{metric}_std"] = float(np.std(values, ddof=1))
    return row


def _screening_specs() -> list[ModelSpec]:
    baseline_hgb = {
        "loss": "squared_error",
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
    }
    return [
        ModelSpec("dummy_median", "identity", False, "dummy", {}),
        ModelSpec("hist_gradient_boosting_original", "identity", False, "hist_gradient_boosting", baseline_hgb),
        ModelSpec("hist_gradient_boosting_log1p_baseline", "log1p", False, "hist_gradient_boosting", baseline_hgb),
        ModelSpec("hist_gradient_boosting_absolute", "identity", False, "hist_gradient_boosting", {**baseline_hgb, "loss": "absolute_error"}),
        ModelSpec("hist_gradient_boosting_log1p_engineered", "log1p", True, "hist_gradient_boosting", baseline_hgb),
        ModelSpec(
            "random_forest_original",
            "identity",
            False,
            "random_forest",
            {
                "n_estimators": 180,
                "max_depth": 22,
                "min_samples_leaf": 8,
                "max_features": 0.8,
            },
        ),
        ModelSpec(
            "gradient_boosting_original",
            "identity",
            False,
            "gradient_boosting",
            {
                "loss": "huber",
                "n_estimators": 140,
                "learning_rate": 0.05,
                "max_depth": 3,
                "min_samples_leaf": 25,
                "subsample": 0.8,
            },
        ),
        ModelSpec(
            "catboost_original",
            "identity",
            False,
            "catboost",
            {
                "loss_function": "RMSE",
                "iterations": 180,
                "learning_rate": 0.08,
                "depth": 7,
                "l2_leaf_reg": 5.0,
            },
        ),
        ModelSpec(
            "catboost_log1p",
            "log1p",
            False,
            "catboost",
            {
                "loss_function": "RMSE",
                "iterations": 220,
                "learning_rate": 0.07,
                "depth": 8,
                "l2_leaf_reg": 5.0,
            },
        ),
    ]


def _tuning_specs() -> list[ModelSpec]:
    hgb_grid = [
        (0.05, 400, 31, 20, 1.0),
        (0.05, 500, 63, 30, 2.0),
        (0.08, 400, 31, 40, 2.0),
        (0.08, 500, 63, 40, 5.0),
        (0.10, 350, 63, 20, 5.0),
    ]
    catboost_grid = [
        (180, 0.08, 7, 5.0),
        (250, 0.06, 8, 8.0),
        (300, 0.05, 8, 12.0),
    ]
    specs: list[ModelSpec] = []
    for index, (rate, iterations, leaves, minimum, regularization) in enumerate(
        hgb_grid, start=1
    ):
        specs.append(
            ModelSpec(
                f"tuned_hgb_log1p_engineered_{index}",
                "log1p",
                True,
                "hist_gradient_boosting",
                {
                    "loss": "squared_error",
                    "learning_rate": rate,
                    "max_iter": iterations,
                    "max_leaf_nodes": leaves,
                    "min_samples_leaf": minimum,
                    "l2_regularization": regularization,
                },
            )
        )
    for index, (iterations, rate, depth, regularization) in enumerate(
        catboost_grid, start=1
    ):
        specs.append(
            ModelSpec(
                f"tuned_catboost_log1p_{index}",
                "log1p",
                False,
                "catboost",
                {
                    "loss_function": "RMSE",
                    "iterations": iterations,
                    "learning_rate": rate,
                    "depth": depth,
                    "l2_leaf_reg": regularization,
                },
            )
        )
    return specs


def _original_feature_hgb_tuning_specs() -> list[ModelSpec]:
    grid = [
        (0.08, 300, 31, 30, 1.0),
        (0.05, 500, 63, 30, 2.0),
        (0.08, 500, 63, 40, 5.0),
        (0.10, 400, 63, 20, 5.0),
        (0.05, 600, 127, 40, 10.0),
    ]
    return [
        ModelSpec(
            f"tuned_hgb_log1p_original_{index}",
            "log1p",
            False,
            "hist_gradient_boosting",
            {
                "loss": "squared_error",
                "learning_rate": rate,
                "max_iter": iterations,
                "max_leaf_nodes": leaves,
                "min_samples_leaf": minimum,
                "l2_regularization": regularization,
            },
        )
        for index, (rate, iterations, leaves, minimum, regularization) in enumerate(
            grid, start=1
        )
    ]


def _model_selection_frame() -> pd.DataFrame:
    frame = load_and_prepare_dataset(RAW_DIR).modeling.reset_index(drop=True)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    development_index, _ = next(
        outer.split(frame[FEATURES], frame[TARGET], frame["fsu_id"])
    )
    development = frame.iloc[development_index].reset_index(drop=True)
    calibration_split = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=RANDOM_STATE + 1
    )
    model_selection_position, _ = next(
        calibration_split.split(
            development[FEATURES], development[TARGET], development["fsu_id"]
        )
    )
    return development.iloc[model_selection_position].reset_index(drop=True)


def run_original_feature_hgb_tuning() -> pd.DataFrame:
    """Tune the deployable original feature set on the locked grouped folds."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_selection = _model_selection_frame()
    rows: list[dict[str, Any]] = []
    for spec in _original_feature_hgb_tuning_specs():
        print(f"Tuning evaluation {spec.name}...", flush=True)
        rows.append(_cross_validate(spec, model_selection, folds=3))
        pd.DataFrame(rows).sort_values("cv_mae_mean", ignore_index=True).to_csv(
            OUTPUT_DIR / "tuning_results_original_features.partial.csv", index=False
        )
    results = pd.DataFrame(rows).sort_values("cv_mae_mean", ignore_index=True)
    results.to_csv(OUTPUT_DIR / "tuning_results_original_features.csv", index=False)
    print("\nOriginal-feature HGB tuning results:\n", results.to_string(index=False))
    return results


def run() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = load_and_prepare_dataset(RAW_DIR)
    frame = audit.modeling.reset_index(drop=True)

    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    development_index, test_index = next(
        outer.split(frame[FEATURES], frame[TARGET], frame["fsu_id"])
    )
    development = frame.iloc[development_index].reset_index(drop=True)
    untouched_test = frame.iloc[test_index].reset_index(drop=True)

    calibration_split = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=RANDOM_STATE + 1
    )
    model_selection_position, calibration_position = next(
        calibration_split.split(
            development[FEATURES], development[TARGET], development["fsu_id"]
        )
    )
    model_selection = development.iloc[model_selection_position].reset_index(drop=True)
    interval_calibration = development.iloc[calibration_position].reset_index(drop=True)

    screening_split = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=RANDOM_STATE + 2
    )
    screening_train_position, screening_valid_position = next(
        screening_split.split(
            model_selection[FEATURES],
            model_selection[TARGET],
            model_selection["fsu_id"],
        )
    )
    screening_train = model_selection.iloc[screening_train_position].reset_index(drop=True)
    screening_valid = model_selection.iloc[screening_valid_position].reset_index(drop=True)

    screening_rows: list[dict[str, Any]] = []
    for spec in _screening_specs():
        print(f"Screening {spec.name}...", flush=True)
        row, _ = _evaluate_spec(spec, screening_train, screening_valid)
        screening_rows.append(row)
        pd.DataFrame(screening_rows).sort_values("mae", ignore_index=True).to_csv(
            OUTPUT_DIR / "model_screening.partial.csv", index=False
        )
        print(
            f"  MAE={row['mae']:.2f}, RMSE={row['rmse']:.2f}, "
            f"R2={row['r2']:.4f}, median AE={row['median_absolute_error']:.2f}",
            flush=True,
        )
    screening = pd.DataFrame(screening_rows).sort_values("mae", ignore_index=True)
    screening.to_csv(OUTPUT_DIR / "model_screening.csv", index=False)

    tuning_rows: list[dict[str, Any]] = []
    for spec in _tuning_specs():
        print(f"Tuning evaluation {spec.name}...", flush=True)
        tuning_rows.append(_cross_validate(spec, model_selection, folds=3))
        pd.DataFrame(tuning_rows).sort_values(
            "cv_mae_mean", ignore_index=True
        ).to_csv(OUTPUT_DIR / "tuning_results.partial.csv", index=False)
    tuning = pd.DataFrame(tuning_rows).sort_values("cv_mae_mean", ignore_index=True)
    tuning.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)

    target = frame[TARGET]
    target_distribution = {
        "count": int(len(target)),
        "minimum": float(target.min()),
        "maximum": float(target.max()),
        "mean": float(target.mean()),
        "median": float(target.median()),
        "standard_deviation": float(target.std(ddof=1)),
        "p25": float(target.quantile(0.25)),
        "p75": float(target.quantile(0.75)),
        "p90": float(target.quantile(0.90)),
        "p95": float(target.quantile(0.95)),
        "p99": float(target.quantile(0.99)),
        "skewness": float(target.skew()),
        "zero_rows": int(target.eq(0).sum()),
    }
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Development-only screening and tuning; final test target untouched.",
        "split": {
            "model_selection_rows": int(len(model_selection)),
            "interval_calibration_rows": int(len(interval_calibration)),
            "untouched_test_rows": int(len(untouched_test)),
            "screening_train_rows": int(len(screening_train)),
            "screening_validation_rows": int(len(screening_valid)),
        },
        "target_distribution_inr": target_distribution,
        "best_screening_model": screening.iloc[0].to_dict(),
        "best_tuned_configuration": tuning.iloc[0].to_dict(),
        "xgboost_decision": (
            "Not added: CatBoost directly supports the NSS categorical predictors; "
            "adding a second external boosting dependency would not add a distinct "
            "methodological benefit at this stage."
        ),
    }
    (OUTPUT_DIR / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nScreening results:\n", screening.to_string(index=False), flush=True)
    print("\nTuning results:\n", tuning.to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--original-hgb-only":
        run_original_feature_hgb_tuning()
    else:
        run()
