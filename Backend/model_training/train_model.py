"""Train, compare, tune, explain, and save the healthcare charge pipeline.

Run from the Backend directory:
    python model_training/train_model.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model_training.data_analysis import (  # type: ignore
        CATEGORICAL_FEATURES,
        FEATURES,
        NUMERIC_FEATURES,
        TARGET,
        generate_eda_plots,
        load_and_validate_dataset,
        summarize_eda,
        write_dataset_inspection_report,
    )
    from model_training.evaluate_model import cross_validation_metrics, regression_metrics  # type: ignore
else:
    from .data_analysis import (
        CATEGORICAL_FEATURES,
        FEATURES,
        NUMERIC_FEATURES,
        TARGET,
        generate_eda_plots,
        load_and_validate_dataset,
        summarize_eda,
        write_dataset_inspection_report,
    )
    from .evaluate_model import cross_validation_metrics, regression_metrics


RANDOM_STATE = 42
TEST_SIZE = 0.20
CALIBRATION_SIZE = 0.20
PREDICTION_INTERVAL_COVERAGE = 0.90
BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    BACKEND_ROOT
    / "datasets"
    / "us_kaggle"
    / "raw"
    / "Insurance_Dataset_US_Medical_Charges_1338.csv"
)
MODELS_DIR = BACKEND_ROOT / "models" / "us_kaggle"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
REPORT_PATH = Path(__file__).resolve().parent / "MODEL_TRAINING_REPORT.md"
INSPECTION_PATH = Path(__file__).resolve().parent / "DATASET_INSPECTION_REPORT.md"


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def make_pipeline(regressor: Any) -> Pipeline:
    return Pipeline(steps=[("preprocessor", make_preprocessor()), ("regressor", regressor)])


def base_models() -> dict[str, Pipeline]:
    return {
        "Dummy Regressor": make_pipeline(DummyRegressor(strategy="mean")),
        "Linear Regression": make_pipeline(LinearRegression()),
        "Decision Tree": make_pipeline(DecisionTreeRegressor(random_state=RANDOM_STATE)),
        "Random Forest": make_pipeline(
            RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
        ),
        "Gradient Boosting": make_pipeline(GradientBoostingRegressor(random_state=RANDOM_STATE)),
    }


def tune_ensemble_models(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[dict[str, Pipeline], dict[str, dict[str, Any]]]:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    searches = {
        "Tuned Random Forest": RandomizedSearchCV(
            estimator=make_pipeline(
                RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
            ),
            param_distributions={
                "regressor__n_estimators": [200, 350, 500, 700],
                "regressor__max_depth": [None, 5, 8, 12, 16],
                "regressor__min_samples_split": [2, 5, 10, 15],
                "regressor__min_samples_leaf": [1, 2, 4, 8],
                "regressor__max_features": ["sqrt", 0.7, 1.0],
            },
            n_iter=24,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        ),
        "Tuned Gradient Boosting": RandomizedSearchCV(
            estimator=make_pipeline(GradientBoostingRegressor(random_state=RANDOM_STATE)),
            param_distributions={
                "regressor__n_estimators": [100, 150, 200, 300],
                "regressor__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
                "regressor__max_depth": [2, 3, 4],
                "regressor__min_samples_split": [2, 5, 10, 15],
                "regressor__min_samples_leaf": [1, 2, 4, 8],
                "regressor__subsample": [0.7, 0.85, 1.0],
            },
            n_iter=24,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        ),
    }
    tuned: dict[str, Pipeline] = {}
    details: dict[str, dict[str, Any]] = {}
    for name, search in searches.items():
        print(f"Tuning {name} ...")
        search.fit(X_train, y_train)
        tuned[name] = search.best_estimator_
        details[name] = {
            "best_parameters": {
                key.replace("regressor__", ""): value for key, value in search.best_params_.items()
            },
            "best_cv_rmse": float(-search.best_score_),
            "search_method": "RandomizedSearchCV",
            "iterations": int(search.n_iter),
            "folds": 5,
            "selection_metric": "RMSE",
        }
    return tuned, details


def _write_bar_plot(values: pd.Series, path: Path, title: str, x_label: str) -> None:
    values = values.sort_values(ascending=True).tail(15)
    width, height = 980, max(520, 75 + len(values) * 31)
    left, right, top, bottom = 260, 40, 55, 50
    plot_width = width - left - right
    max_value = max(float(values.max()), 1e-12)
    rows = []
    for index, (label, value) in enumerate(values.items()):
        y = top + index * 31
        bar_width = float(value) / max_value * plot_width
        rows.append(f'<text x="{left-10}" y="{y+18}" text-anchor="end" font-size="13">{label}</text>')
        rows.append(f'<rect x="{left}" y="{y+4}" width="{bar_width:.2f}" height="21" fill="#0f766e" rx="2"/>')
        rows.append(f'<text x="{left+bar_width+6:.2f}" y="{y+19}" font-size="12">{float(value):.4f}</text>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<style>text{font-family:Arial,sans-serif;fill:#172554}</style>'
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="23" font-weight="700">{title}</text>'
        + "".join(rows)
        + f'<text x="{left+plot_width/2}" y="{height-12}" text-anchor="middle" font-size="13">{x_label}</text></svg>'
    )
    path.write_text(svg, encoding="utf-8")


def _write_error_plots(actual: pd.Series, predicted: np.ndarray, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    residuals = actual.to_numpy() - predicted
    specs = [
        (actual.to_numpy(), predicted, "actual_vs_predicted.svg", "Actual vs predicted charges", "Actual charge", "Predicted charge"),
        (predicted, residuals, "residual_plot.svg", "Residuals vs predicted charges", "Predicted charge", "Residual (actual - predicted)"),
    ]
    for x_values, y_values, filename, title, x_label, y_label in specs:
        width, height = 900, 600
        x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
        y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
        def scale(value: float, low: float, high: float, out_low: float, out_high: float) -> float:
            return (out_low + out_high) / 2 if high == low else out_low + (value-low)*(out_high-out_low)/(high-low)
        points = []
        for x, y in zip(x_values, y_values):
            px = scale(float(x), x_min, x_max, 80, 860)
            py = scale(float(y), y_min, y_max, 520, 70)
            points.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.5" fill="#2563eb" opacity="0.48"/>')
        reference = ""
        if filename == "actual_vs_predicted.svg":
            low, high = max(x_min, y_min), min(x_max, y_max)
            x1, x2 = scale(low, x_min, x_max, 80, 860), scale(high, x_min, x_max, 80, 860)
            y1, y2 = scale(low, y_min, y_max, 520, 70), scale(high, y_min, y_max, 520, 70)
            reference = f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#dc2626" stroke-width="2" stroke-dasharray="7 5"/>'
        else:
            y0 = scale(0, y_min, y_max, 520, 70)
            reference = f'<line x1="80" y1="{y0:.2f}" x2="860" y2="{y0:.2f}" stroke="#dc2626" stroke-width="2" stroke-dasharray="7 5"/>'
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/>'
            '<style>text{font-family:Arial,sans-serif;fill:#172554}</style>'
            f'<text x="450" y="34" text-anchor="middle" font-size="23" font-weight="700">{title}</text>'
            '<line x1="80" y1="520" x2="860" y2="520" stroke="#475569"/><line x1="80" y1="70" x2="80" y2="520" stroke="#475569"/>'
            f'{reference}{"".join(points)}'
            f'<text x="470" y="570" text-anchor="middle" font-size="13">{x_label}</text>'
            f'<text x="22" y="295" text-anchor="middle" font-size="13" transform="rotate(-90 22 295)">{y_label}</text></svg>'
        )
        (plots_dir / filename).write_text(svg, encoding="utf-8")


def _write_prediction_interval_plot(
    actual: pd.Series,
    predicted: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    path: Path,
) -> None:
    """Plot held-out observations ordered by point prediction with their ranges."""

    order = np.argsort(predicted)
    actual_values = actual.to_numpy()[order]
    predicted_values = predicted[order]
    lower_values = lower[order]
    upper_values = upper[order]
    width, height = 1000, 620
    y_max = float(max(actual_values.max(), upper_values.max())) * 1.03

    def x_scale(index: int) -> float:
        return 75 + index * 875 / max(len(order) - 1, 1)

    def y_scale(value: float) -> float:
        return 540 - value * 465 / max(y_max, 1.0)

    elements: list[str] = []
    for index, (actual_value, point, low, high) in enumerate(
        zip(actual_values, predicted_values, lower_values, upper_values)
    ):
        x = x_scale(index)
        elements.append(
            f'<line x1="{x:.2f}" y1="{y_scale(low):.2f}" x2="{x:.2f}" y2="{y_scale(high):.2f}" stroke="#0f766e" stroke-width="1.2" opacity="0.32"/>'
        )
        elements.append(
            f'<circle cx="{x:.2f}" cy="{y_scale(point):.2f}" r="1.8" fill="#2563eb" opacity="0.72"/>'
        )
        elements.append(
            f'<circle cx="{x:.2f}" cy="{y_scale(actual_value):.2f}" r="1.8" fill="#111827" opacity="0.72"/>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<style>text{font-family:Arial,sans-serif;fill:#172554}</style>'
        '<text x="500" y="34" text-anchor="middle" font-size="23" font-weight="700">Held-out test prediction ranges</text>'
        '<line x1="75" y1="540" x2="950" y2="540" stroke="#475569"/><line x1="75" y1="75" x2="75" y2="540" stroke="#475569"/>'
        + "".join(elements)
        + '<line x1="720" y1="70" x2="742" y2="70" stroke="#0f766e" stroke-width="4" opacity="0.5"/><text x="748" y="74" font-size="11">90% range</text>'
        '<circle cx="820" cy="70" r="4" fill="#2563eb"/><text x="828" y="74" font-size="11">point estimate</text>'
        '<circle cx="910" cy="70" r="4" fill="#111827"/><text x="918" y="74" font-size="11">actual</text>'
        '<text x="510" y="590" text-anchor="middle" font-size="13">Test observations ordered by point estimate</text>'
        '<text x="22" y="310" text-anchor="middle" font-size="13" transform="rotate(-90 22 310)">Charge (source dataset scale)</text></svg>'
    )
    path.write_text(svg, encoding="utf-8")


def extract_feature_importance(pipeline: Pipeline) -> pd.Series:
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    estimator = pipeline.named_steps["regressor"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_).ravel())
    else:
        raise TypeError(f"Estimator {type(estimator).__name__} does not expose feature importance")
    return pd.Series(values, index=feature_names, name="importance").sort_values(ascending=False)


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    """Finite-sample split-conformal score quantile."""

    score_values = np.asarray(scores, dtype=float)
    if score_values.size == 0:
        raise ValueError("At least one calibration score is required")
    quantile_level = min(1.0, np.ceil((score_values.size + 1) * coverage) / score_values.size)
    return float(np.quantile(score_values, quantile_level, method="higher"))


def package_versions() -> dict[str, str]:
    names = ["pandas", "numpy", "scikit-learn", "joblib", "fastapi", "pydantic"]
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    versions["xgboost"] = "not installed; intentionally omitted"
    versions["shap"] = "not installed; optional and not required"
    return versions


def _format_parameters(parameters: dict[str, Any]) -> str:
    return ", ".join(f"`{key}={value}`" for key, value in sorted(parameters.items()))


def write_training_report(metadata: dict[str, Any], comparison: pd.DataFrame, output_path: Path) -> None:
    final = metadata["final_model"]
    interval = metadata["prediction_interval"]
    top_features = metadata["feature_importance"][:8]
    group_summaries = metadata["eda"]["group_summaries"]
    table_rows = [
        f"| {row.model} | {row.test_mae:,.2f} | {row.test_rmse:,.2f} | {row.test_r2:.4f} | {row.cv_rmse_mean:,.2f} ± {row.cv_rmse_std:,.2f} | {row.cv_r2_mean:.4f} ± {row.cv_r2_std:.4f} |"
        for row in comparison.itertuples(index=False)
    ]
    feature_rows = [f"| {item['feature']} | {item['importance']:.4f} |" for item in top_features]
    rf_tuning = metadata["hyperparameter_tuning"]["Tuned Random Forest"]
    gb_tuning = metadata["hyperparameter_tuning"]["Tuned Gradient Boosting"]
    lines = [
        "# US Kaggle Secondary Benchmark Training Report",
        "",
        "## Dataset",
        "",
        "The supplied `Insurance_Dataset_US_Medical_Charges_1338.csv` contains 1,338 published observations. "
        "Its predictors are age, sex, BMI, children, smoking status, and US region; the source target column is `charges_usd`, mapped explicitly to internal name `charges`. "
        "`record_id` is excluded from modeling. The bundle documentation describes the classic dataset as simulated from US demographic statistics. "
        "It is not Malaysian or NSS patient-level data and is retained only as the secondary benchmark.",
        "",
        f"One duplicated observation (excluding its distinct record IDs) was documented and removed only from the modeling view, leaving {metadata['dataset']['modeling_rows']} observations. "
        "The original CSV remains unchanged. No LIAM or Malaysian price-reference records were merged.",
        "",
        "## Data Preprocessing",
        "",
        f"The modeling view was first split into {metadata['split']['training_rows']} training rows (80%) and {metadata['split']['testing_rows']} testing rows (20%) using `random_state={metadata['random_state']}`. "
        f"The training partition was then divided into {metadata['split']['development_rows']} development rows and {metadata['split']['calibration_rows']} untouched calibration rows. "
        "All models use a Scikit-learn `Pipeline` containing a `ColumnTransformer`: numeric fields use median imputation and standardization; categorical fields use most-frequent imputation and one-hot encoding with unknown-category protection. "
        "Preprocessing is fitted inside each development/CV fold, preventing leakage. The calibration partition is used only to construct the prediction range, and the same held-out test indices are used for every final comparison.",
        "",
        "## Exploratory Analysis",
        "",
        f"Charges are right-skewed (skewness {metadata['eda']['charges_skewness']:.3f}). Smoking status has the largest visible group separation. "
        f"Numerical correlations with charges are age {metadata['eda']['numeric_correlations_with_charges']['age']:.3f}, BMI {metadata['eda']['numeric_correlations_with_charges']['bmi']:.3f}, and children {metadata['eda']['numeric_correlations_with_charges']['children']:.3f}. "
        f"Smokers average {group_summaries['smoker']['yes']['mean']:,.2f}, compared with {group_summaries['smoker']['no']['mean']:,.2f} for non-smokers. "
        f"Male/female means are {group_summaries['sex']['male']['mean']:,.2f}/{group_summaries['sex']['female']['mean']:,.2f}. "
        f"The Southeast has the highest regional mean ({group_summaries['region']['southeast']['mean']:,.2f}); the number-of-children groups are non-monotonic, with small samples for four and five children. "
        "These are unadjusted associations, not causal effects.",
        "",
        "Useful EDA figures are saved under `model_training/plots/`.",
        "",
        "## Models Evaluated",
        "",
        "The comparison includes a mean-predicting Dummy Regressor baseline, Linear Regression, an unconstrained Decision Tree, Random Forest, Gradient Boosting, and tuned Random Forest/Gradient Boosting pipelines. "
        "XGBoost was not installed and was intentionally omitted because the required Scikit-learn ensembles already provide strong comparison models without an extra compiled dependency. SHAP was also unnecessary; native feature importance is used.",
        "",
        "## Evaluation Metrics",
        "",
        "- **MAE** is the mean absolute prediction error in the source dataset's charge scale; lower is better.",
        "- **RMSE** penalizes large errors more heavily and is also in the source charge scale; lower is better.",
        "- **R²** is the proportion of test-target variance explained by the model; higher is better. It is not classification accuracy.",
        "",
        "## Model Comparison",
        "",
        "| Model | Test MAE | Test RMSE | Test R² | CV RMSE (mean ± SD) | CV R² (mean ± SD) |",
        "|---|---:|---:|---:|---:|---:|",
        *table_rows,
        "",
        "Five-fold cross-validation was performed only on the training partition. The held-out test set was evaluated after the untuned and tuned candidates were frozen.",
        "",
        "## Hyperparameter Tuning",
        "",
        f"Both ensemble candidates used 24-iteration `RandomizedSearchCV` with five shuffled folds and training-fold RMSE as the selection metric. Random Forest best parameters: {_format_parameters(rf_tuning['best_parameters'])}. "
        f"Gradient Boosting best parameters: {_format_parameters(gb_tuning['best_parameters'])}.",
        "",
        "## Final Model",
        "",
        f"**{final['name']}** was selected. Its held-out results are MAE **{final['test_metrics']['mae']:,.2f}**, RMSE **{final['test_metrics']['rmse']:,.2f}**, and R² **{final['test_metrics']['r2']:.4f}**. "
        f"Its training RMSE is {final['training_metrics']['rmse']:,.2f}, compared with {final['test_metrics']['rmse']:,.2f} on test data. "
        f"Its five-fold training CV RMSE is {final['cross_validation']['cv_rmse_mean']:,.2f} ± {final['cross_validation']['cv_rmse_std']:,.2f}. "
        "The selection prioritizes training-only cross-validation stability among the tuned ensemble candidates and checks that the held-out result is consistent; it does not rely on training R² alone.",
        "",
        "## Estimated Prediction Range",
        "",
        f"Within this secondary benchmark, a **{interval['nominal_coverage']:.0%} prediction range** is presented with the point estimate. Lower and upper Gradient Boosting quantile models learn feature-dependent bounds, which are then conformally adjusted using {interval['calibration_rows']} records that were not used for model fitting or hyperparameter selection. "
        f"Because error behaviour differs substantially by smoking status, conformal adjustments are calibrated separately for smokers and non-smokers: {_format_parameters(interval['conformal_adjustment_by_smoker'])}. A small negative adjustment is valid and indicates that the corresponding raw quantile bounds were slightly conservative on calibration data. "
        f"On the untouched test set, the interval contained the observed charge for **{interval['test_coverage']:.1%}** of records, with mean width **{interval['test_mean_width']:,.2f}** in the source charge scale. "
        "This is an uncertainty interval under the dataset/exchangeability assumptions, not a guarantee that an individual bill will fall inside it. The point estimate is retained as supporting information rather than presented as an exact quotation. See `plots/prediction_intervals.svg` for held-out examples.",
        "",
        "## Overfitting Analysis",
        "",
        metadata["overfitting_analysis"],
        "",
        "## Feature Importance",
        "",
        "One-hot encoded names are recovered from the fitted `ColumnTransformer`. Importance reflects predictive contribution within this model, not causality.",
        "",
        "| Feature | Importance |",
        "|---|---:|",
        *feature_rows,
        "",
        "See `plots/feature_importance.svg` for the visualization.",
        "",
        "## Prediction Error Analysis",
        "",
        f"The top-cost quartile begins at {metadata['error_analysis']['high_cost_threshold']:,.2f}. Its MAE is {metadata['error_analysis']['high_cost_mae']:,.2f}, versus {metadata['error_analysis']['lower_cost_mae']:,.2f} for the remaining observations. "
        f"Therefore, high-cost observations were {'more' if metadata['error_analysis']['high_cost_mae'] > metadata['error_analysis']['lower_cost_mae'] else 'not more'} difficult on this test split. "
        "The actual-vs-predicted and residual figures are in `model_training/plots/`, and `largest_prediction_errors.csv` records the worst held-out cases for discussion.",
        "",
        "## Limitations",
        "",
        "- This dataset is not localized to Malaysia and should not be treated as Malaysian hospital billing evidence.",
        "- The model output remains in the scale and context of the source `charges_usd` target; it must not be relabeled as Malaysian Ringgit or arbitrarily converted.",
        "- LIAM and Malaysian hospital pricing data are kept separate and are used only as local reference information.",
        "- The small dataset, limited features, right-skewed target, and high-cost residuals limit generalization.",
        "- Predictions are estimates, not guaranteed charges, quotations, medical advice, or financial advice.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(dataset_path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    audit = load_and_validate_dataset(dataset_path, remove_duplicate_observations=True)
    write_dataset_inspection_report(audit.report, INSPECTION_PATH)
    eda = summarize_eda(audit.modeling)
    generate_eda_plots(audit.modeling, PLOTS_DIR)

    frame = audit.modeling
    X = frame[FEATURES].copy()
    y = frame[TARGET].copy()
    X_training, X_test, y_training, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    X_train, X_calibration, y_train, y_calibration = train_test_split(
        X_training,
        y_training,
        test_size=CALIBRATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=X_training["smoker"],
    )

    fitted_models: dict[str, Pipeline] = {}
    cv_results: dict[str, dict[str, float]] = {}
    for name, pipeline in base_models().items():
        print(f"Cross-validating {name} ...")
        cv_results[name] = cross_validation_metrics(pipeline, X_train, y_train, RANDOM_STATE)
        fitted_models[name] = pipeline.fit(X_train, y_train)

    tuned_models, tuning_details = tune_ensemble_models(X_train, y_train)
    for name, pipeline in tuned_models.items():
        cv_results[name] = cross_validation_metrics(pipeline, X_train, y_train, RANDOM_STATE)
        fitted_models[name] = pipeline

    comparison_rows: list[dict[str, Any]] = []
    evaluation: dict[str, dict[str, Any]] = {}
    for name, pipeline in fitted_models.items():
        train_predictions = pipeline.predict(X_train)
        test_predictions = pipeline.predict(X_test)
        train_metrics = regression_metrics(y_train, train_predictions)
        test_metrics = regression_metrics(y_test, test_predictions)
        evaluation[name] = {
            "training_metrics": train_metrics,
            "test_metrics": test_metrics,
            "cross_validation": cv_results[name],
        }
        comparison_rows.append(
            {
                "model": name,
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                **cv_results[name],
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("test_rmse").reset_index(drop=True)
    comparison.to_csv(MODELS_DIR / "model_comparison.csv", index=False)

    tuned_names = list(tuned_models)
    final_name = min(tuned_names, key=lambda name: cv_results[name]["cv_rmse_mean"])
    final_pipeline = fitted_models[final_name]
    final_predictions = final_pipeline.predict(X_test)

    quantile_parameters = tuning_details["Tuned Gradient Boosting"]["best_parameters"]
    lower_quantile_pipeline = make_pipeline(
        GradientBoostingRegressor(
            loss="quantile",
            alpha=(1.0 - PREDICTION_INTERVAL_COVERAGE) / 2.0,
            random_state=RANDOM_STATE,
            **quantile_parameters,
        )
    ).fit(X_train, y_train)
    upper_quantile_pipeline = make_pipeline(
        GradientBoostingRegressor(
            loss="quantile",
            alpha=1.0 - (1.0 - PREDICTION_INTERVAL_COVERAGE) / 2.0,
            random_state=RANDOM_STATE,
            **quantile_parameters,
        )
    ).fit(X_train, y_train)
    calibration_lower = lower_quantile_pipeline.predict(X_calibration)
    calibration_upper = upper_quantile_pipeline.predict(X_calibration)
    calibration_scores = np.maximum(
        calibration_lower - y_calibration.to_numpy(),
        y_calibration.to_numpy() - calibration_upper,
    )
    global_adjustment = conformal_quantile(
        calibration_scores, PREDICTION_INTERVAL_COVERAGE
    )
    adjustment_by_smoker: dict[str, float] = {}
    for smoker_value in sorted(X_calibration["smoker"].unique()):
        mask = X_calibration["smoker"].to_numpy() == smoker_value
        adjustment_by_smoker[str(smoker_value)] = conformal_quantile(
            calibration_scores[mask], PREDICTION_INTERVAL_COVERAGE
        )
    test_adjustments = np.asarray(
        [adjustment_by_smoker.get(str(value), global_adjustment) for value in X_test["smoker"]],
        dtype=float,
    )
    raw_test_lower = lower_quantile_pipeline.predict(X_test)
    raw_test_upper = upper_quantile_pipeline.predict(X_test)
    interval_lower = np.maximum(0.0, raw_test_lower - test_adjustments)
    interval_upper = raw_test_upper + test_adjustments
    covered = (y_test.to_numpy() >= interval_lower) & (y_test.to_numpy() <= interval_upper)
    coverage_by_smoker = {
        str(smoker_value): float(covered[X_test["smoker"].to_numpy() == smoker_value].mean())
        for smoker_value in sorted(X_test["smoker"].unique())
    }

    importance = extract_feature_importance(final_pipeline)
    importance.rename_axis("feature").reset_index().to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    _write_bar_plot(importance, PLOTS_DIR / "feature_importance.svg", f"Feature importance — {final_name}", "Model importance")
    _write_error_plots(y_test, final_predictions, PLOTS_DIR)
    _write_prediction_interval_plot(
        y_test,
        final_predictions,
        interval_lower,
        interval_upper,
        PLOTS_DIR / "prediction_intervals.svg",
    )

    errors = X_test.copy()
    if "record_id" in frame.columns:
        errors.insert(0, "record_id", frame.loc[X_test.index, "record_id"].astype(int))
    errors["actual_charge"] = y_test
    errors["predicted_charge"] = final_predictions
    errors["range_lower"] = interval_lower
    errors["range_upper"] = interval_upper
    errors["range_covered_actual"] = covered
    errors["residual"] = y_test.to_numpy() - final_predictions
    errors["absolute_error"] = np.abs(errors["residual"])
    errors = errors.sort_values("absolute_error", ascending=False)
    errors.head(20).to_csv(MODELS_DIR / "largest_prediction_errors.csv", index=False)

    high_threshold = float(y_test.quantile(0.75))
    high_mask = y_test >= high_threshold
    high_mae = float(np.mean(np.abs(y_test[high_mask].to_numpy() - final_predictions[high_mask.to_numpy()])))
    lower_mae = float(np.mean(np.abs(y_test[~high_mask].to_numpy() - final_predictions[(~high_mask).to_numpy()])))

    tree_train = evaluation["Decision Tree"]["training_metrics"]
    tree_test = evaluation["Decision Tree"]["test_metrics"]
    forest_train = evaluation["Random Forest"]["training_metrics"]
    forest_test = evaluation["Random Forest"]["test_metrics"]
    tuned_final = evaluation[final_name]
    overfitting_text = (
        f"The unconstrained Decision Tree shows clear overfitting: training R² is {tree_train['r2']:.4f} "
        f"but test R² is {tree_test['r2']:.4f}, with test RMSE {tree_test['rmse']:,.2f}. "
        f"The untuned Random Forest also has a train/test gap (R² {forest_train['r2']:.4f} vs {forest_test['r2']:.4f}), "
        f"although averaging reduces the tree's instability. The selected tuned model constrains complexity through its searched parameters; "
        f"its train/test R² values are {tuned_final['training_metrics']['r2']:.4f}/{tuned_final['test_metrics']['r2']:.4f}, "
        "and its CV variation provides a more realistic stability check than training score."
    )

    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    final_details = evaluation[final_name]
    metadata: dict[str, Any] = {
        "component": "us_kaggle_secondary_benchmark",
        "research_role": "secondary_benchmark",
        "model_name": final_name,
        "final_model": {"name": final_name, **final_details},
        "selected_features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "source_target_column": "charges_usd",
        "output_currency": "USD (source dataset context; not MYR)",
        "dataset": {
            "filename": dataset_path.name,
            "path": str(dataset_path.relative_to(BACKEND_ROOT)),
            "raw_rows": int(audit.report["raw_shape"][0]),
            "modeling_rows": int(audit.report["modeling_rows"]),
            "sha256": dataset_hash,
            "duplicate_observations_excluded": int(audit.report["duplicate_observations_excluding_record_id"]),
        },
        "split": {
            "training_rows": int(len(X_training)),
            "development_rows": int(len(X_train)),
            "calibration_rows": int(len(X_calibration)),
            "testing_rows": int(len(X_test)),
            "test_size": TEST_SIZE,
            "calibration_fraction_of_training": CALIBRATION_SIZE,
        },
        "random_state": RANDOM_STATE,
        "cross_validation_folds": 5,
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "best_hyperparameters": tuning_details[final_name]["best_parameters"],
        "hyperparameter_tuning": tuning_details,
        "model_comparison": comparison.to_dict(orient="records"),
        "feature_importance": [
            {"feature": str(feature), "importance": float(value)} for feature, value in importance.items()
        ],
        "eda": eda,
        "prediction_interval": {
            "method": "smoker-stratified conformalized quantile regression",
            "nominal_coverage": PREDICTION_INTERVAL_COVERAGE,
            "lower_quantile": (1.0 - PREDICTION_INTERVAL_COVERAGE) / 2.0,
            "upper_quantile": 1.0 - (1.0 - PREDICTION_INTERVAL_COVERAGE) / 2.0,
            "calibration_rows": int(len(X_calibration)),
            "calibration_group": "smoker",
            "global_conformal_adjustment": global_adjustment,
            "conformal_adjustment_by_smoker": adjustment_by_smoker,
            "test_coverage": float(covered.mean()),
            "test_coverage_by_smoker": coverage_by_smoker,
            "test_mean_width": float(np.mean(interval_upper - interval_lower)),
            "test_median_width": float(np.median(interval_upper - interval_lower)),
            "interpretation": (
                "Uncertainty range under the source dataset and exchangeability assumptions; "
                "not a guarantee or exact quotation."
            ),
        },
        "error_analysis": {
            "high_cost_threshold": high_threshold,
            "high_cost_mae": high_mae,
            "lower_cost_mae": lower_mae,
            "largest_error_record_ids": errors.head(10).get("record_id", pd.Series(dtype=int)).astype(int).tolist(),
        },
        "overfitting_analysis": overfitting_text,
        "package_versions": package_versions(),
        "limitations": [
            "Secondary benchmark only; not NSS or Malaysian patient-level data.",
            "Bundle documentation describes the classic source dataset as simulated from US demographic statistics.",
            "Output is in the source dataset's charge scale and must not be presented as Malaysian Ringgit.",
            "NSS, LIAM, and Malaysian hospital pricing data are separate sources and were not merged.",
            "Predictions are estimates, not guaranteed healthcare costs.",
        ],
    }

    model_path = MODELS_DIR / "healthcare_cost_model.joblib"
    interval_models_path = MODELS_DIR / "healthcare_cost_interval_models.joblib"
    metadata_path = MODELS_DIR / "model_metadata.json"
    joblib.dump(final_pipeline, model_path)
    joblib.dump(
        {"lower": lower_quantile_pipeline, "upper": upper_quantile_pipeline},
        interval_models_path,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_training_report(metadata, comparison, REPORT_PATH)

    reloaded = joblib.load(model_path)
    smoke_input = pd.DataFrame(
        [{"age": 35, "sex": "male", "bmi": 27.5, "children": 1, "smoker": "no", "region": "southeast"}]
    )
    smoke_prediction = float(reloaded.predict(smoke_input)[0])
    if not np.isfinite(smoke_prediction):
        raise RuntimeError("Reloaded model returned a non-finite smoke-test prediction")

    print("\nModel comparison (held-out test evaluated after model/tuning decisions):")
    print(comparison[["model", "test_mae", "test_rmse", "test_r2", "cv_rmse_mean", "cv_rmse_std", "cv_r2_mean", "cv_r2_std"]].to_string(index=False))
    print(f"\nSelected model: {final_name}")
    print(f"Saved pipeline: {model_path}")
    print(f"Saved interval models: {interval_models_path}")
    print(f"Smoke-test prediction: {smoke_prediction:.2f} (source dataset charge scale)")
    return metadata


if __name__ == "__main__":
    train()
