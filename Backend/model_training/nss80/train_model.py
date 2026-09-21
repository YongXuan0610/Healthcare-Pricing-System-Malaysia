"""Train the NSS 80th Round inpatient medical-expenditure primary model.

Run from ``Backend``:
    .\.venv\Scripts\python.exe -m model_training.nss80.train_model
"""

from __future__ import annotations

from datetime import datetime, timezone
import html
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from model_training.nss80.codebook import allowed_values  # type: ignore
    from model_training.nss80.data import (  # type: ignore
        CATEGORICAL_FEATURES,
        FEATURES,
        NUMERIC_FEATURES,
        TARGET,
        load_and_prepare_dataset,
        sha256_file,
    )
    from model_training.nss80.evaluate import interval_metrics, regression_metrics  # type: ignore
else:
    from .codebook import allowed_values
    from .data import (
        CATEGORICAL_FEATURES,
        FEATURES,
        NUMERIC_FEATURES,
        TARGET,
        load_and_prepare_dataset,
        sha256_file,
    )
    from .evaluate import interval_metrics, regression_metrics


RANDOM_STATE = 42
TEST_SIZE = 0.20
CALIBRATION_SHARE_OF_DEVELOPMENT = 0.20
CV_FOLDS = 3
INTERVAL_COVERAGE = 0.90
BACKEND_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = BACKEND_ROOT / "datasets" / "nss80" / "raw"
PROCESSED_PATH = (
    BACKEND_ROOT / "datasets" / "nss80" / "processed" / "nss80_inpatient_model.csv"
)
MODELS_DIR = BACKEND_ROOT / "models" / "nss80"
REPORTS_DIR = BACKEND_ROOT / "reports" / "nss80"
PLOTS_DIR = REPORTS_DIR / "plots"
TRAINING_REPORT_PATH = REPORTS_DIR / "MODEL_TRAINING_REPORT.md"
INSPECTION_REPORT_PATH = REPORTS_DIR / "DATASET_INSPECTION_REPORT.md"


def make_preprocessor() -> ColumnTransformer:
    categorical = Pipeline(
        steps=[
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
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def make_histogram_pipeline(
    *, loss: str = "squared_error", quantile: float | None = None
) -> Pipeline:
    categorical_mask = [False] * len(NUMERIC_FEATURES) + [True] * len(
        CATEGORICAL_FEATURES
    )
    model = HistGradientBoostingRegressor(
        loss=loss,
        quantile=quantile,
        learning_rate=0.08,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        categorical_features=categorical_mask,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=25,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessor", make_preprocessor()), ("model", model)])


def _predict(pipeline: Pipeline, X: pd.DataFrame, transform: str) -> np.ndarray:
    prediction = np.asarray(pipeline.predict(X), dtype=float)
    if transform == "log1p":
        prediction = np.expm1(prediction)
    return np.maximum(0.0, prediction)


def _normalized_weights(values: pd.Series) -> np.ndarray:
    weights = values.to_numpy(dtype=float)
    return weights / float(np.mean(weights))


def _fit_candidate(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
) -> tuple[Pipeline, str]:
    if name == "hist_gradient_boosting_log1p":
        pipeline = make_histogram_pipeline(loss="squared_error")
        pipeline.fit(X, np.log1p(y), model__sample_weight=weights)
        return pipeline, "log1p"
    if name == "hist_gradient_boosting_absolute":
        pipeline = make_histogram_pipeline(loss="absolute_error")
        pipeline.fit(X, y, model__sample_weight=weights)
        return pipeline, "identity"
    raise ValueError(f"Unknown candidate: {name}")


def _cross_validate_candidates(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    weights: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cv = GroupKFold(n_splits=CV_FOLDS)
    candidate_names = [
        "hist_gradient_boosting_log1p",
        "hist_gradient_boosting_absolute",
    ]
    for name in candidate_names:
        fold_metrics: list[dict[str, float]] = []
        for fold, (train_index, valid_index) in enumerate(
            group_cv.split(X, y, groups), start=1
        ):
            pipeline, transform = _fit_candidate(
                name,
                X.iloc[train_index],
                y.iloc[train_index],
                weights[train_index],
            )
            prediction = _predict(pipeline, X.iloc[valid_index], transform)
            unweighted = regression_metrics(
                y.iloc[valid_index].to_numpy(), prediction
            )
            weighted = regression_metrics(
                y.iloc[valid_index].to_numpy(),
                prediction,
                sample_weight=weights[valid_index],
            )
            metrics = {
                **unweighted,
                **{f"weighted_{key}": value for key, value in weighted.items()},
            }
            fold_metrics.append(metrics)
            print(
                f"{name} fold {fold}: MAE={metrics['mae']:.2f}, "
                f"RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}"
            )
        row: dict[str, Any] = {"model": name}
        for metric in fold_metrics[0]:
            values = [fold[metric] for fold in fold_metrics]
            row[f"cv_{metric}_mean"] = float(np.mean(values))
            row[f"cv_{metric}_std"] = float(np.std(values, ddof=1))
        rows.append(row)

    median_prediction = np.full(len(y), float(np.median(y)))
    baseline = regression_metrics(y.to_numpy(), median_prediction)
    rows.append(
        {
            "model": "dummy_median_reference",
            **{f"cv_{key}_mean": value for key, value in baseline.items()},
            **{f"cv_{key}_std": 0.0 for key in baseline},
            **{f"cv_weighted_{key}_mean": value for key, value in baseline.items()},
            **{f"cv_weighted_{key}_std": 0.0 for key in baseline},
        }
    )
    return pd.DataFrame(rows).sort_values("cv_mae_mean", ignore_index=True)


def _conformal_quantile(scores: np.ndarray, coverage: float) -> tuple[float, float]:
    n = len(scores)
    probability = min(1.0, math.ceil((n + 1) * coverage) / n)
    adjustment = float(np.quantile(scores, probability, method="higher"))
    return adjustment, probability


def _permutation_importance(
    pipeline: Pipeline,
    transform: str,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    sample_size: int = 10_000,
    repeats: int = 3,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    if len(X) > sample_size:
        positions = rng.choice(len(X), size=sample_size, replace=False)
        sample = X.iloc[positions].reset_index(drop=True)
        sample_y = y.iloc[positions].reset_index(drop=True)
    else:
        sample = X.reset_index(drop=True)
        sample_y = y.reset_index(drop=True)
    baseline = regression_metrics(
        sample_y.to_numpy(), _predict(pipeline, sample, transform)
    )["mae"]
    rows = []
    for feature in FEATURES:
        increases = []
        for _ in range(repeats):
            shuffled = sample.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            mae = regression_metrics(
                sample_y.to_numpy(), _predict(pipeline, shuffled, transform)
            )["mae"]
            increases.append(mae - baseline)
        rows.append(
            {
                "feature": feature,
                "mae_increase_mean": float(np.mean(increases)),
                "mae_increase_std": float(np.std(increases, ddof=1)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mae_increase_mean", ascending=False, ignore_index=True
    )


def _svg_scatter(
    actual: np.ndarray,
    predicted: np.ndarray,
    path: Path,
    *,
    limit: int = 5_000,
) -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    if len(actual) > limit:
        selected = rng.choice(len(actual), size=limit, replace=False)
        actual = actual[selected]
        predicted = predicted[selected]
    cap = float(max(np.quantile(actual, 0.99), np.quantile(predicted, 0.99), 1.0))
    width, height, margin = 860, 540, 60
    plot_width, plot_height = width - 2 * margin, height - 2 * margin
    points = []
    for x_value, y_value in zip(actual, predicted):
        x = margin + min(float(x_value), cap) / cap * plot_width
        y = height - margin - min(float(y_value), cap) / cap * plot_height
        points.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.5" fill="#2563eb" fill-opacity="0.28" />'
        )
    diagonal = (
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" '
        f'y2="{margin}" stroke="#dc2626" stroke-width="2" />'
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">NSS 80 test set: actual vs predicted medical expenditure</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827" />
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827" />
{diagonal}
{''.join(points)}
<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="Arial" font-size="13">Actual expenditure (INR; values capped at test P99 for display)</text>
<text x="18" y="{height/2}" text-anchor="middle" transform="rotate(-90 18 {height/2})" font-family="Arial" font-size="13">Predicted expenditure (INR)</text>
<text x="{width-margin}" y="{height-margin+20}" text-anchor="end" font-family="Arial" font-size="11">INR {cap:,.0f}</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def _svg_importance(importance: pd.DataFrame, path: Path) -> None:
    frame = importance.sort_values("mae_increase_mean").tail(12)
    width, height, margin_left, margin_right = 940, 600, 270, 50
    top, bottom = 55, 45
    plot_width = width - margin_left - margin_right
    bar_height = (height - top - bottom) / max(len(frame), 1)
    maximum = max(float(frame["mae_increase_mean"].max()), 1.0)
    bars = []
    for index, row in enumerate(frame.itertuples(index=False)):
        y = top + index * bar_height + 4
        bar_width = max(0.0, row.mae_increase_mean) / maximum * plot_width
        bars.append(
            f'<text x="{margin_left-10}" y="{y+bar_height*0.55:.1f}" text-anchor="end" font-family="Arial" font-size="12">{html.escape(row.feature)}</text>'
        )
        bars.append(
            f'<rect x="{margin_left}" y="{y:.1f}" width="{bar_width:.1f}" height="{max(bar_height-8, 4):.1f}" fill="#0f766e" />'
        )
        bars.append(
            f'<text x="{margin_left+bar_width+6:.1f}" y="{y+bar_height*0.55:.1f}" font-family="Arial" font-size="11">{row.mae_increase_mean:,.0f}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">Permutation importance on NSS 80 test data</text>
{''.join(bars)}
<text x="{margin_left + plot_width/2}" y="{height-12}" text-anchor="middle" font-family="Arial" font-size="13">Increase in MAE after permutation (INR)</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def _write_inspection_report(report: dict[str, Any], path: Path) -> None:
    target = report["target_summary_inr"]
    content = f"""# NSS 80 Dataset Inspection Report

## Source

- Official survey: Survey on Household Social Consumption: Health, January-December 2025 (NSS 80th Round, Schedule 25.0).
- Raw files are preserved byte-for-byte in `Backend/datasets/nss80/raw/`.
- Level 2 rows: {report['source_level_rows']['hhscsL2']:,}; Level 3 rows: {report['source_level_rows']['hhscsL3']:,}; Level 4 inpatient rows: {report['source_level_rows']['hhscsL4']:,}.

## Modeling cohort

- Inpatient cases retained: {report['modeling_rows']:,}.
- First-stage units represented: {report['fsu_count']:,}.
- Households represented: {report['household_count']:,}.
- Roster joins unavailable and excluded: {report['unmatched_roster_rows_excluded']:,} ({report['unmatched_roster_rows_excluded'] / report['source_level_rows']['hhscsL4']:.3%}).
- Duplicate raw rows: {report['duplicate_raw_rows']:,}; duplicate case IDs: {report['duplicate_case_ids']:,}.

## Target

{report['target_definition']}

- Zero-cost cases retained as valid free/subsidised care: {report['zero_target_rows']:,}.
- Median: INR {target['50%']:,.0f}; mean: INR {target['mean']:,.0f}; P90: INR {target['90%']:,.0f}; P99: INR {target['99%']:,.0f}; maximum: INR {target['max']:,.0f}.

## Leakage controls

The model excludes every component used to calculate the target, target-derived totals, reimbursement, and income-loss fields. Predictors are demographic, geographic, clinical, institution/ward, length-of-stay, service-receipt, and free-care descriptors.

The raw source is never overwritten. `nss80_inpatient_model.csv` is a decoded processed copy with a documented join from the living/deceased member rosters to inpatient cases.
"""
    path.write_text(content, encoding="utf-8")


def _write_training_report(metadata: dict[str, Any], path: Path) -> None:
    test = metadata["test_metrics"]
    weighted = metadata["weighted_test_metrics"]
    interval = metadata["prediction_interval"]
    content = f"""# NSS 80 Primary Model Training Report

## Research role

This is the project's primary machine-learning research model. It predicts per-case inpatient medical expenditure in the source survey's Indian-rupee context. It is separate from the US Kaggle benchmark and is never merged with Malaysian LIAM or hospital-pricing records.

## Design

- Observations: {metadata['dataset']['modeling_rows']:,} inpatient cases from {metadata['dataset']['fsu_count']:,} first-stage units.
- Target: NSS Schedule 25.0 Block 7 item 12, total medical expenditure (items 6-11), INR.
- Leakage control: component expenditures, derived totals, reimbursement, and income loss are excluded.
- Split: first-stage-unit grouped {metadata['split']['train_rows']:,} train / {metadata['split']['calibration_rows']:,} calibration / {metadata['split']['test_rows']:,} test rows. No FSU occurs in more than one split.
- Candidate selection: {CV_FOLDS}-fold grouped cross-validation on the development partition, ranked by MAE.
- Selected model: `{metadata['selected_model']}` using `{metadata['target_transform']}` target handling.
- Survey multipliers are normalized to mean 1 and used as training weights; unweighted and survey-weighted test metrics are both reported.

## Held-out test results

| Metric | Unweighted | Survey-weighted |
|---|---:|---:|
| MAE (INR) | {test['mae']:,.2f} | {weighted['mae']:,.2f} |
| RMSE (INR) | {test['rmse']:,.2f} | {weighted['rmse']:,.2f} |
| R-squared | {test['r2']:.4f} | {weighted['r2']:.4f} |
| Median absolute error (INR) | {test['median_absolute_error']:,.2f} | {weighted['median_absolute_error']:,.2f} |
| RMSLE | {test['rmsle']:.4f} | {weighted['rmsle']:.4f} |

## Prediction interval

The 90% interval uses split conformalized quantile regression: 5th- and 95th-percentile histogram-gradient-boosting models plus a calibration-set correction computed on unseen FSUs.

- Nominal coverage: {interval['nominal_coverage']:.0%}
- Held-out empirical coverage: {interval['test_empirical_coverage']:.2%}
- Median held-out width: INR {interval['test_median_width']:,.2f}
- Conformal adjustment: INR {interval['conformal_adjustment']:,.2f}

Coverage is marginal for data exchangeable with the survey sample; it is not a guarantee for an individual bill.

## Limitations

- The model represents India in January-December 2025 and outputs INR. It is not a Malaysian price estimator and no currency conversion is applied.
- Survey responses and expenditure are observational/self-reported and subject to recall and measurement error.
- High-cost care is strongly right-skewed; point errors can be large, especially in the tail.
- Results are research estimates, not clinical advice, insurance authorization, or a hospital quotation.
- State, institution, ward, and service fields must be interpreted using the official Schedule 25.0 codebook.
"""
    path.write_text(content, encoding="utf-8")


def train_v1() -> dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    audit = load_and_prepare_dataset(RAW_DIR)
    frame = audit.modeling.reset_index(drop=True)
    frame.to_csv(PROCESSED_PATH, index=False)

    X = frame[FEATURES]
    y = frame[TARGET]
    groups = frame["fsu_id"]
    weights = _normalized_weights(frame["survey_multiplier"])

    outer = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    development_index, test_index = next(outer.split(X, y, groups))
    development = frame.iloc[development_index].reset_index(drop=True)
    test = frame.iloc[test_index].reset_index(drop=True)

    calibration_split = GroupShuffleSplit(
        n_splits=1,
        test_size=CALIBRATION_SHARE_OF_DEVELOPMENT,
        random_state=RANDOM_STATE + 1,
    )
    train_position, calibration_position = next(
        calibration_split.split(
            development[FEATURES], development[TARGET], development["fsu_id"]
        )
    )
    train_frame = development.iloc[train_position].reset_index(drop=True)
    calibration = development.iloc[calibration_position].reset_index(drop=True)

    split_groups = [
        set(train_frame["fsu_id"]),
        set(calibration["fsu_id"]),
        set(test["fsu_id"]),
    ]
    if (
        split_groups[0] & split_groups[1]
        or split_groups[0] & split_groups[2]
        or split_groups[1] & split_groups[2]
    ):
        raise RuntimeError("FSU leakage detected across train/calibration/test splits")

    development_weights = _normalized_weights(development["survey_multiplier"])
    comparison = _cross_validate_candidates(
        development[FEATURES],
        development[TARGET],
        development["fsu_id"],
        development_weights,
    )
    candidate_rows = comparison[
        comparison["model"].str.startswith("hist_gradient_boosting")
    ]
    selected_name = str(candidate_rows.iloc[0]["model"])

    train_weights = _normalized_weights(train_frame["survey_multiplier"])
    point_pipeline, target_transform = _fit_candidate(
        selected_name,
        train_frame[FEATURES],
        train_frame[TARGET],
        train_weights,
    )

    lower_pipeline = make_histogram_pipeline(
        loss="quantile", quantile=(1 - INTERVAL_COVERAGE) / 2
    )
    upper_pipeline = make_histogram_pipeline(
        loss="quantile", quantile=1 - (1 - INTERVAL_COVERAGE) / 2
    )
    lower_pipeline.fit(
        train_frame[FEATURES],
        train_frame[TARGET],
        model__sample_weight=train_weights,
    )
    upper_pipeline.fit(
        train_frame[FEATURES],
        train_frame[TARGET],
        model__sample_weight=train_weights,
    )

    calibration_lower = np.maximum(
        0.0, lower_pipeline.predict(calibration[FEATURES])
    )
    calibration_upper = np.maximum(
        calibration_lower, upper_pipeline.predict(calibration[FEATURES])
    )
    scores = np.maximum(
        calibration_lower - calibration[TARGET].to_numpy(),
        calibration[TARGET].to_numpy() - calibration_upper,
    )
    conformal_adjustment, conformal_probability = _conformal_quantile(
        scores, INTERVAL_COVERAGE
    )

    test_prediction = _predict(point_pipeline, test[FEATURES], target_transform)
    raw_test_lower = np.maximum(0.0, lower_pipeline.predict(test[FEATURES]))
    raw_test_upper = np.maximum(raw_test_lower, upper_pipeline.predict(test[FEATURES]))
    test_lower = np.maximum(0.0, raw_test_lower - conformal_adjustment)
    test_upper = raw_test_upper + conformal_adjustment
    test_lower = np.minimum(test_lower, test_prediction)
    test_upper = np.maximum(test_upper, test_prediction)

    test_weights = _normalized_weights(test["survey_multiplier"])
    test_metrics = regression_metrics(test[TARGET].to_numpy(), test_prediction)
    weighted_test_metrics = regression_metrics(
        test[TARGET].to_numpy(), test_prediction, sample_weight=test_weights
    )
    test_interval = interval_metrics(
        test[TARGET].to_numpy(), test_lower, test_upper
    )

    importance = _permutation_importance(
        point_pipeline,
        target_transform,
        test[FEATURES],
        test[TARGET],
    )
    largest_errors = test[["case_id", *FEATURES, TARGET]].copy()
    largest_errors["predicted_medical_expenditure_inr"] = test_prediction
    largest_errors["absolute_error_inr"] = np.abs(
        largest_errors[TARGET] - test_prediction
    )
    largest_errors = largest_errors.nlargest(50, "absolute_error_inr")

    comparison.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    largest_errors.to_csv(MODELS_DIR / "largest_prediction_errors.csv", index=False)
    joblib.dump(
        {
            "pipeline": point_pipeline,
            "target_transform": target_transform,
            "features": FEATURES,
        },
        MODELS_DIR / "primary_model.joblib",
    )
    joblib.dump(
        {"lower": lower_pipeline, "upper": upper_pipeline},
        MODELS_DIR / "interval_models.joblib",
    )

    metadata: dict[str, Any] = {
        "component": "nss80_primary_research_model",
        "research_role": "primary",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected_name,
        "target_transform": target_transform,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "allowed_values": allowed_values(),
        "target": TARGET,
        "target_unit": "INR per inpatient case",
        "dataset": {
            **audit.report,
            "processed_path": str(PROCESSED_PATH.relative_to(BACKEND_ROOT)),
            "processed_sha256": sha256_file(PROCESSED_PATH),
            "catalog_url": "https://microdata.gov.in/NADA/index.php/catalog/290",
            "reference_id": "DDI-IND-NSO-HSCHealth80R-Jan2025-Dec2025",
        },
        "split": {
            "strategy": "GroupShuffleSplit by first-stage unit (FSU)",
            "random_state": RANDOM_STATE,
            "train_rows": int(len(train_frame)),
            "calibration_rows": int(len(calibration)),
            "test_rows": int(len(test)),
            "train_fsu_count": int(train_frame["fsu_id"].nunique()),
            "calibration_fsu_count": int(calibration["fsu_id"].nunique()),
            "test_fsu_count": int(test["fsu_id"].nunique()),
            "fsu_overlap_count": 0,
        },
        "cross_validation": comparison.to_dict(orient="records"),
        "test_metrics": test_metrics,
        "weighted_test_metrics": weighted_test_metrics,
        "prediction_interval": {
            "method": "split_conformalized_quantile_regression",
            "nominal_coverage": INTERVAL_COVERAGE,
            "lower_quantile": (1 - INTERVAL_COVERAGE) / 2,
            "upper_quantile": 1 - (1 - INTERVAL_COVERAGE) / 2,
            "calibration_rows": int(len(calibration)),
            "conformal_quantile_probability": conformal_probability,
            "conformal_adjustment": conformal_adjustment,
            "test_empirical_coverage": test_interval["empirical_coverage"],
            "test_mean_width": test_interval["mean_width"],
            "test_median_width": test_interval["median_width"],
            "interpretation": (
                "A marginal research uncertainty interval calibrated on unseen NSS "
                "first-stage units; not an individual guarantee or quotation."
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "joblib": importlib.metadata.version("joblib"),
        },
        "limitations": [
            "India January-December 2025 context; output is INR, not MYR or USD.",
            "Observational self-reported survey expenditure with recall and measurement error.",
            "Strongly right-skewed cost distribution produces large high-cost tail errors.",
            "Research estimate only; not clinical advice, insurance authorization, or a bill.",
        ],
    }
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_inspection_report(audit.report, INSPECTION_REPORT_PATH)
    _write_training_report(metadata, TRAINING_REPORT_PATH)
    _svg_scatter(
        test[TARGET].to_numpy(),
        test_prediction,
        PLOTS_DIR / "actual_vs_predicted.svg",
    )
    _svg_importance(importance, PLOTS_DIR / "feature_importance.svg")

    reloaded = joblib.load(MODELS_DIR / "primary_model.joblib")
    smoke_prediction = _predict(
        reloaded["pipeline"], test[FEATURES].iloc[:1], reloaded["target_transform"]
    )[0]
    if not np.isfinite(smoke_prediction):
        raise RuntimeError("Reloaded NSS model produced a non-finite prediction")

    print("\nNSS 80 model comparison:")
    print(comparison.to_string(index=False))
    print(f"\nSelected: {selected_name}")
    print(f"Test MAE: INR {test_metrics['mae']:,.2f}")
    print(f"Test R2: {test_metrics['r2']:.4f}")
    print(f"90% interval coverage: {test_interval['empirical_coverage']:.2%}")
    print(f"Saved primary model: {MODELS_DIR / 'primary_model.joblib'}")
    return metadata


from .train_model_v2 import train


if __name__ == "__main__":
    train()
