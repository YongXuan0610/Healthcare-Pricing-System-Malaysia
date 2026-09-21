"""Train and analyse the payment-safe NSS 80 primary research model.

The final test split is touched only after model selection. Raw NSS files are
never modified, and the US/LIAM components are never read by this module.
"""

from __future__ import annotations

from datetime import datetime, timezone
import html
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .codebook import allowed_values
from .data import FEATURES as INPUT_FEATURES
from .data import TARGET, load_and_prepare_dataset, sha256_file
from .evaluate import interval_metrics, regression_metrics
from .modeling import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    leakage_safe_features,
    make_leakage_safe_hgb_pipeline,
)


RANDOM_STATE = 42
INTERVAL_COVERAGE = 0.90
BACKEND_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = BACKEND_ROOT / "datasets" / "nss80" / "raw"
PROCESSED_PATH = (
    BACKEND_ROOT / "datasets" / "nss80" / "processed" / "nss80_inpatient_model.csv"
)
MODELS_DIR = BACKEND_ROOT / "models" / "nss80"
REPORTS_DIR = BACKEND_ROOT / "reports" / "nss80"
EXPERIMENTS_DIR = REPORTS_DIR / "experiments_v2"
PLOTS_DIR = REPORTS_DIR / "plots"

BASELINE_PARAMETERS = {
    "learning_rate": 0.08,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
}
SELECTED_PARAMETERS = {
    "learning_rate": 0.05,
    "max_iter": 600,
    "max_leaf_nodes": 127,
    "min_samples_leaf": 40,
    "l2_regularization": 10.0,
}


def _weights(series: pd.Series) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    return values / float(values.mean())


def _predict_log_model(model: Any, X_safe: pd.DataFrame) -> np.ndarray:
    return np.maximum(0.0, np.expm1(np.asarray(model.predict(X_safe), dtype=float)))


def _legacy_prediction(artifact: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    prediction = np.asarray(artifact["pipeline"].predict(X[INPUT_FEATURES]), dtype=float)
    if artifact["target_transform"] == "log1p":
        prediction = np.expm1(prediction)
    return np.maximum(0.0, prediction)


def _conformal_quantile(scores: np.ndarray, coverage: float) -> tuple[float, float]:
    count = len(scores)
    probability = min(1.0, math.ceil((count + 1) * coverage) / count)
    adjustment = max(0.0, float(np.quantile(scores, probability, method="higher")))
    return adjustment, probability


def _grouped_cv(
    frame: pd.DataFrame,
    name: str,
    parameters: dict[str, Any] | None,
) -> dict[str, Any]:
    folds: list[dict[str, float]] = []
    started = time.perf_counter()
    splitter = GroupKFold(n_splits=3)
    X_safe = leakage_safe_features(frame[INPUT_FEATURES])
    for fold, (train_index, valid_index) in enumerate(
        splitter.split(X_safe, frame[TARGET], frame["fsu_id"]), start=1
    ):
        if parameters is None:
            model: Any = DummyRegressor(strategy="median")
            model.fit(
                X_safe.iloc[train_index],
                frame[TARGET].iloc[train_index],
                sample_weight=_weights(frame["survey_multiplier"].iloc[train_index]),
            )
            prediction = np.maximum(
                0.0, np.asarray(model.predict(X_safe.iloc[valid_index]), dtype=float)
            )
        else:
            model = make_leakage_safe_hgb_pipeline(**parameters)
            model.fit(
                X_safe.iloc[train_index],
                np.log1p(frame[TARGET].iloc[train_index]),
                model__sample_weight=_weights(
                    frame["survey_multiplier"].iloc[train_index]
                ),
            )
            prediction = _predict_log_model(model, X_safe.iloc[valid_index])
        metrics = regression_metrics(
            frame[TARGET].iloc[valid_index].to_numpy(), prediction
        )
        weighted = regression_metrics(
            frame[TARGET].iloc[valid_index].to_numpy(),
            prediction,
            sample_weight=_weights(frame["survey_multiplier"].iloc[valid_index]),
        )
        folds.append(
            {**metrics, **{f"weighted_{key}": value for key, value in weighted.items()}}
        )
        print(
            f"{name} fold {fold}: MAE={metrics['mae']:.2f}, "
            f"RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}",
            flush=True,
        )
    result: dict[str, Any] = {
        "model": name,
        "target_transform": "identity" if parameters is None else "log1p",
        "feature_set": "payment_safe_v2",
        "parameters": parameters or {"strategy": "median"},
        "cv_folds": 3,
        "fit_seconds": time.perf_counter() - started,
    }
    for metric in folds[0]:
        values = [fold[metric] for fold in folds]
        result[f"cv_{metric}_mean"] = float(np.mean(values))
        result[f"cv_{metric}_std"] = float(np.std(values, ddof=1))
    return result


def _metric_row(
    model: str,
    y: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    metrics = regression_metrics(y, prediction)
    weighted = regression_metrics(y, prediction, sample_weight=weights)
    return {
        "model": model,
        "evaluation_data": "locked FSU-grouped test",
        **metrics,
        **{f"weighted_{key}": value for key, value in weighted.items()},
    }


def _safe_group_metrics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    group: pd.Series,
    dimension: str,
    *,
    minimum_records: int = 100,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    actual = frame[TARGET].to_numpy(dtype=float)
    group = group.reset_index(drop=True)
    for value in sorted(group.dropna().astype(str).unique()):
        mask = group.astype(str).eq(value).to_numpy()
        count = int(mask.sum())
        if count < minimum_records:
            continue
        metrics = regression_metrics(actual[mask], prediction[mask])
        values = actual[mask]
        rows.append(
            {
                "dimension": dimension,
                "group": value,
                "records": count,
                "actual_mean_inr": float(values.mean()),
                "actual_median_inr": float(np.median(values)),
                "actual_p90_inr": float(np.quantile(values, 0.90)),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _permutation_importance(
    model: Any,
    X_safe: pd.DataFrame,
    y: pd.Series,
    sample_size: int = 10_000,
    repeats: int = 3,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    if len(X_safe) > sample_size:
        selected = rng.choice(len(X_safe), sample_size, replace=False)
        sample_X = X_safe.iloc[selected].reset_index(drop=True)
        sample_y = y.iloc[selected].reset_index(drop=True)
    else:
        sample_X = X_safe.reset_index(drop=True)
        sample_y = y.reset_index(drop=True)
    baseline = regression_metrics(
        sample_y.to_numpy(), _predict_log_model(model, sample_X)
    )["mae"]
    rows: list[dict[str, Any]] = []
    for feature in MODEL_FEATURES:
        increases: list[float] = []
        for _ in range(repeats):
            shuffled = sample_X.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            mae = regression_metrics(
                sample_y.to_numpy(), _predict_log_model(model, shuffled)
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


def _svg_histogram(
    values: np.ndarray,
    path: Path,
    title: str,
    axis_label: str,
    *,
    log_transform: bool,
) -> None:
    transformed = np.log1p(values) if log_transform else values.copy()
    display = (
        transformed
        if log_transform
        else np.minimum(transformed, np.quantile(transformed, 0.99))
    )
    counts, edges = np.histogram(display, bins=40)
    width, height = 900, 540
    left, right, top, bottom = 75, 35, 55, 65
    plot_width, plot_height = width - left - right, height - top - bottom
    maximum = max(int(counts.max()), 1)
    bars = []
    bar_width = plot_width / len(counts)
    for index, count in enumerate(counts):
        x = left + index * bar_width
        bar_height = count / maximum * plot_height
        bars.append(
            f'<rect x="{x:.1f}" y="{height-bottom-bar_height:.1f}" '
            f'width="{max(bar_width-1, 1):.1f}" height="{bar_height:.1f}" fill="#2563eb" />'
        )
    p99_note = "" if log_transform else " (display capped at P99; no records removed)"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">{html.escape(title)}</text>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>
{''.join(bars)}
<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(axis_label + p99_note)}</text>
<text x="18" y="{height/2}" text-anchor="middle" transform="rotate(-90 18 {height/2})" font-family="Arial" font-size="13">Records</text>
<text x="{left}" y="{height-bottom+20}" font-family="Arial" font-size="11">{edges[0]:,.1f}</text>
<text x="{width-right}" y="{height-bottom+20}" text-anchor="end" font-family="Arial" font-size="11">{edges[-1]:,.1f}</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def _svg_scatter(
    x_values: np.ndarray,
    y_values: np.ndarray,
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    *,
    diagonal: bool = False,
) -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    if len(x_values) > 5_000:
        selected = rng.choice(len(x_values), 5_000, replace=False)
        x_values, y_values = x_values[selected], y_values[selected]
    x_min, x_max = float(np.quantile(x_values, 0.01)), float(np.quantile(x_values, 0.99))
    y_min, y_max = float(np.quantile(y_values, 0.01)), float(np.quantile(y_values, 0.99))
    if diagonal:
        x_min = y_min = 0.0
        x_max = y_max = max(x_max, y_max, 1.0)
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    width, height, margin = 900, 550, 65
    plot_width, plot_height = width - 2 * margin, height - 2 * margin
    points = []
    for x_value, y_value in zip(x_values, y_values):
        x_value = min(max(float(x_value), x_min), x_max)
        y_value = min(max(float(y_value), y_min), y_max)
        x = margin + (x_value - x_min) / (x_max - x_min) * plot_width
        y = height - margin - (y_value - y_min) / (y_max - y_min) * plot_height
        points.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.5" fill="#0f766e" fill-opacity="0.28"/>'
        )
    reference = ""
    if diagonal:
        reference = f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{margin}" stroke="#dc2626" stroke-width="2"/>'
    elif y_min < 0 < y_max:
        zero_y = height - margin - (0 - y_min) / (y_max - y_min) * plot_height
        reference = f'<line x1="{margin}" y1="{zero_y:.1f}" x2="{width-margin}" y2="{zero_y:.1f}" stroke="#dc2626" stroke-width="1.5"/>'
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">{html.escape(title)}</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827"/>
{reference}{''.join(points)}
<text x="{width/2}" y="{height-16}" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(x_label)} (P1-P99 display range)</text>
<text x="18" y="{height/2}" text-anchor="middle" transform="rotate(-90 18 {height/2})" font-family="Arial" font-size="13">{html.escape(y_label)}</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def _svg_importance(importance: pd.DataFrame, path: Path) -> None:
    frame = importance.sort_values("mae_increase_mean").tail(12)
    width, height, left, right, top, bottom = 940, 600, 275, 50, 55, 45
    plot_width = width - left - right
    row_height = (height - top - bottom) / max(len(frame), 1)
    maximum = max(float(frame["mae_increase_mean"].max()), 1.0)
    elements: list[str] = []
    for index, row in enumerate(frame.itertuples(index=False)):
        y = top + index * row_height + 4
        bar_width = max(0.0, row.mae_increase_mean) / maximum * plot_width
        elements.append(
            f'<text x="{left-10}" y="{y+row_height*.55:.1f}" text-anchor="end" font-family="Arial" font-size="12">{html.escape(row.feature)}</text>'
        )
        elements.append(
            f'<rect x="{left}" y="{y:.1f}" width="{bar_width:.1f}" height="{max(row_height-8,4):.1f}" fill="#0f766e"/>'
        )
        elements.append(
            f'<text x="{left+bar_width+6:.1f}" y="{y+row_height*.55:.1f}" font-family="Arial" font-size="11">{row.mae_increase_mean:,.0f}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">Payment-safe permutation importance</text>
{''.join(elements)}
<text x="{left+plot_width/2}" y="{height-12}" text-anchor="middle" font-family="Arial" font-size="13">Increase in test-sample MAE after permutation (INR)</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def _svg_public_private(frame: pd.DataFrame, path: Path) -> None:
    """Plot observed expenditure summaries and model MAE by institution sector."""

    metrics = [
        ("Actual median", "actual_median_inr"),
        ("Actual mean", "actual_mean_inr"),
        ("Actual P90", "actual_p90_inr"),
        ("Model MAE", "mae"),
    ]
    colours = {"Public": "#2563eb", "Private": "#f97316"}
    width, height, left, right, top, bottom = 940, 560, 90, 40, 70, 70
    plot_width, plot_height = width - left - right, height - top - bottom
    maximum = max(float(frame[column].max()) for _, column in metrics)
    group_width = plot_width / len(metrics)
    bar_width = group_width * 0.28
    elements: list[str] = []
    for metric_index, (label, column) in enumerate(metrics):
        centre = left + (metric_index + 0.5) * group_width
        for sector_index, sector in enumerate(["Public", "Private"]):
            row = frame.loc[frame["group"].eq(sector)].iloc[0]
            value = float(row[column])
            x = centre + (sector_index - 0.5) * bar_width - bar_width / 2
            bar_height = value / maximum * plot_height
            y = height - bottom - bar_height
            elements.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{colours[sector]}"/>'
            )
            elements.append(
                f'<text x="{x+bar_width/2:.1f}" y="{max(y-5, 48):.1f}" text-anchor="middle" font-family="Arial" font-size="10">{value:,.0f}</text>'
            )
        elements.append(
            f'<text x="{centre:.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="Arial" font-size="12">{html.escape(label)}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">Observed NSS expenditure and error by institution sector</text>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>
{''.join(elements)}
<rect x="{width-210}" y="45" width="12" height="12" fill="#2563eb"/><text x="{width-192}" y="56" font-family="Arial" font-size="12">Public</text>
<rect x="{width-125}" y="45" width="12" height="12" fill="#f97316"/><text x="{width-107}" y="56" font-family="Arial" font-size="12">Private</text>
<text x="22" y="{height/2}" text-anchor="middle" transform="rotate(-90 22 {height/2})" font-family="Arial" font-size="13">INR</text>
<text x="{width/2}" y="{height-14}" text-anchor="middle" font-family="Arial" font-size="11">Descriptive survey pattern only; not a causal ownership effect.</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def _write_leakage_audit(path: Path) -> None:
    included = {
        "age_years": "Age from the linked person roster; not expenditure-derived.",
        "length_of_stay_days": "Episode utilisation known before the final bill in the stated post-service prediction setting.",
        "gender": "Demographic roster field.",
        "sector": "Rural/urban survey stratum descriptor.",
        "state": "Geographic context.",
        "ailment": "Clinical ailment category.",
        "treatment_system": "Treatment system category.",
        "medical_institution": "Public, private, or charitable/NGO provider type.",
        "ward_class": "Special versus standard/free ward; free versus paying-general detail is collapsed.",
        "*_received": "Binary service utilisation; free/partly-free/paid detail is collapsed.",
    }
    excluded = {
        "b7i6-b7i11": "Direct medical-expenditure components that sum to the target.",
        "b7i13-b7i15": "Target-related totals and non-medical expenditure fields.",
        "b7i16": "Reimbursement information available after expenditure/payment.",
        "b7i20": "Income-loss outcome arising from the episode.",
        "b7i5 / free_medical_service": "Payment/free-care status is target-adjacent and excluded from the estimator.",
        "b6i13-b6i16 payment mode": "Free/partly-free/on-payment distinctions are removed; only service receipt is retained.",
        "b6i9 free vs paying-general": "Payment distinction is removed; both map to standard_or_free.",
    }
    content = "# NSS 80 Feature and Target-Leakage Audit\n\n"
    content += "## Prediction-time assumption\n\nThe model is an episode-level estimator used after the care setting, ward class, services received, and length of stay are known but before the final expenditure amount is supplied. It is not a pre-admission forecast.\n\n"
    content += "## Included features\n\n"
    content += "\n".join(f"- `{key}` — {reason}" for key, reason in included.items())
    content += "\n\n## Excluded leakage variables with reasons\n\n"
    content += "\n".join(f"- `{key}` — {reason}" for key, reason in excluded.items())
    content += "\n\n## Audit conclusion\n\nThe v1 baseline contained categorical payment-status proxies. No direct monetary component was used, but the proxies were removed or collapsed in v2 because they reveal how component services were paid. All raw records, including genuine high-cost and zero-cost observations, remain unchanged.\n"
    path.write_text(content, encoding="utf-8")


def train() -> dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    audit = load_and_prepare_dataset(RAW_DIR)
    frame = audit.modeling.reset_index(drop=True)
    frame.to_csv(PROCESSED_PATH, index=False)

    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    development_index, test_index = next(
        outer.split(frame[INPUT_FEATURES], frame[TARGET], frame["fsu_id"])
    )
    development = frame.iloc[development_index].reset_index(drop=True)
    test = frame.iloc[test_index].reset_index(drop=True)
    calibration_split = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=RANDOM_STATE + 1
    )
    train_position, calibration_position = next(
        calibration_split.split(
            development[INPUT_FEATURES], development[TARGET], development["fsu_id"]
        )
    )
    train_frame = development.iloc[train_position].reset_index(drop=True)
    calibration = development.iloc[calibration_position].reset_index(drop=True)

    split_sets = [
        set(train_frame["fsu_id"]),
        set(calibration["fsu_id"]),
        set(test["fsu_id"]),
    ]
    if (
        split_sets[0] & split_sets[1]
        or split_sets[0] & split_sets[2]
        or split_sets[1] & split_sets[2]
    ):
        raise RuntimeError("FSU leakage detected across train/calibration/test splits")

    cv_results = pd.DataFrame(
        [
            _grouped_cv(train_frame, "dummy_median", None),
            _grouped_cv(
                train_frame,
                "payment_safe_hgb_log1p_baseline",
                BASELINE_PARAMETERS,
            ),
            _grouped_cv(
                train_frame,
                "payment_safe_hgb_log1p_tuned",
                SELECTED_PARAMETERS,
            ),
        ]
    ).sort_values("cv_mae_mean", ignore_index=True)

    X_train = leakage_safe_features(train_frame[INPUT_FEATURES])
    X_calibration = leakage_safe_features(calibration[INPUT_FEATURES])
    X_test = leakage_safe_features(test[INPUT_FEATURES])
    train_weights = _weights(train_frame["survey_multiplier"])
    test_weights = _weights(test["survey_multiplier"])

    point_model = make_leakage_safe_hgb_pipeline(**SELECTED_PARAMETERS)
    point_model.fit(
        X_train,
        np.log1p(train_frame[TARGET]),
        model__sample_weight=train_weights,
    )
    test_prediction = _predict_log_model(point_model, X_test)

    dummy = DummyRegressor(strategy="median")
    dummy.fit(X_train, train_frame[TARGET], sample_weight=train_weights)
    dummy_prediction = np.maximum(0.0, dummy.predict(X_test))

    baseline_artifact = joblib.load(
        MODELS_DIR / "baseline_v1_20260823" / "primary_model.joblib"
    )
    baseline_prediction = _legacy_prediction(
        baseline_artifact, test[INPUT_FEATURES]
    )
    test_comparison = pd.DataFrame(
        [
            _metric_row(
                "dummy_median",
                test[TARGET].to_numpy(),
                dummy_prediction,
                test_weights,
            ),
            _metric_row(
                "original_deployed_v1_log_hgb",
                test[TARGET].to_numpy(),
                baseline_prediction,
                test_weights,
            ),
            _metric_row(
                "selected_payment_safe_v2_log_hgb",
                test[TARGET].to_numpy(),
                test_prediction,
                test_weights,
            ),
        ]
    )

    lower_model = make_leakage_safe_hgb_pipeline(
        **SELECTED_PARAMETERS,
        loss="quantile",
        quantile=(1 - INTERVAL_COVERAGE) / 2,
    )
    upper_model = make_leakage_safe_hgb_pipeline(
        **SELECTED_PARAMETERS,
        loss="quantile",
        quantile=1 - (1 - INTERVAL_COVERAGE) / 2,
    )
    lower_model.fit(
        X_train, train_frame[TARGET], model__sample_weight=train_weights
    )
    upper_model.fit(
        X_train, train_frame[TARGET], model__sample_weight=train_weights
    )
    calibration_lower = np.maximum(0.0, lower_model.predict(X_calibration))
    calibration_upper = np.maximum(
        calibration_lower, upper_model.predict(X_calibration)
    )
    scores = np.maximum(
        calibration_lower - calibration[TARGET].to_numpy(),
        calibration[TARGET].to_numpy() - calibration_upper,
    )
    adjustment, adjustment_probability = _conformal_quantile(
        scores, INTERVAL_COVERAGE
    )
    test_lower = np.maximum(0.0, lower_model.predict(X_test) - adjustment)
    test_upper = np.maximum(test_lower, upper_model.predict(X_test) + adjustment)
    test_lower = np.minimum(test_lower, test_prediction)
    test_upper = np.maximum(test_upper, test_prediction)
    interval_result = interval_metrics(
        test[TARGET].to_numpy(), test_lower, test_upper
    )

    final_metrics = regression_metrics(test[TARGET].to_numpy(), test_prediction)
    weighted_final_metrics = regression_metrics(
        test[TARGET].to_numpy(), test_prediction, sample_weight=test_weights
    )
    dummy_metrics = regression_metrics(test[TARGET].to_numpy(), dummy_prediction)
    baseline_metrics = regression_metrics(
        test[TARGET].to_numpy(), baseline_prediction
    )

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

    safe_test = X_test.copy()
    age_group = pd.cut(
        test["age_years"],
        bins=[-1, 4, 17, 44, 59, 74, np.inf],
        labels=["0-4", "5-17", "18-44", "45-59", "60-74", "75+"],
    ).astype(str)
    stay_group = pd.cut(
        test["length_of_stay_days"],
        bins=[0, 1, 3, 7, 14, 30, np.inf],
        labels=["1", "2-3", "4-7", "8-14", "15-30", "31+"],
    ).astype(str)
    institution_group = test["medical_institution"].map(
        {
            "government_or_public_hospital": "Public",
            "private_hospital": "Private",
            "charitable_trust_or_ngo_hospital": "Charitable/NGO",
        }
    )
    subgroup_frames = [
        _safe_group_metrics(
            test,
            test_prediction,
            institution_group,
            "hospital_sector",
            minimum_records=100,
        ),
        _safe_group_metrics(
            test,
            test_prediction,
            safe_test["ward_class"],
            "ward_class",
            minimum_records=100,
        ),
        _safe_group_metrics(
            test, test_prediction, age_group, "age_group", minimum_records=100
        ),
        _safe_group_metrics(
            test, test_prediction, test["state"], "state", minimum_records=300
        ),
        _safe_group_metrics(
            test,
            test_prediction,
            safe_test["surgery_received"],
            "surgery_received",
            minimum_records=100,
        ),
        _safe_group_metrics(
            test,
            test_prediction,
            stay_group,
            "length_of_stay_category",
            minimum_records=100,
        ),
    ]
    subgroup_performance = pd.concat(subgroup_frames, ignore_index=True)
    public_private = subgroup_performance[
        subgroup_performance["dimension"].eq("hospital_sector")
        & subgroup_performance["group"].isin(["Public", "Private"])
    ].reset_index(drop=True)

    training_median = float(train_frame[TARGET].quantile(0.50))
    training_p90 = float(train_frame[TARGET].quantile(0.90))
    expenditure_band = pd.cut(
        test[TARGET],
        bins=[-1, training_median, training_p90, np.inf],
        labels=[
            "low_at_or_below_training_median",
            "medium_training_p50_to_p90",
            "high_above_training_p90",
        ],
    ).astype(str)
    band_performance = _safe_group_metrics(
        test,
        test_prediction,
        expenditure_band,
        "actual_expenditure_band",
        minimum_records=100,
    )

    p95 = float(train_frame[TARGET].quantile(0.95))
    p99 = float(train_frame[TARGET].quantile(0.99))
    outlier_rows: list[dict[str, Any]] = []
    for label, mask in {
        "all_valid_records": np.ones(len(test), dtype=bool),
        "below_training_p95": test[TARGET].to_numpy() < p95,
        "top_5_percent_threshold": test[TARGET].to_numpy() >= p95,
        "top_1_percent_threshold": test[TARGET].to_numpy() >= p99,
    }.items():
        metrics = regression_metrics(
            test.loc[mask, TARGET].to_numpy(), test_prediction[mask]
        )
        outlier_rows.append(
            {
                "group": label,
                "records": int(mask.sum()),
                "threshold_p95_inr": p95,
                "threshold_p99_inr": p99,
                **metrics,
            }
        )
    outlier_analysis = pd.DataFrame(outlier_rows)

    predictions = test[
        ["case_id", "fsu_id", "medical_institution", TARGET]
    ].copy()
    predictions["predicted_medical_expenditure_inr"] = test_prediction
    predictions["residual_inr"] = test[TARGET].to_numpy() - test_prediction
    predictions["absolute_error_inr"] = np.abs(predictions["residual_inr"])
    predictions["prediction_interval_lower_inr"] = test_lower
    predictions["prediction_interval_upper_inr"] = test_upper
    largest_errors = predictions.nlargest(100, "absolute_error_inr")

    importance = _permutation_importance(
        point_model, X_test, test[TARGET]
    )
    cv_results.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    test_comparison.to_csv(
        EXPERIMENTS_DIR / "final_test_comparison.csv", index=False
    )
    subgroup_performance.to_csv(
        EXPERIMENTS_DIR / "subgroup_performance.csv", index=False
    )
    public_private.to_csv(
        EXPERIMENTS_DIR / "public_private_performance.csv", index=False
    )
    band_performance.to_csv(
        EXPERIMENTS_DIR / "expenditure_band_performance.csv", index=False
    )
    outlier_analysis.to_csv(
        EXPERIMENTS_DIR / "outlier_analysis.csv", index=False
    )
    predictions.to_csv(EXPERIMENTS_DIR / "test_predictions.csv", index=False)
    importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    largest_errors.to_csv(
        MODELS_DIR / "largest_prediction_errors.csv", index=False
    )

    joblib.dump(
        {
            "pipeline": point_model,
            "target_transform": "log1p",
            "features": INPUT_FEATURES,
            "model_features": MODEL_FEATURES,
            "feature_transform": "payment_safe_v2",
        },
        MODELS_DIR / "primary_model.joblib",
    )
    joblib.dump(
        {
            "lower": lower_model,
            "upper": upper_model,
            "features": INPUT_FEATURES,
            "model_features": MODEL_FEATURES,
            "feature_transform": "payment_safe_v2",
        },
        MODELS_DIR / "interval_models.joblib",
    )

    metadata: dict[str, Any] = {
        "component": "nss80_primary_research_model",
        "research_role": "primary",
        "model_version": "payment_safe_v2",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": "hist_gradient_boosting_log1p_payment_safe_tuned",
        "selection_reason": (
            "Selected for strict exclusion of payment-status target proxies and "
            "the best grouped-CV MAE among payment-safe candidates. The archived "
            "v1 model has lower apparent error but uses target-adjacent payment categories."
        ),
        "target_transform": "log1p",
        "selected_parameters": SELECTED_PARAMETERS,
        "input_features": INPUT_FEATURES,
        "features": INPUT_FEATURES,
        "model_features": MODEL_FEATURES,
        "numeric_features": MODEL_NUMERIC_FEATURES,
        "categorical_features": MODEL_CATEGORICAL_FEATURES,
        "allowed_values": allowed_values(),
        "feature_transform": {
            "name": "payment_safe_v2",
            "payment_status_detail_used": False,
            "free_medical_service_used": False,
            "service_fields": "collapsed to received yes/no",
            "ward_type": "collapsed to special versus standard_or_free",
        },
        "target": TARGET,
        "target_unit": "INR per inpatient case",
        "dataset": {
            **audit.report,
            "processed_path": str(PROCESSED_PATH.relative_to(BACKEND_ROOT)),
            "processed_sha256": sha256_file(PROCESSED_PATH),
            "catalog_url": "https://microdata.gov.in/NADA/index.php/catalog/290",
            "reference_id": "DDI-IND-NSO-HSCHealth80R-Jan2025-Dec2025",
            "target_distribution_inr": target_distribution,
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
            "test_used_for_selection": False,
        },
        "cross_validation": cv_results.to_dict(orient="records"),
        "test_metrics": final_metrics,
        "weighted_test_metrics": weighted_final_metrics,
        "dummy_test_metrics": dummy_metrics,
        "archived_v1_test_metrics": baseline_metrics,
        "prediction_interval": {
            "method": "split_conformalized_quantile_regression",
            "nominal_coverage": INTERVAL_COVERAGE,
            "lower_quantile": (1 - INTERVAL_COVERAGE) / 2,
            "upper_quantile": 1 - (1 - INTERVAL_COVERAGE) / 2,
            "calibration_rows": int(len(calibration)),
            "conformal_quantile_probability": adjustment_probability,
            "conformal_adjustment": adjustment,
            "test_empirical_coverage": interval_result["empirical_coverage"],
            "test_mean_width": interval_result["mean_width"],
            "test_median_width": interval_result["median_width"],
            "interpretation": (
                "Marginal uncertainty interval calibrated on unseen NSS FSUs; "
                "not an individual guarantee or quotation."
            ),
        },
        "outlier_handling": {
            "records_removed": 0,
            "winsorisation_applied": False,
            "robust_approach": (
                "log1p target modelling with all valid observations retained"
            ),
            "training_p95_inr": p95,
            "training_p99_inr": p99,
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
            "Episode-level estimator, not a pre-admission forecast.",
            "Observational self-reported expenditure with recall and measurement error.",
            "High-cost tail errors remain substantial despite log-target modelling.",
            "Research estimate only; not clinical advice, authorization, or a bill.",
        ],
    }
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (MODELS_DIR / "evaluation.json").write_text(
        json.dumps(
            {
                "test_comparison": test_comparison.to_dict(orient="records"),
                "public_private": public_private.to_dict(orient="records"),
                "expenditure_bands": band_performance.to_dict(orient="records"),
                "outliers": outlier_analysis.to_dict(orient="records"),
                "prediction_interval": metadata["prediction_interval"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _write_leakage_audit(REPORTS_DIR / "FEATURE_LEAKAGE_AUDIT.md")
    _svg_histogram(
        target.to_numpy(),
        PLOTS_DIR / "target_distribution_original.svg",
        "NSS 80 total medical expenditure",
        "Medical expenditure (INR)",
        log_transform=False,
    )
    _svg_histogram(
        target.to_numpy(),
        PLOTS_DIR / "target_distribution_log1p.svg",
        "NSS 80 log-transformed medical expenditure",
        "log1p(medical expenditure in INR)",
        log_transform=True,
    )
    _svg_scatter(
        test[TARGET].to_numpy(),
        test_prediction,
        PLOTS_DIR / "actual_vs_predicted.svg",
        "Payment-safe NSS 80: actual versus predicted",
        "Actual expenditure (INR)",
        "Predicted expenditure (INR)",
        diagonal=True,
    )
    residual = test[TARGET].to_numpy() - test_prediction
    _svg_scatter(
        test_prediction,
        residual,
        PLOTS_DIR / "residual_plot.svg",
        "Payment-safe NSS 80 residuals",
        "Predicted expenditure (INR)",
        "Residual: actual minus predicted (INR)",
    )
    _svg_histogram(
        residual,
        PLOTS_DIR / "residual_distribution.svg",
        "Payment-safe NSS 80 residual distribution",
        "Residual (INR)",
        log_transform=False,
    )
    _svg_importance(importance, PLOTS_DIR / "feature_importance.svg")
    _svg_public_private(
        public_private, PLOTS_DIR / "public_private_comparison.svg"
    )

    report = _build_fyp_report(
        metadata,
        test_comparison,
        public_private,
        band_performance,
        outlier_analysis,
        importance,
    )
    (REPORTS_DIR / "FYP_MODEL_IMPROVEMENT_REPORT.md").write_text(
        report, encoding="utf-8"
    )

    reloaded = joblib.load(MODELS_DIR / "primary_model.joblib")
    smoke = _predict_log_model(reloaded["pipeline"], X_test.iloc[:1])[0]
    if not np.isfinite(smoke):
        raise RuntimeError("Reloaded NSS model produced a non-finite prediction")

    print("\nLocked-test comparison:\n", test_comparison.to_string(index=False))
    print("\nPublic/private comparison:\n", public_private.to_string(index=False))
    print(f"\n90% interval coverage: {interval_result['empirical_coverage']:.2%}")
    print(f"Saved primary model: {MODELS_DIR / 'primary_model.joblib'}")
    return metadata


def _build_fyp_report(
    metadata: dict[str, Any],
    test_comparison: pd.DataFrame,
    public_private: pd.DataFrame,
    band_performance: pd.DataFrame,
    outlier_analysis: pd.DataFrame,
    importance: pd.DataFrame,
) -> str:
    weighted = metadata["weighted_test_metrics"]
    interval = metadata["prediction_interval"]
    distribution = metadata["dataset"]["target_distribution_inr"]
    rows = {
        row["model"]: row for row in test_comparison.to_dict(orient="records")
    }
    dummy = rows["dummy_median"]
    baseline = rows["original_deployed_v1_log_hgb"]
    selected = rows["selected_payment_safe_v2_log_hgb"]
    public_private_rows = "\n".join(
        f"| {row.group} | {int(row.records):,} | {row.mae:,.2f} | {row.rmse:,.2f} | {row.r2:.4f} | {row.median_absolute_error:,.2f} |"
        for row in public_private.itertuples(index=False)
    )
    band_rows = "\n".join(
        f"| {row.group} | {int(row.records):,} | {row.mae:,.2f} | {row.rmse:,.2f} | {row.median_absolute_error:,.2f} |"
        for row in band_performance.itertuples(index=False)
    )
    important_rows = "\n".join(
        f"- `{row.feature}`: test-sample MAE increased by INR {row.mae_increase_mean:,.0f} after permutation."
        for row in importance.head(8).itertuples(index=False)
    )
    top5 = outlier_analysis.loc[
        outlier_analysis["group"].eq("top_5_percent_threshold")
    ].iloc[0]
    top1 = outlier_analysis.loc[
        outlier_analysis["group"].eq("top_1_percent_threshold")
    ].iloc[0]
    screening = pd.read_csv(EXPERIMENTS_DIR / "model_screening.csv")
    screening_rows = "\n".join(
        f"| {row.model} | {row.target_transform} | {row.feature_set} | {row.mae:,.2f} | {row.rmse:,.2f} | {row.r2:.4f} | {row.median_absolute_error:,.2f} |"
        for row in screening.itertuples(index=False)
    )
    return f"""# NSS 80 Model Improvement and Evaluation Report

## Dataset

- Final cohort: {metadata['dataset']['modeling_rows']:,} inpatient cases from {metadata['dataset']['fsu_count']:,} first-stage units.
- Target: NSS Schedule 25.0 Block 7 item 12, total medical expenditure in INR per inpatient case.
- Input features: {len(metadata['input_features'])}; payment-safe fitted features: {len(metadata['model_features'])}.
- Raw NSS files were not modified, and no US Kaggle, Malaysian LIAM, or hospital-pricing data were merged.

## Target distribution

| Statistic | INR |
|---|---:|
| Minimum | {distribution['minimum']:,.2f} |
| Maximum | {distribution['maximum']:,.2f} |
| Mean | {distribution['mean']:,.2f} |
| Median | {distribution['median']:,.2f} |
| Standard deviation | {distribution['standard_deviation']:,.2f} |
| P25 | {distribution['p25']:,.2f} |
| P75 | {distribution['p75']:,.2f} |
| P90 | {distribution['p90']:,.2f} |
| P95 | {distribution['p95']:,.2f} |
| P99 | {distribution['p99']:,.2f} |

Skewness is {distribution['skewness']:.2f}; the target is highly right-skewed. Both original-scale and `log1p` histograms are saved under `Backend/reports/nss80/plots/`.

## Experimental setup

- Locked split: {metadata['split']['train_rows']:,} training, {metadata['split']['calibration_rows']:,} interval-calibration, and {metadata['split']['test_rows']:,} test records.
- All splits are grouped by FSU with zero overlap.
- Model screening and tuning used training/development data only; the test target was not used for selection.
- Candidate algorithms actually run: DummyRegressor, original/log/absolute HistGradientBoosting, Random Forest, Gradient Boosting, and CatBoost original/log.
- CatBoost 1.2.8 was evaluated because of its native categorical handling. XGBoost was not added because it would introduce a second external boosting dependency without a distinct methodological benefit after CatBoost failed to improve grouped-CV MAE.
- Tuning grids and fold-level summary statistics are preserved in `experiments_v2/`.

### Development-only screening comparison

| Model | Target | Features | MAE | RMSE | R-squared | Median AE |
|---|---|---|---:|---:|---:|---:|
{screening_rows}

## Target transformation and feature engineering

The original-target histogram model improved RMSE/R-squared but materially worsened MAE and median absolute error. `log1p` therefore remains justified because it performs better for the typical inpatient case while still being evaluated after `expm1` on the original INR scale.

Age bands, length-of-stay bands, institution/ward, surgery/stay, treatment/institution, and service-count features were tested. They did not improve grouped validation and were not forced into deployment.

## Leakage audit and final selection

The v1 model excluded all monetary expenditure components, totals, reimbursement, and income-loss fields. This audit additionally identified target-adjacent categorical payment proxies: free/partly-free/on-payment service statuses, free medical service, and free versus paying-general ward labels.

The selected v2 model collapses service fields to received yes/no, collapses ward to special versus standard/free, and excludes free-medical-service status from the estimator. This reduces apparent predictive performance but is substantially more defensible and robust against payment-status leakage. See `FEATURE_LEAKAGE_AUDIT.md`.

## Locked-test model comparison

| Model | MAE | RMSE | R-squared | Median AE | RMSLE |
|---|---:|---:|---:|---:|---:|
| Dummy median | {dummy['mae']:,.2f} | {dummy['rmse']:,.2f} | {dummy['r2']:.4f} | {dummy['median_absolute_error']:,.2f} | {dummy['rmsle']:.4f} |
| Archived v1 log HGB | {baseline['mae']:,.2f} | {baseline['rmse']:,.2f} | {baseline['r2']:.4f} | {baseline['median_absolute_error']:,.2f} | {baseline['rmsle']:.4f} |
| Selected payment-safe v2 log HGB | {selected['mae']:,.2f} | {selected['rmse']:,.2f} | {selected['r2']:.4f} | {selected['median_absolute_error']:,.2f} | {selected['rmsle']:.4f} |

The selected model improves MAE over DummyRegressor by {(1-selected['mae']/dummy['mae']):.1%}. The archived v1 result is reported transparently but is not selected because its stronger apparent result depends partly on payment-status proxies.

Survey-weighted final results are MAE INR {weighted['mae']:,.2f}, RMSE INR {weighted['rmse']:,.2f}, R-squared {weighted['r2']:.4f}, and median absolute error INR {weighted['median_absolute_error']:,.2f}.

## Public/private performance

| Hospital sector | Records | MAE | RMSE | R-squared | Median AE |
|---|---:|---:|---:|---:|---:|
{public_private_rows}

These are observed NSS patterns, not causal claims about provider ownership.

## Error by expenditure band

Bands use training-target thresholds, not hand-selected values.

| Actual-expenditure band | Records | MAE | RMSE | Median AE |
|---|---:|---:|---:|---:|
{band_rows}

## Outliers and residuals

No valid observation was deleted or winsorised. `log1p` target modelling is the robust experiment. The top-5%-threshold group contains {int(top5['records']):,} test cases with MAE INR {top5['mae']:,.2f}; the top-1%-threshold group contains {int(top1['records']):,} cases with MAE INR {top1['mae']:,.2f}. Residual magnitude increases sharply in the high-cost tail.

## Explainability

Permutation importance is associational, not causal:

{important_rows}

## Prediction interval

The recalibrated nominal 90% split-conformal quantile interval covered {interval['test_empirical_coverage']:.2%} of the locked test cases. Median interval width was INR {interval['test_median_width']:,.2f}. Coverage is marginal, not an individual guarantee.

## Limitations

- The model represents Indian NSS 80 data and outputs INR. It is not a Malaysian hospital-price predictor.
- It is an episode-level estimator, not a pre-admission forecast.
- Survey expenditure is self-reported and observational.
- High-cost tail errors remain substantial.
- Subgroup differences are descriptive and must not be interpreted causally.
"""
