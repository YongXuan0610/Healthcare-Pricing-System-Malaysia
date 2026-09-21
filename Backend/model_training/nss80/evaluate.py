"""Evaluation helpers for the NSS 80 primary research model."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    cutoff = 0.5 * float(np.sum(weights))
    return float(sorted_values[np.searchsorted(cumulative, cutoff, side="left")])


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(0.0, np.asarray(y_pred, dtype=float))
    absolute_errors = np.abs(y_true - y_pred)
    squared_log_errors = (np.log1p(y_true) - np.log1p(y_pred)) ** 2
    if sample_weight is None:
        median_absolute = float(np.median(absolute_errors))
        rmsle = float(np.mean(squared_log_errors) ** 0.5)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)
        median_absolute = _weighted_median(absolute_errors, sample_weight)
        rmsle = float(np.average(squared_log_errors, weights=sample_weight) ** 0.5)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)),
        "rmse": float(
            mean_squared_error(
                y_true,
                y_pred,
                sample_weight=sample_weight,
            )
            ** 0.5
        ),
        "r2": float(r2_score(y_true, y_pred, sample_weight=sample_weight)),
        "median_absolute_error": median_absolute,
        "rmsle": rmsle,
    }


def interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    covered = (y_true >= lower) & (y_true <= upper)
    return {
        "empirical_coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(upper - lower)),
        "median_width": float(np.median(upper - lower)),
    }
