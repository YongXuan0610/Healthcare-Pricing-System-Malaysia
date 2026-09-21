"""Grouped-CV comparison for the stricter NSS payment-safe feature set."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .data import FEATURES, TARGET, load_and_prepare_dataset
from .evaluate import regression_metrics
from .modeling import leakage_safe_features, make_leakage_safe_hgb_pipeline


BACKEND_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BACKEND_ROOT / "reports" / "nss80" / "experiments_v2"
RANDOM_STATE = 42
PARAMETER_GRID = [
    {
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.05,
        "max_iter": 500,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 30,
        "l2_regularization": 2.0,
    },
    {
        "learning_rate": 0.08,
        "max_iter": 500,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 40,
        "l2_regularization": 5.0,
    },
    {
        "learning_rate": 0.10,
        "max_iter": 400,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 20,
        "l2_regularization": 5.0,
    },
    {
        "learning_rate": 0.05,
        "max_iter": 600,
        "max_leaf_nodes": 127,
        "min_samples_leaf": 40,
        "l2_regularization": 10.0,
    },
]


def _weights(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    return array / float(array.mean())


def run() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_and_prepare_dataset(
        BACKEND_ROOT / "datasets" / "nss80" / "raw"
    ).modeling.reset_index(drop=True)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    development_index, _ = next(
        outer.split(frame[FEATURES], frame[TARGET], frame["fsu_id"])
    )
    development = frame.iloc[development_index].reset_index(drop=True)
    calibration_split = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=RANDOM_STATE + 1
    )
    selection_position, _ = next(
        calibration_split.split(
            development[FEATURES], development[TARGET], development["fsu_id"]
        )
    )
    selection = development.iloc[selection_position].reset_index(drop=True)
    X = leakage_safe_features(selection[FEATURES])
    y = selection[TARGET]
    groups = selection["fsu_id"]

    rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=3)
    for configuration, parameters in enumerate(PARAMETER_GRID, start=1):
        started = time.perf_counter()
        folds: list[dict[str, float]] = []
        for fold, (train_index, valid_index) in enumerate(
            splitter.split(X, y, groups), start=1
        ):
            model = make_leakage_safe_hgb_pipeline(**parameters)
            model.fit(
                X.iloc[train_index],
                np.log1p(y.iloc[train_index]),
                model__sample_weight=_weights(
                    selection.iloc[train_index]["survey_multiplier"]
                ),
            )
            prediction = np.maximum(
                0.0, np.expm1(model.predict(X.iloc[valid_index]))
            )
            metrics = regression_metrics(y.iloc[valid_index].to_numpy(), prediction)
            weighted = regression_metrics(
                y.iloc[valid_index].to_numpy(),
                prediction,
                sample_weight=_weights(
                    selection.iloc[valid_index]["survey_multiplier"]
                ),
            )
            folds.append(
                {
                    **metrics,
                    **{f"weighted_{key}": value for key, value in weighted.items()},
                }
            )
            print(
                f"payment-safe config {configuration} fold {fold}: "
                f"MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, "
                f"R2={metrics['r2']:.4f}",
                flush=True,
            )
        row: dict[str, Any] = {
            "model": f"payment_safe_hgb_log1p_{configuration}",
            "parameters": json.dumps(parameters, sort_keys=True),
            "cv_folds": 3,
            "fit_seconds": time.perf_counter() - started,
        }
        for metric in folds[0]:
            values = [fold[metric] for fold in folds]
            row[f"cv_{metric}_mean"] = float(np.mean(values))
            row[f"cv_{metric}_std"] = float(np.std(values, ddof=1))
        rows.append(row)
        pd.DataFrame(rows).to_csv(
            OUTPUT_DIR / "leakage_safe_tuning.partial.csv", index=False
        )
    results = pd.DataFrame(rows).sort_values("cv_mae_mean", ignore_index=True)
    results.to_csv(OUTPUT_DIR / "leakage_safe_tuning.csv", index=False)
    print("\nPayment-safe tuning results:\n", results.to_string(index=False))
    return results


if __name__ == "__main__":
    run()
