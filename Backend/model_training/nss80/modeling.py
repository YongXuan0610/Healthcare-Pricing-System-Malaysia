"""Leakage-controlled feature transformation and estimators for NSS 80."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .data import FEATURES as INPUT_FEATURES


MODEL_NUMERIC_FEATURES = ["age_years", "length_of_stay_days"]
MODEL_CATEGORICAL_FEATURES = [
    "gender",
    "sector",
    "state",
    "ailment",
    "treatment_system",
    "medical_institution",
    "ward_class",
    "surgery_received",
    "medicine_received",
    "imaging_received",
    "other_diagnostics_received",
]
MODEL_FEATURES = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES


def leakage_safe_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove payment-status detail while retaining service-utilisation signals.

    The API input contract remains backward compatible. Payment distinctions in
    the four service fields and ``free_medical_service`` are deliberately not
    presented to the fitted estimator.
    """

    missing = sorted(set(INPUT_FEATURES).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing NSS input feature(s): {missing}")

    output = frame[
        [
            "age_years",
            "length_of_stay_days",
            "gender",
            "sector",
            "state",
            "ailment",
            "treatment_system",
            "medical_institution",
        ]
    ].copy()
    output["ward_class"] = np.where(
        frame["ward_type"].eq("paying_special"), "special", "standard_or_free"
    )
    for source in ["surgery", "medicine", "imaging", "other_diagnostics"]:
        output[f"{source}_received"] = np.where(
            frame[source].eq("not_received"), "no", "yes"
        )
    return output[MODEL_FEATURES]


def make_leakage_safe_hgb_pipeline(**parameters: Any) -> Pipeline:
    categorical = Pipeline(
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
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median"),
                MODEL_NUMERIC_FEATURES,
            ),
            ("categorical", categorical, MODEL_CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    defaults: dict[str, Any] = {
        "loss": "squared_error",
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.10,
        "n_iter_no_change": 25,
        "random_state": 42,
    }
    defaults.update(parameters)
    model = HistGradientBoostingRegressor(
        categorical_features=[False] * len(MODEL_NUMERIC_FEATURES)
        + [True] * len(MODEL_CATEGORICAL_FEATURES),
        **defaults,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])
