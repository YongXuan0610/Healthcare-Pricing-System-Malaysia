"""Structured leakage-safe NSS 80 model-improvement experiments.

The runner deliberately reuses the corrected v3 preparation logic and the
saved v3 train/test episode IDs.  Model/feature selection uses only the
person-grouped inner validation split.  The locked outer test set is evaluated
only after all candidate and architecture choices have been frozen.

Run from ``Backend``::

    .\.venv\Scripts\python.exe -m model_training.nss80.safe_improvement_v4

All outputs are written to new ``safe_improvement_v4`` directories.  The
production ``models/nss80/primary_model.joblib`` is hash-checked and is never
written by this module.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

from .codebook import AILMENT
from .evaluate import regression_metrics
from .retrain_v3 import (
    BINARY_FEATURES as V3_BINARY_FEATURES,
    HGB_PARAMETERS,
    KNOWN_LEAKAGE,
    NOMINAL_FEATURES as V3_NOMINAL_FEATURES,
    NUMERIC_FEATURES as V3_NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    _fit_candidate as fit_v3_candidate,
    _hgb_factory,
    _metric_payload,
    _predict as predict_v3_candidate,
    _sha256,
    ModelCandidate,
    prepare_hospitalisation_data,
)
from .schema_v3 import HOUSEHOLD_KEY, decode_yes_no_binary


BACKEND_ROOT = Path(__file__).resolve().parents[2]
V3_REPORT_DIR = BACKEND_ROOT / "reports" / "nss80" / "retraining_v3"
REPORT_DIR = BACKEND_ROOT / "reports" / "nss80" / "safe_improvement_v4"
MODEL_DIR = BACKEND_ROOT / "models" / "nss80" / "safe_improvement_v4"
PRODUCTION_MODEL = BACKEND_ROOT / "models" / "nss80" / "primary_model.joblib"
RAW_DIR = BACKEND_ROOT / "datasets" / "nss80" / "raw"

BASE_FEATURES = [
    "household_size",
    "household_usual_consumer_expenditure_rs",
    "age_years",
    "number_of_hospitalisations",
    "length_of_stay_days",
    "gender",
    "chronic_ailment",
    "health_financing_or_insurance_coverage",
    "surgery",
    "medicine",
    "xray_ecg_eeg_scan",
    "other_diagnostic_tests",
    "sector",
    "state",
    "ailment_nature",
    "hospitalisation_treatment_nature",
    "medical_institution_type",
    "ward_type",
    "household_type",
    "medical_insurance_premium_rs",
    "relation_to_household_head",
    "marital_status",
    "highest_education_level",
    "reason_not_using_government_public_hospital",
    "treated_on_medical_advice_before_hospitalisation",
    "pre_hospitalisation_treatment_nature",
    "pre_hospitalisation_level_of_care",
    "pre_hospitalisation_treatment_duration_days",
]

GEOGRAPHY_FEATURES = [
    "nss_region",
    "district",
    "place_of_hospitalisation",
    "treatment_state_code",
]
HEALTH_HISTORY_FEATURES = [
    "community_communicable_disease_outbreak",
    "pregnant",
    "communicable_disease",
    "other_ailment_last_15_days",
    "other_ailment_previous_day",
]
BASIC_ENGINEERED_FEATURES = [
    "age_group",
    "length_of_stay_group",
    "household_expenditure_per_person",
    "hospitalisations_per_household_member",
    "diagnostic_use_count",
    "service_use_count",
    "ailment_group",
]
INTERACTION_FEATURES = [
    "ailment_x_institution",
    "ailment_x_surgery",
    "ailment_x_ward",
    "ailment_x_length_of_stay_group",
    "institution_x_ward",
    "institution_x_state",
    "surgery_x_length_of_stay_group",
]
L5_FEATURES = [
    "recent_ailment_spell_count",
    "any_recent_chronic_ailment",
    "max_recent_ailment_duration_days",
    "mean_recent_ailment_duration_days",
    "any_recent_hospitalisation",
    "any_recent_medical_advice",
    "recent_treatment_type_count",
    "recent_level_of_care_diversity",
    "recent_private_care_indicator",
    "recent_public_care_indicator",
]

DYNAMIC_NUMERIC_FEATURES = {
    "household_expenditure_per_person",
    "hospitalisations_per_household_member",
    "diagnostic_use_count",
    "service_use_count",
    "recent_ailment_spell_count",
    "max_recent_ailment_duration_days",
    "mean_recent_ailment_duration_days",
    "recent_treatment_type_count",
    "recent_level_of_care_diversity",
}
DYNAMIC_BINARY_FEATURES = {
    "community_communicable_disease_outbreak",
    "any_recent_chronic_ailment",
    "any_recent_hospitalisation",
    "any_recent_medical_advice",
    "recent_private_care_indicator",
    "recent_public_care_indicator",
}
DYNAMIC_NOMINAL_FEATURES = set(BASIC_ENGINEERED_FEATURES + INTERACTION_FEATURES) - DYNAMIC_NUMERIC_FEATURES

NUMERIC_FEATURES = set(V3_NUMERIC_FEATURES) | DYNAMIC_NUMERIC_FEATURES
BINARY_FEATURES = set(V3_BINARY_FEATURES) | DYNAMIC_BINARY_FEATURES
NOMINAL_FEATURES = set(V3_NOMINAL_FEATURES) | DYNAMIC_NOMINAL_FEATURES

PROHIBITED_FEATURES = set(KNOWN_LEAKAGE) | {
    "medical_service_free_fully_or_partly",
    "package_component_rs",
    "doctor_surgeon_fee_rs",
    "medicines_rs",
    "diagnostic_tests_rs",
    "bed_charges_rs",
    "other_medical_expenses_rs",
    "patient_transport_rs",
    "other_non_medical_household_expenses_rs",
    "total_expenditure_rs",
    "insurance_or_employer_reimbursement_rs",
    "major_source_of_finance",
    "household_income_loss_due_to_hospitalisation_rs",
}
L5_ALLOWED_RAW_COLUMNS = {
    *HOUSEHOLD_KEY,
    "b8i1",
    "b8i2",
    "b8i6",
    "b8i8",
    "b8i9",
    "b8i10",
    "b8i11",
    "b8i12",
}
L5_PROHIBITED_PAYMENT_RAW_COLUMNS = {
    "b9i5",  # free/partly-free payment status
    "b9i11",  # doctor/surgeon fee
    "b9i12",  # Ayush medicine expenditure
    "b9i13",  # non-Ayush medicine expenditure
    "b9i14",  # diagnostic expenditure
    "b9i15",  # other medical expenditure
    "b9i16",  # total medical expenditure
    "b9i17",  # transport expenditure
    "b9i18",  # other household expenditure
    "b9i19",  # total expenditure
    "b9i20",  # reimbursement
    "b9i21",  # source of finance
    "b9i24",  # household income loss
}

AILMENT_GROUP_CODES = {
    "Infection": {f"{code:02d}" for code in range(1, 13)},
    "Cancer": {"13"},
    "Blood diseases": {"14", "15", "16"},
    "Endocrine / metabolic / nutritional": {"17", "18", "19", "20"},
    "Psychiatric / neurological": {f"{code:02d}" for code in range(21, 28)},
    "Eye": {f"{code:02d}" for code in range(28, 33)},
    "Ear": {"33", "34"},
    "Cardiovascular": {"35", "36"},
    "Respiratory": {"37", "38", "39"},
    "Gastrointestinal": {"40", "41", "42", "43"},
    "Skin": {"44"},
    "Musculoskeletal": {"45", "46"},
    "Genito-urinary": {"47", "48", "49", "60"},
    "Obstetric / childbirth": {"50", "51", "52", "87", "88", "89"},
    "Injuries": {"53", "54", "55", "56", "57", "58", "59"},
    "Other / unspecified": {"61", "62"},
}
AILMENT_GROUP_BY_LABEL = {
    AILMENT[code]: group
    for group, codes in AILMENT_GROUP_CODES.items()
    for code in codes
}


def ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def weights(frame: pd.DataFrame) -> np.ndarray:
    values = frame["survey_multiplier"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Survey multipliers must be finite and positive")
    return values / values.mean()


def feature_types(features: list[str]) -> tuple[list[str], list[str], list[str]]:
    numeric = [feature for feature in features if feature in NUMERIC_FEATURES]
    binary = [feature for feature in features if feature in BINARY_FEATURES]
    nominal = [feature for feature in features if feature in NOMINAL_FEATURES]
    classified = set(numeric) | set(binary) | set(nominal)
    undeclared = sorted(set(features) - classified)
    if undeclared:
        raise ValueError(f"Feature type not declared: {undeclared}")
    overlap = (set(numeric) & set(binary)) | (set(numeric) & set(nominal)) | (set(binary) & set(nominal))
    if overlap:
        raise ValueError(f"Feature type overlap: {sorted(overlap)}")
    return numeric, binary, nominal


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    numeric, binary, nominal = feature_types(features)
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if binary:
        transformers.append(("binary", SimpleImputer(strategy="most_frequent"), binary))
    if nominal:
        transformers.append(
            (
                "nominal",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing or not applicable")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
                    ]
                ),
                nominal,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    target_transform: str
    parameters: dict[str, Any]
    native_categorical: bool = False


class SafeRegressor:
    """Serializable leakage-safe estimator with train-only target transforms."""

    def __init__(self, spec: CandidateSpec, features: list[str]) -> None:
        self.spec = spec
        self.features = list(features)
        self.numeric, self.binary, self.nominal = feature_types(self.features)
        prohibited = sorted(set(self.features) & PROHIBITED_FEATURES)
        if prohibited:
            raise ValueError(f"Prohibited leakage features requested: {prohibited}")

    def _target_fit_transform(self, y: np.ndarray) -> np.ndarray:
        if self.spec.target_transform == "identity":
            self.target_transformer_ = None
            return y
        if self.spec.target_transform == "log1p":
            self.target_transformer_ = None
            return np.log1p(y)
        if self.spec.target_transform == "yeo_johnson":
            self.target_transformer_ = PowerTransformer(method="yeo-johnson", standardize=True)
            return self.target_transformer_.fit_transform(y.reshape(-1, 1)).ravel()
        raise ValueError(f"Unknown target transform {self.spec.target_transform}")

    def _target_inverse(self, values: np.ndarray) -> np.ndarray:
        if self.spec.target_transform == "identity":
            output = values
        elif self.spec.target_transform == "log1p":
            output = np.expm1(np.clip(values, -30.0, 30.0))
        elif self.spec.target_transform == "yeo_johnson":
            output = self.target_transformer_.inverse_transform(values.reshape(-1, 1)).ravel()
        else:
            raise ValueError(f"Unknown target transform {self.spec.target_transform}")
        return np.maximum(0.0, np.asarray(output, dtype=float))

    def _make_sklearn_model(self) -> Any:
        params = dict(self.spec.parameters)
        if self.spec.family == "hist_gradient_boosting":
            return HistGradientBoostingRegressor(**params)
        if self.spec.family == "extra_trees":
            return ExtraTreesRegressor(**params)
        if self.spec.family == "random_forest":
            return RandomForestRegressor(**params)
        if self.spec.family == "gradient_boosting":
            return GradientBoostingRegressor(**params)
        if self.spec.family == "xgboost":
            from xgboost import XGBRegressor

            return XGBRegressor(**params)
        raise ValueError(f"Unsupported sklearn model family {self.spec.family}")

    def _prepare_catboost_frame(self, frame: pd.DataFrame, *, fitting: bool) -> pd.DataFrame:
        output = frame[self.features].copy()
        if fitting:
            self.numeric_medians_ = {
                column: float(pd.to_numeric(output[column], errors="coerce").median())
                for column in self.numeric
            }
            self.binary_modes_ = {}
            for column in self.binary:
                values = pd.to_numeric(output[column], errors="coerce")
                mode = values.mode(dropna=True)
                self.binary_modes_[column] = float(mode.iloc[0]) if not mode.empty else 0.0
        for column in self.numeric:
            output[column] = pd.to_numeric(output[column], errors="coerce").fillna(self.numeric_medians_[column]).astype(float)
        for column in self.binary:
            output[column] = pd.to_numeric(output[column], errors="coerce").fillna(self.binary_modes_[column]).astype(float)
        for column in self.nominal:
            output[column] = output[column].astype("string").fillna("Missing or not applicable").astype(str)
        return output

    def fit(self, frame: pd.DataFrame) -> "SafeRegressor":
        y = frame[TARGET].to_numpy(dtype=float)
        transformed_y = self._target_fit_transform(y)
        sample_weight = weights(frame)
        if self.spec.native_categorical:
            from catboost import CatBoostRegressor

            x = self._prepare_catboost_frame(frame, fitting=True)
            self.preprocessor_ = None
            self.model_ = CatBoostRegressor(**self.spec.parameters)
            self.model_.fit(x, transformed_y, cat_features=self.nominal, sample_weight=sample_weight)
            self.encoded_feature_names_ = list(self.features)
        else:
            self.preprocessor_ = make_preprocessor(self.features)
            x = self.preprocessor_.fit_transform(frame[self.features])
            self.model_ = self._make_sklearn_model()
            self.model_.fit(x, transformed_y, sample_weight=sample_weight)
            self.encoded_feature_names_ = [str(value) for value in self.preprocessor_.get_feature_names_out()]
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.spec.native_categorical:
            x = self._prepare_catboost_frame(frame, fitting=False)
        else:
            x = self.preprocessor_.transform(frame[self.features])
        return self._target_inverse(np.asarray(self.model_.predict(x), dtype=float))

    @property
    def encoded_feature_count(self) -> int:
        return len(self.encoded_feature_names_)


# ``python -m`` executes this file as ``__main__``.  Give serializable classes
# their stable import path so artifacts reload correctly in services/tests.
_STABLE_MODULE_NAME = "model_training.nss80.safe_improvement_v4"
if _STABLE_MODULE_NAME not in sys.modules:
    sys.modules[_STABLE_MODULE_NAME] = sys.modules[__name__]
CandidateSpec.__module__ = _STABLE_MODULE_NAME
SafeRegressor.__module__ = _STABLE_MODULE_NAME


def metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    return regression_metrics(frame[TARGET].to_numpy(dtype=float), prediction)


def weighted_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    return regression_metrics(frame[TARGET].to_numpy(dtype=float), prediction, sample_weight=weights(frame))


def evaluate_spec(
    spec: CandidateSpec,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> tuple[dict[str, Any], SafeRegressor]:
    started = time.perf_counter()
    model = SafeRegressor(spec, features).fit(train)
    prediction = model.predict(validation)
    row = {
        "experiment": spec.name,
        "family": spec.family,
        "target_transform": spec.target_transform,
        "features": "v3 safe" if features == BASE_FEATURES else "custom safe",
        "original_feature_count": len(features),
        "encoded_feature_count": model.encoded_feature_count,
        "fit_seconds": time.perf_counter() - started,
        **metrics(validation, prediction),
        "parameters": json.dumps(spec.parameters, sort_keys=True),
        "evaluation_split": "person-grouped inner validation",
    }
    return row, model


def _base_model_parameters(family: str) -> dict[str, Any]:
    if family == "hist_gradient_boosting":
        return {
            "loss": "squared_error",
            "learning_rate": 0.05,
            "max_iter": 600,
            "max_leaf_nodes": 127,
            "max_depth": None,
            "min_samples_leaf": 40,
            "l2_regularization": 10.0,
            "early_stopping": True,
            "validation_fraction": 0.10,
            "n_iter_no_change": 25,
            "random_state": RANDOM_STATE,
        }
    raise KeyError(family)


def candidate_specs() -> list[CandidateSpec]:
    common_hgb = {
        "loss": "squared_error",
        "early_stopping": True,
        "validation_fraction": 0.10,
        "n_iter_no_change": 25,
        "random_state": RANDOM_STATE,
    }
    hgb_configs = [
        {"learning_rate": 0.05, "max_iter": 600, "max_leaf_nodes": 127, "max_depth": None, "min_samples_leaf": 40, "l2_regularization": 10.0},
        {"learning_rate": 0.04, "max_iter": 800, "max_leaf_nodes": 63, "max_depth": None, "min_samples_leaf": 30, "l2_regularization": 5.0},
        {"learning_rate": 0.06, "max_iter": 500, "max_leaf_nodes": 31, "max_depth": None, "min_samples_leaf": 25, "l2_regularization": 3.0},
        {"learning_rate": 0.04, "max_iter": 700, "max_leaf_nodes": 127, "max_depth": 12, "min_samples_leaf": 35, "l2_regularization": 15.0},
        {"learning_rate": 0.03, "max_iter": 900, "max_leaf_nodes": 255, "max_depth": None, "min_samples_leaf": 50, "l2_regularization": 20.0},
    ]
    specs: list[CandidateSpec] = []
    for index, config in enumerate(hgb_configs):
        specs.append(CandidateSpec(f"hgb_identity_{index + 1}", "hist_gradient_boosting", "identity", {**common_hgb, **config}))
    for index in range(2):
        specs.append(CandidateSpec(f"hgb_log1p_{index + 1}", "hist_gradient_boosting", "log1p", {**common_hgb, **hgb_configs[index]}))
    specs.append(CandidateSpec("hgb_yeo_johnson_1", "hist_gradient_boosting", "yeo_johnson", {**common_hgb, **hgb_configs[0]}))

    tree_common = {"n_jobs": -1, "random_state": RANDOM_STATE}
    extra_configs = [
        {"n_estimators": 240, "max_depth": None, "min_samples_split": 8, "min_samples_leaf": 3, "max_features": 1.0, "bootstrap": False},
        {"n_estimators": 300, "max_depth": 30, "min_samples_split": 10, "min_samples_leaf": 4, "max_features": 0.8, "bootstrap": False},
        {"n_estimators": 300, "max_depth": 24, "min_samples_split": 12, "min_samples_leaf": 6, "max_features": 0.7, "bootstrap": False},
        {"n_estimators": 260, "max_depth": 32, "min_samples_split": 8, "min_samples_leaf": 4, "max_features": "sqrt", "bootstrap": False},
        {"n_estimators": 260, "max_depth": 28, "min_samples_split": 10, "min_samples_leaf": 5, "max_features": 0.8, "bootstrap": True},
    ]
    for index, config in enumerate(extra_configs):
        specs.append(CandidateSpec(f"extra_trees_identity_{index + 1}", "extra_trees", "identity", {**tree_common, **config}))
    for index in range(2):
        specs.append(CandidateSpec(f"extra_trees_log1p_{index + 1}", "extra_trees", "log1p", {**tree_common, **extra_configs[index]}))

    rf_configs = [
        {"n_estimators": 220, "max_depth": 24, "min_samples_split": 10, "min_samples_leaf": 5, "max_features": 0.8, "bootstrap": True},
        {"n_estimators": 260, "max_depth": 30, "min_samples_split": 8, "min_samples_leaf": 4, "max_features": 1.0, "bootstrap": True},
        {"n_estimators": 260, "max_depth": 20, "min_samples_split": 12, "min_samples_leaf": 6, "max_features": 0.7, "bootstrap": True},
        {"n_estimators": 240, "max_depth": None, "min_samples_split": 14, "min_samples_leaf": 7, "max_features": "sqrt", "bootstrap": True},
    ]
    for index, config in enumerate(rf_configs):
        specs.append(CandidateSpec(f"random_forest_identity_{index + 1}", "random_forest", "identity", {**tree_common, **config}))
    specs.append(CandidateSpec("random_forest_log1p_1", "random_forest", "log1p", {**tree_common, **rf_configs[0]}))

    gbr_configs = [
        {"loss": "huber", "learning_rate": 0.05, "n_estimators": 160, "max_depth": 3, "min_samples_split": 10, "min_samples_leaf": 25, "max_features": None, "subsample": 0.8, "random_state": RANDOM_STATE},
        {"loss": "squared_error", "learning_rate": 0.04, "n_estimators": 220, "max_depth": 3, "min_samples_split": 12, "min_samples_leaf": 30, "max_features": 0.8, "subsample": 0.85, "random_state": RANDOM_STATE},
        {"loss": "huber", "learning_rate": 0.03, "n_estimators": 260, "max_depth": 4, "min_samples_split": 14, "min_samples_leaf": 35, "max_features": 0.8, "subsample": 0.8, "random_state": RANDOM_STATE},
    ]
    for index, config in enumerate(gbr_configs):
        specs.append(CandidateSpec(f"gradient_boosting_identity_{index + 1}", "gradient_boosting", "identity", config))
    specs.append(CandidateSpec("gradient_boosting_log1p_1", "gradient_boosting", "log1p", gbr_configs[0]))

    cat_common = {
        "loss_function": "RMSE",
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
        "bootstrap_type": "Bayesian",
    }
    cat_configs = [
        {"iterations": 500, "depth": 7, "learning_rate": 0.06, "l2_leaf_reg": 5.0, "random_strength": 1.0, "bagging_temperature": 1.0},
        {"iterations": 700, "depth": 8, "learning_rate": 0.045, "l2_leaf_reg": 8.0, "random_strength": 0.5, "bagging_temperature": 0.5},
        {"iterations": 850, "depth": 9, "learning_rate": 0.035, "l2_leaf_reg": 10.0, "random_strength": 1.0, "bagging_temperature": 1.0},
        {"iterations": 650, "depth": 10, "learning_rate": 0.04, "l2_leaf_reg": 12.0, "random_strength": 0.25, "bagging_temperature": 0.75},
        {"iterations": 900, "depth": 6, "learning_rate": 0.04, "l2_leaf_reg": 6.0, "random_strength": 1.5, "bagging_temperature": 1.5},
        {"iterations": 600, "depth": 8, "learning_rate": 0.055, "l2_leaf_reg": 15.0, "random_strength": 0.75, "bagging_temperature": 0.25},
    ]
    for index, config in enumerate(cat_configs):
        specs.append(CandidateSpec(f"catboost_identity_{index + 1}", "catboost", "identity", {**cat_common, **config}, True))
    for index in range(2):
        specs.append(CandidateSpec(f"catboost_log1p_{index + 1}", "catboost", "log1p", {**cat_common, **cat_configs[index]}, True))

    xgb_common = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "verbosity": 0,
    }
    xgb_configs = [
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.04, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 8, "reg_alpha": 0.0, "reg_lambda": 5.0},
        {"n_estimators": 700, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 10, "reg_alpha": 0.2, "reg_lambda": 8.0},
        {"n_estimators": 450, "max_depth": 8, "learning_rate": 0.045, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 12, "reg_alpha": 0.5, "reg_lambda": 10.0},
        {"n_estimators": 800, "max_depth": 4, "learning_rate": 0.025, "subsample": 0.9, "colsample_bytree": 0.75, "min_child_weight": 6, "reg_alpha": 0.1, "reg_lambda": 6.0},
        {"n_estimators": 550, "max_depth": 7, "learning_rate": 0.035, "subsample": 0.75, "colsample_bytree": 1.0, "min_child_weight": 15, "reg_alpha": 1.0, "reg_lambda": 12.0},
        {"n_estimators": 650, "max_depth": 6, "learning_rate": 0.03, "subsample": 1.0, "colsample_bytree": 0.8, "min_child_weight": 20, "reg_alpha": 0.5, "reg_lambda": 15.0},
    ]
    for index, config in enumerate(xgb_configs):
        specs.append(CandidateSpec(f"xgboost_identity_{index + 1}", "xgboost", "identity", {**xgb_common, **config}))
    for index in range(2):
        specs.append(CandidateSpec(f"xgboost_log1p_{index + 1}", "xgboost", "log1p", {**xgb_common, **xgb_configs[index]}))
    return specs


def _entity_id(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame[columns].fillna("<missing>").astype(str).agg("|".join, axis=1)


def add_community_outbreak(frame: pd.DataFrame) -> pd.DataFrame:
    l1 = pd.read_csv(RAW_DIR / "hhscsL1.csv", dtype=str, usecols=[*HOUSEHOLD_KEY, "b5i5"], low_memory=False)
    if l1.duplicated(HOUSEHOLD_KEY).any():
        raise ValueError("L1 has duplicate household keys during v4 feature preparation")
    l1["household_id"] = _entity_id(l1, HOUSEHOLD_KEY)
    l1["community_communicable_disease_outbreak"] = decode_yes_no_binary(l1["b5i5"], "b5i5").astype(float)
    before = len(frame)
    output = frame.merge(
        l1[["household_id", "community_communicable_disease_outbreak"]],
        on="household_id",
        how="left",
        validate="many_to_one",
    )
    if len(output) != before:
        raise RuntimeError("L1 feature join multiplied hospitalisation rows")
    return output


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["ailment_group"] = output["ailment_nature"].map(AILMENT_GROUP_BY_LABEL).fillna("Other / unspecified")
    output["age_group"] = pd.cut(
        output["age_years"],
        bins=[-np.inf, 4, 14, 24, 44, 64, np.inf],
        labels=["0-4", "5-14", "15-24", "25-44", "45-64", "65+"],
    ).astype("string")
    output["length_of_stay_group"] = pd.cut(
        output["length_of_stay_days"],
        bins=[-np.inf, 1, 3, 7, 14, np.inf],
        labels=["0-1", "2-3", "4-7", "8-14", "15+"],
    ).astype("string")
    household_size = output["household_size"].clip(lower=1.0)
    output["household_expenditure_per_person"] = output["household_usual_consumer_expenditure_rs"] / household_size
    output["hospitalisations_per_household_member"] = output["number_of_hospitalisations"] / household_size
    output["diagnostic_use_count"] = output[["xray_ecg_eeg_scan", "other_diagnostic_tests"]].sum(axis=1)
    output["service_use_count"] = output[["surgery", "medicine", "xray_ecg_eeg_scan", "other_diagnostic_tests"]].sum(axis=1)

    def interaction(left: str, right: str) -> pd.Series:
        return output[left].astype("string").fillna("Missing") + " | " + output[right].astype("string").fillna("Missing")

    output["ailment_x_institution"] = interaction("ailment_nature", "medical_institution_type")
    output["ailment_x_surgery"] = interaction("ailment_nature", "surgery")
    output["ailment_x_ward"] = interaction("ailment_nature", "ward_type")
    output["ailment_x_length_of_stay_group"] = interaction("ailment_nature", "length_of_stay_group")
    output["institution_x_ward"] = interaction("medical_institution_type", "ward_type")
    output["institution_x_state"] = interaction("medical_institution_type", "state")
    output["surgery_x_length_of_stay_group"] = interaction("surgery", "length_of_stay_group")
    return output


def add_l5_aggregates(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    l5_path = RAW_DIR / "hhscsL5.csv"
    l5 = pd.read_csv(l5_path, dtype=str, usecols=sorted(L5_ALLOWED_RAW_COLUMNS), low_memory=False)
    unexpected = set(l5.columns) - L5_ALLOWED_RAW_COLUMNS
    if unexpected:
        raise RuntimeError(f"Unexpected L5 columns entered safe aggregation: {sorted(unexpected)}")
    l5["person_group_id"] = _entity_id(l5, [*HOUSEHOLD_KEY, "b8i2"])
    l5["duration"] = pd.to_numeric(l5["b8i8"], errors="coerce")
    l5["chronic"] = l5["b8i6"].astype("string").str.strip().eq("1").fillna(False).astype(int)
    l5["hospitalised"] = l5["b8i10"].astype("string").str.strip().eq("1").fillna(False).astype(int)
    l5["medical_advice"] = l5["b8i11"].astype("string").str.strip().eq("1").fillna(False).astype(int)
    care = l5["b8i12"].astype("string").str.strip()
    l5["private_care"] = care.isin(["3", "6"]).astype(int)
    l5["public_care"] = care.isin(["1", "4", "5"]).astype(int)
    l5["treatment_type"] = l5["b8i9"].astype("string").str.strip().replace("", pd.NA)
    l5["care_type"] = care.replace("", pd.NA)

    aggregates = l5.groupby("person_group_id", sort=False).agg(
        recent_ailment_spell_count=("b8i1", "size"),
        any_recent_chronic_ailment=("chronic", "max"),
        max_recent_ailment_duration_days=("duration", "max"),
        mean_recent_ailment_duration_days=("duration", "mean"),
        any_recent_hospitalisation=("hospitalised", "max"),
        any_recent_medical_advice=("medical_advice", "max"),
        recent_treatment_type_count=("treatment_type", "nunique"),
        recent_level_of_care_diversity=("care_type", "nunique"),
        recent_private_care_indicator=("private_care", "max"),
        recent_public_care_indicator=("public_care", "max"),
    ).reset_index()
    if aggregates["person_group_id"].duplicated().any():
        raise RuntimeError("L5 person aggregation is not unique")
    before = len(frame)
    output = frame.merge(aggregates, on="person_group_id", how="left", validate="many_to_one")
    if len(output) != before:
        raise RuntimeError("L5 person aggregate join multiplied hospitalisation rows")
    matched = output["recent_ailment_spell_count"].notna()
    for feature in L5_FEATURES:
        output[feature] = output[feature].fillna(0.0).astype(float)
    modelling_people = set(frame["person_group_id"])
    aggregate_people = set(aggregates["person_group_id"])
    matched_people = modelling_people & aggregate_people
    summary = {
        "l5_raw_spell_rows": int(len(l5)),
        "l5_distinct_people": int(len(aggregates)),
        "modelling_people_with_l5_data": int(len(matched_people)),
        "modelling_people_without_l5_data": int(len(modelling_people - aggregate_people)),
        "hospitalisation_episodes_with_l5_data": int(matched.sum()),
        "hospitalisation_episodes_without_l5_data": int((~matched).sum()),
        "rows_before_join": int(before),
        "rows_after_join": int(len(output)),
        "row_multiplication": int(len(output) - before),
        "aggregation_level": "person_group_id",
        "allowed_raw_columns": sorted(L5_ALLOWED_RAW_COLUMNS),
        "omitted_raw_columns": sorted(set(pd.read_csv(l5_path, nrows=0).columns) - L5_ALLOWED_RAW_COLUMNS),
        "prohibited_payment_or_expenditure_columns_confirmed_absent": sorted(L5_PROHIBITED_PAYMENT_RAW_COLUMNS),
        "missing_handling": "People without an L5 spell receive zero for every count, duration, diversity, and indicator aggregate.",
        "aggregation_rules": {
            "recent_ailment_spell_count": "number of L5 spell rows",
            "any_recent_chronic_ailment": "maximum of official yes/no chronic indicator",
            "max_recent_ailment_duration_days": "maximum reported non-monetary duration",
            "mean_recent_ailment_duration_days": "mean reported non-monetary duration",
            "any_recent_hospitalisation": "maximum of official yes/no hospitalised indicator",
            "any_recent_medical_advice": "maximum of official yes/no medical-advice indicator",
            "recent_treatment_type_count": "distinct official nature-of-treatment codes",
            "recent_level_of_care_diversity": "distinct official level-of-care codes",
            "recent_private_care_indicator": "any level-of-care code 3 or 6",
            "recent_public_care_indicator": "any level-of-care code 1, 4, or 5",
        },
    }
    return output, summary


def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    prepared = prepare_hospitalisation_data()
    modelling = add_community_outbreak(prepared.modelling)
    modelling = add_engineered_features(modelling)
    modelling, l5_summary = add_l5_aggregates(modelling)

    train_ids = pd.read_csv(V3_REPORT_DIR / "train_record_ids.csv", dtype=str)["hospitalisation_record_id"].tolist()
    test_ids = pd.read_csv(V3_REPORT_DIR / "test_record_ids.csv", dtype=str)["hospitalisation_record_id"].tolist()
    if len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)):
        raise RuntimeError("Saved v3 split contains duplicate episode IDs")
    indexed = modelling.set_index("hospitalisation_record_id", drop=False)
    missing = (set(train_ids) | set(test_ids)) - set(indexed.index)
    if missing:
        raise RuntimeError(f"Saved v3 split IDs missing from corrected modelling data: {len(missing)}")
    train = indexed.loc[train_ids].reset_index(drop=True)
    test = indexed.loc[test_ids].reset_index(drop=True)
    overlap = set(train["person_group_id"]) & set(test["person_group_id"])
    if overlap:
        raise RuntimeError("Person overlap found in saved v3 outer split")

    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE + 1)
    selection_index, validation_index = next(inner.split(train, train[TARGET], train["person_group_id"]))
    selection = train.iloc[selection_index].reset_index(drop=True)
    validation = train.iloc[validation_index].reset_index(drop=True)
    inner_overlap = set(selection["person_group_id"]) & set(validation["person_group_id"])
    if inner_overlap:
        raise RuntimeError("Person overlap found in inner model-selection split")
    split = {
        "source": "saved retraining_v3 episode IDs",
        "random_state": RANDOM_STATE,
        "train_records": int(len(train)),
        "test_records": int(len(test)),
        "train_people": int(train["person_group_id"].nunique()),
        "test_people": int(test["person_group_id"].nunique()),
        "person_overlap_count": 0,
        "selection_train_records": int(len(selection)),
        "validation_records": int(len(validation)),
        "selection_validation_person_overlap_count": 0,
        "saved_train_id_order_identical": train["hospitalisation_record_id"].tolist() == train_ids,
        "saved_test_id_order_identical": test["hospitalisation_record_id"].tolist() == test_ids,
    }
    return modelling, train, test, selection, validation, split, l5_summary, prepared.audit


def fit_routed(
    spec: CandidateSpec,
    train: pd.DataFrame,
    features: list[str],
    route_columns: list[str],
    minimum_records: int,
) -> dict[str, Any]:
    global_model = SafeRegressor(spec, features).fit(train)
    route_key = train[route_columns].astype("string").fillna("Missing").agg(" | ".join, axis=1)
    route_models: dict[str, SafeRegressor] = {}
    route_counts: dict[str, int] = {}
    for key, count in route_key.value_counts().items():
        route_counts[str(key)] = int(count)
        if int(count) >= minimum_records:
            route_models[str(key)] = SafeRegressor(spec, features).fit(train.loc[route_key.eq(key)].reset_index(drop=True))
    return {
        "architecture": "routed_regression",
        "spec": spec,
        "features": features,
        "route_columns": route_columns,
        "minimum_training_records": minimum_records,
        "global_model": global_model,
        "route_models": route_models,
        "route_training_counts": route_counts,
    }


def predict_routed(artifact: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    prediction = artifact["global_model"].predict(frame)
    route_key = frame[artifact["route_columns"]].astype("string").fillna("Missing").agg(" | ".join, axis=1)
    used_specific = np.zeros(len(frame), dtype=bool)
    for key, model in artifact["route_models"].items():
        mask = route_key.eq(key).to_numpy()
        if mask.any():
            prediction[mask] = model.predict(frame.loc[mask])
            used_specific[mask] = True
    return prediction, used_specific


def fit_high_cost_architecture(
    spec: CandidateSpec,
    train: pd.DataFrame,
    features: list[str],
    cutoff: float,
) -> dict[str, Any]:
    label = train[TARGET].ge(cutoff).astype(int).to_numpy()
    preprocessor = make_preprocessor(features)
    x = preprocessor.fit_transform(train[features])
    base_weight = weights(train)
    positives = max(1, int(label.sum()))
    negatives = max(1, int((1 - label).sum()))
    class_balance = np.where(label == 1, len(label) / (2.0 * positives), len(label) / (2.0 * negatives))
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=350,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=5.0,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
    classifier.fit(x, label, sample_weight=base_weight * class_balance)
    normal = train.loc[label == 0].reset_index(drop=True)
    high = train.loc[label == 1].reset_index(drop=True)
    return {
        "architecture": "high_cost_classifier_plus_regressors",
        "spec": spec,
        "features": features,
        "cutoff": float(cutoff),
        "classifier_preprocessor": preprocessor,
        "classifier": classifier,
        "normal_model": SafeRegressor(spec, features).fit(normal),
        "high_model": SafeRegressor(spec, features).fit(high),
        "normal_training_records": int(len(normal)),
        "high_training_records": int(len(high)),
    }


def predict_high_cost(artifact: dict[str, Any], frame: pd.DataFrame, probability_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    x = artifact["classifier_preprocessor"].transform(frame[artifact["features"]])
    probability = artifact["classifier"].predict_proba(x)[:, 1]
    high_prediction = probability >= probability_threshold
    prediction = artifact["normal_model"].predict(frame)
    if high_prediction.any():
        prediction[high_prediction] = artifact["high_model"].predict(frame.loc[high_prediction])
    return prediction, probability


def subgroup_rows(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    column: str,
    experiment: str,
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, indices in frame.groupby(column, dropna=False).groups.items():
        positions = frame.index.get_indexer(indices)
        subset = frame.loc[indices]
        rows.append(
            {
                "experiment": experiment,
                "evaluation_split": split,
                "subgroup_column": column,
                "subgroup": "Missing" if pd.isna(value) else str(value),
                "records": int(len(subset)),
                **metrics(subset, prediction[positions]),
            }
        )
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def spec_from_row(row: pd.Series, specs: list[CandidateSpec]) -> CandidateSpec:
    return next(spec for spec in specs if spec.name == str(row["experiment"]))


def model_payload(model: SafeRegressor, *, model_version: str, validation_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "artifact_type": "global_safe_regressor",
        "model_version": model_version,
        "estimator": model,
        "features": model.features,
        "target": TARGET,
        "target_transform": model.spec.target_transform,
        "model_family": model.spec.family,
        "parameters": model.spec.parameters,
        "encoded_feature_count": model.encoded_feature_count,
        "validation_metrics": validation_metrics,
        "known_leakage_columns": [],
    }


def routed_payload(artifact: dict[str, Any], *, model_version: str, validation_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "artifact_type": "routed_safe_regressor",
        "model_version": model_version,
        "artifact": artifact,
        "features": artifact["features"],
        "target": TARGET,
        "target_transform": artifact["spec"].target_transform,
        "model_family": artifact["spec"].family,
        "parameters": artifact["spec"].parameters,
        "route_columns": artifact["route_columns"],
        "minimum_training_records": artifact["minimum_training_records"],
        "validation_metrics": validation_metrics,
        "known_leakage_columns": [],
    }


def high_cost_payload(artifact: dict[str, Any], *, threshold: float, model_version: str, validation_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "artifact_type": "high_cost_two_stage_safe_regressor",
        "model_version": model_version,
        "artifact": artifact,
        "probability_threshold": threshold,
        "features": artifact["features"],
        "target": TARGET,
        "target_transform": artifact["spec"].target_transform,
        "model_family": artifact["spec"].family,
        "parameters": artifact["spec"].parameters,
        "validation_metrics": validation_metrics,
        "known_leakage_columns": [],
    }


def predict_payload(payload: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    if payload["artifact_type"] == "global_safe_regressor":
        return payload["estimator"].predict(frame)
    if payload["artifact_type"] == "routed_safe_regressor":
        return predict_routed(payload["artifact"], frame)[0]
    if payload["artifact_type"] == "high_cost_two_stage_safe_regressor":
        return predict_high_cost(payload["artifact"], frame, payload["probability_threshold"])[0]
    raise ValueError(f"Unknown artifact type {payload['artifact_type']}")


def save_plot_target(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = frame[TARGET].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(values, bins=80, color="#2563eb", alpha=0.85)
    axes[0].set_title("Target distribution (original INR scale)")
    axes[0].set_xlabel("Total medical expenditure (INR)")
    axes[0].set_ylabel("Episodes")
    axes[1].hist(np.log1p(values), bins=80, color="#0f766e", alpha=0.85)
    axes[1].set_title("Target distribution (log1p scale)")
    axes[1].set_xlabel("log1p(total medical expenditure)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_plot_actual_predicted(actual: np.ndarray, predicted: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual, predicted, s=7, alpha=0.25, color="#2563eb", edgecolors="none")
    limit = float(max(np.max(actual), np.max(predicted)))
    ax.plot([0, limit], [0, limit], linestyle="--", color="#dc2626", linewidth=1.2)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Actual expenditure (INR; symlog)")
    ax.set_ylabel("Predicted expenditure (INR; symlog)")
    ax.set_title("Best leakage-safe model: actual vs predicted")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_plot_residual(predicted: np.ndarray, residual: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(predicted, residual, s=7, alpha=0.25, color="#7c3aed", edgecolors="none")
    ax.axhline(0.0, linestyle="--", color="#111827", linewidth=1.0)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Predicted expenditure (INR; symlog)")
    ax.set_ylabel("Residual: actual - predicted (INR; symlog)")
    ax.set_title("Best leakage-safe model residuals")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def permutation_feature_importance(payload: dict[str, Any], frame: pd.DataFrame, maximum_rows: int = 4000) -> pd.DataFrame:
    sample = frame.sample(n=min(maximum_rows, len(frame)), random_state=RANDOM_STATE).reset_index(drop=True)
    actual = sample[TARGET].to_numpy(dtype=float)
    baseline_prediction = predict_payload(payload, sample)
    baseline_r2 = r2_score(actual, baseline_prediction)
    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[dict[str, Any]] = []
    for feature in payload["features"]:
        shuffled = sample.copy()
        shuffled[feature] = shuffled[feature].iloc[rng.permutation(len(shuffled))].to_numpy()
        shuffled_r2 = r2_score(actual, predict_payload(payload, shuffled))
        rows.append(
            {
                "feature": feature,
                "permutation_r2_decrease": float(baseline_r2 - shuffled_r2),
                "sample_records": int(len(sample)),
                "baseline_sample_r2": float(baseline_r2),
            }
        )
    return pd.DataFrame(rows).sort_values("permutation_r2_decrease", ascending=False, ignore_index=True)


def reproduce_baseline(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[str, Any], Pipeline, np.ndarray]:
    candidate = ModelCandidate("hist_gradient_boosting_log1p", "log1p", _hgb_factory, HGB_PARAMETERS)
    pipeline = fit_v3_candidate(candidate, train, BASE_FEATURES)
    prediction = predict_v3_candidate(pipeline, "log1p", test, BASE_FEATURES)
    unweighted, weighted = _metric_payload(test, prediction)
    row = {
        "experiment": "Baseline reproduced",
        "model": "HistGradientBoostingRegressor",
        "model_family": "hist_gradient_boosting",
        "target_transform": "log1p",
        "features": "v3 safe",
        "original_feature_count": len(BASE_FEATURES),
        "encoded_feature_count": len(pipeline.named_steps["preprocessor"].get_feature_names_out()),
        **unweighted,
        "survey_weighted_metrics": weighted,
        "parameters": HGB_PARAMETERS,
        "evaluation_split": "saved v3 person-grouped final test",
    }
    expected = {
        "mae": 17327.781543178316,
        "rmse": 59612.090744084795,
        "r2": 0.3557576273604861,
        "median_absolute_error": 3798.7131392552837,
        "rmsle": 2.339498310160965,
    }
    differences = {key: abs(float(unweighted[key]) - value) for key, value in expected.items()}
    if differences["r2"] > 1e-9 or differences["mae"] > 1e-6 or differences["rmse"] > 1e-6:
        raise RuntimeError(f"v3 baseline reproduction differs materially: {differences}")
    return row, pipeline, prediction


def validation_architecture_row(name: str, prediction: np.ndarray, validation: pd.DataFrame, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "architecture": name,
        **metrics(validation, prediction),
        "evaluation_split": "person-grouped inner validation",
        "details": json.dumps(details, sort_keys=True),
    }


def full_test_row(
    experiment: str,
    model: str,
    features_label: str,
    feature_count: int,
    encoded_count: int | str,
    test: pd.DataFrame,
    prediction: np.ndarray,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "model": model,
        "features": features_label,
        "original_feature_count": feature_count,
        "encoded_feature_count": encoded_count,
        **metrics(test, prediction),
        "parameters": json.dumps(parameters, sort_keys=True),
        "evaluation_split": "saved v3 person-grouped final test",
    }


def _markdown_metric(value: Any) -> str:
    try:
        return f"{float(value):,.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_model_comparison(frame: pd.DataFrame, path: Path) -> None:
    columns = ["experiment", "model", "features", "mae", "rmse", "r2", "median_absolute_error", "rmsle"]
    lines = [
        "# Leakage-safe model comparison",
        "",
        "All rows use the unchanged saved v3 outer test IDs. Candidate selection occurred only on the person-grouped inner validation split.",
        "",
        "| Experiment | Model | Features | MAE | RMSE | R² | Median AE | RMSLE |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in frame[columns].itertuples(index=False):
        lines.append(
            f"| {row.experiment} | {row.model} | {row.features} | {_markdown_metric(row.mae)} | "
            f"{_markdown_metric(row.rmse)} | {_markdown_metric(row.r2)} | "
            f"{_markdown_metric(row.median_absolute_error)} | {_markdown_metric(row.rmsle)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(*, quick: bool = False) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    primary_hash_before = _sha256(PRODUCTION_MODEL)
    started_at = datetime.now(timezone.utc).isoformat()
    print("Preparing corrected v3 data, exact saved split, engineered features, and person-level L5 aggregates", flush=True)
    modelling, train, test, selection, validation, split, l5_summary, preprocessing_audit = load_data()
    write_json(REPORT_DIR / "l5_aggregation_summary.json", l5_summary)
    pd.DataFrame(
        [{"metric": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value} for key, value in l5_summary.items()]
    ).to_csv(REPORT_DIR / "l5_aggregation_summary.csv", index=False)

    print("Reproducing locked v3 baseline", flush=True)
    baseline_row, baseline_pipeline, baseline_prediction = reproduce_baseline(train, test)
    joblib.dump(
        {
            "artifact_type": "reproduced_v3_pipeline",
            "pipeline": baseline_pipeline,
            "features": BASE_FEATURES,
            "target": TARGET,
            "target_transform": "log1p",
            "model_version": "safe_improvement_v4_baseline_reproduction",
        },
        MODEL_DIR / "baseline_v3_reproduced.joblib",
    )
    write_json(REPORT_DIR / "baseline_reproduction.json", baseline_row)
    print(f"Baseline exact: R2={baseline_row['r2']:.6f}, MAE={baseline_row['mae']:.2f}", flush=True)

    specs = candidate_specs()
    if quick:
        keep_names = {
            "hgb_identity_1", "hgb_log1p_1", "hgb_yeo_johnson_1",
            "extra_trees_identity_1", "random_forest_identity_1",
            "gradient_boosting_identity_1", "catboost_identity_1", "xgboost_identity_1",
        }
        specs = [spec for spec in specs if spec.name in keep_names]
    hyper_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        print(f"Tuning {index}/{len(specs)}: {spec.name}", flush=True)
        try:
            row, _ = evaluate_spec(spec, selection, validation, BASE_FEATURES)
            row["status"] = "ok"
            hyper_rows.append(row)
            print(f"  validation R2={row['r2']:.5f}, MAE={row['mae']:.2f}, seconds={row['fit_seconds']:.1f}", flush=True)
        except Exception as exc:
            hyper_rows.append(
                {
                    "experiment": spec.name,
                    "family": spec.family,
                    "target_transform": spec.target_transform,
                    "parameters": json.dumps(spec.parameters, sort_keys=True),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "evaluation_split": "person-grouped inner validation",
                }
            )
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
        pd.DataFrame(hyper_rows).to_csv(REPORT_DIR / "hyperparameter_results.csv", index=False)
    hyper = pd.DataFrame(hyper_rows)
    successful = hyper.loc[hyper["status"].eq("ok")].copy()
    required_families = {"hist_gradient_boosting", "extra_trees", "random_forest", "gradient_boosting", "catboost", "xgboost"}
    missing_families = required_families - set(successful["family"])
    if missing_families:
        raise RuntimeError(f"No successful candidate for required families: {sorted(missing_families)}")
    family_winners = (
        successful.sort_values(["family", "r2", "mae"], ascending=[True, False, True])
        .groupby("family", as_index=False, sort=False)
        .first()
    )
    family_winner_specs = {row.family: spec_from_row(pd.Series(row._asdict()), specs) for row in family_winners.itertuples(index=False)}
    overall_winner_row = successful.sort_values(["r2", "mae"], ascending=[False, True]).iloc[0]
    best_spec = spec_from_row(overall_winner_row, specs)
    print(f"Validation-selected model family: {best_spec.family} ({best_spec.name}, R2={overall_winner_row['r2']:.5f})", flush=True)

    target_transform_results = (
        successful.sort_values(["target_transform", "r2"], ascending=[True, False])
        .groupby("target_transform", as_index=False)
        .first()
    )
    target_transform_results.to_csv(REPORT_DIR / "target_transform_results.csv", index=False)

    print("Assessing engineered features one at a time", flush=True)
    base_row, _ = evaluate_spec(best_spec, selection, validation, BASE_FEATURES)
    engineering_rows: list[dict[str, Any]] = []
    individually_helpful: list[str] = []
    for feature in [*BASIC_ENGINEERED_FEATURES, *INTERACTION_FEATURES]:
        row, _ = evaluate_spec(best_spec, selection, validation, ordered_unique([*BASE_FEATURES, feature]))
        delta = float(row["r2"] - base_row["r2"])
        helpful = delta > 0.0002
        if helpful:
            individually_helpful.append(feature)
        engineering_rows.append(
            {
                "engineered_feature": feature,
                "validation_r2": row["r2"],
                "validation_mae": row["mae"],
                "r2_change_vs_same_model_base": delta,
                "selected_for_combination": helpful,
                "safe_derivation": {
                    "age_group": "age_years bins",
                    "length_of_stay_group": "length_of_stay_days bins",
                    "household_expenditure_per_person": "household usual consumer expenditure / household size",
                    "hospitalisations_per_household_member": "number of hospitalisations / household size",
                    "diagnostic_use_count": "sum of two safe received/not-received diagnostics",
                    "service_use_count": "sum of four safe received/not-received services",
                    "ailment_group": "official ailment-code broad group",
                }.get(feature, "categorical interaction of leakage-safe inputs"),
            }
        )
    engineering = pd.DataFrame(engineering_rows).sort_values("validation_r2", ascending=False, ignore_index=True)
    engineering.to_csv(REPORT_DIR / "feature_engineering_summary.csv", index=False)

    ablation_specs = [
        ("Baseline v3 safe", BASE_FEATURES),
        ("+ geography", ordered_unique([*BASE_FEATURES, *GEOGRAPHY_FEATURES])),
        ("+ health history", ordered_unique([*BASE_FEATURES, *HEALTH_HISTORY_FEATURES])),
        ("+ selected engineered", ordered_unique([*BASE_FEATURES, *individually_helpful])),
        ("+ all engineered", ordered_unique([*BASE_FEATURES, *BASIC_ENGINEERED_FEATURES, *INTERACTION_FEATURES])),
        ("+ L5 aggregate", ordered_unique([*BASE_FEATURES, *L5_FEATURES])),
        ("+ geography + health", ordered_unique([*BASE_FEATURES, *GEOGRAPHY_FEATURES, *HEALTH_HISTORY_FEATURES])),
        (
            "Best combined safe blocks",
            ordered_unique([*BASE_FEATURES, *GEOGRAPHY_FEATURES, *HEALTH_HISTORY_FEATURES, *individually_helpful, *L5_FEATURES]),
        ),
        (
            "All tested safe blocks",
            ordered_unique([*BASE_FEATURES, *GEOGRAPHY_FEATURES, *HEALTH_HISTORY_FEATURES, *BASIC_ENGINEERED_FEATURES, *INTERACTION_FEATURES, *L5_FEATURES]),
        ),
    ]
    feature_rows: list[dict[str, Any]] = []
    print("Running controlled safe feature-block ablations", flush=True)
    for label, features in ablation_specs:
        print(f"  {label} ({len(features)} inputs)", flush=True)
        row, _ = evaluate_spec(best_spec, selection, validation, features)
        row["feature_ablation"] = label
        row["feature_list"] = json.dumps(features)
        row["r2_change_vs_base"] = float(row["r2"] - base_row["r2"])
        feature_rows.append(row)
        pd.DataFrame(feature_rows).to_csv(REPORT_DIR / "feature_ablation_results.csv", index=False)
    feature_results = pd.DataFrame(feature_rows).sort_values(["r2", "mae"], ascending=[False, True], ignore_index=True)
    feature_results.to_csv(REPORT_DIR / "feature_ablation_results.csv", index=False)
    best_feature_row = feature_results.iloc[0]
    best_features = json.loads(best_feature_row["feature_list"])
    if set(best_features) & PROHIBITED_FEATURES:
        raise RuntimeError("Validation-selected feature set contains prohibited leakage")
    print(f"Validation-selected feature set: {best_feature_row['feature_ablation']} ({len(best_features)} inputs, R2={best_feature_row['r2']:.5f})", flush=True)

    print("Evaluating routed and two-stage architectures on validation only", flush=True)
    architecture_rows: list[dict[str, Any]] = []
    architecture_validation_artifacts: dict[str, Any] = {}
    global_validation_model = SafeRegressor(best_spec, best_features).fit(selection)
    global_validation_prediction = global_validation_model.predict(validation)
    architecture_rows.append(validation_architecture_row("Global model", global_validation_prediction, validation, {"feature_set": str(best_feature_row["feature_ablation"])}))
    architecture_validation_artifacts["Global model"] = global_validation_model

    institution_validation = fit_routed(best_spec, selection, best_features, ["medical_institution_type"], 1000)
    institution_validation_prediction, institution_used = predict_routed(institution_validation, validation)
    architecture_rows.append(validation_architecture_row("Institution routed", institution_validation_prediction, validation, {"specific_models": len(institution_validation["route_models"]), "specific_route_coverage": float(institution_used.mean())}))
    architecture_validation_artifacts["Institution routed"] = institution_validation

    ailment_validation = fit_routed(best_spec, selection, best_features, ["ailment_group"], 1500)
    ailment_validation_prediction, ailment_used = predict_routed(ailment_validation, validation)
    architecture_rows.append(validation_architecture_row("Ailment-group routed", ailment_validation_prediction, validation, {"specific_models": len(ailment_validation["route_models"]), "specific_route_coverage": float(ailment_used.mean())}))
    architecture_validation_artifacts["Ailment-group routed"] = ailment_validation

    hierarchy_validation = fit_routed(best_spec, selection, best_features, ["medical_institution_type", "ailment_group"], 900)
    hierarchy_validation_prediction, hierarchy_used = predict_routed(hierarchy_validation, validation)
    architecture_rows.append(validation_architecture_row("Institution + ailment hierarchy", hierarchy_validation_prediction, validation, {"specific_models": len(hierarchy_validation["route_models"]), "specific_route_coverage": float(hierarchy_used.mean())}))
    architecture_validation_artifacts["Institution + ailment hierarchy"] = hierarchy_validation

    selection_cutoff = float(selection[TARGET].quantile(0.99))
    high_validation = fit_high_cost_architecture(best_spec, selection, best_features, selection_cutoff)
    high_threshold_rows: list[dict[str, Any]] = []
    for threshold in [0.20, 0.35, 0.50]:
        prediction, probability = predict_high_cost(high_validation, validation, threshold)
        predicted_high = probability >= threshold
        actual_high = validation[TARGET].ge(selection_cutoff).to_numpy()
        row = validation_architecture_row(
            f"High-cost two-stage @ {threshold:.2f}",
            prediction,
            validation,
            {
                "training_cutoff": selection_cutoff,
                "probability_threshold": threshold,
                "predicted_high_records": int(predicted_high.sum()),
                "precision": float(precision_score(actual_high, predicted_high, zero_division=0)),
                "recall": float(recall_score(actual_high, predicted_high, zero_division=0)),
            },
        )
        high_threshold_rows.append(row)
    best_high_row = sorted(high_threshold_rows, key=lambda row: (row["r2"], -row["mae"]), reverse=True)[0]
    architecture_rows.append(best_high_row)
    best_high_threshold = float(json.loads(best_high_row["details"])["probability_threshold"])
    architecture_validation_artifacts[best_high_row["architecture"]] = high_validation

    architecture_validation = pd.DataFrame(architecture_rows).sort_values(["r2", "mae"], ascending=[False, True], ignore_index=True)
    global_r2 = float(architecture_validation.loc[architecture_validation["architecture"].eq("Global model"), "r2"].iloc[0])
    raw_best_architecture = str(architecture_validation.iloc[0]["architecture"])
    raw_best_r2 = float(architecture_validation.iloc[0]["r2"])
    selected_architecture = "Global model" if raw_best_architecture != "Global model" and raw_best_r2 - global_r2 < 0.005 else raw_best_architecture
    architecture_validation["selected"] = architecture_validation["architecture"].eq(selected_architecture)
    architecture_validation.to_csv(REPORT_DIR / "architecture_validation.csv", index=False)
    print(f"Frozen architecture before final test: {selected_architecture}", flush=True)

    public_private_validation_rows = subgroup_rows(validation, institution_validation_prediction, "medical_institution_type", "Institution routed", "person-grouped inner validation")
    ailment_validation_rows = subgroup_rows(validation, ailment_validation_prediction, "ailment_group", "Ailment-group routed", "person-grouped inner validation")

    # No model or architecture choice below this point may use final-test results.
    print("Selection frozen; fitting family winners and selected architectures on full training data", flush=True)
    final_rows: list[dict[str, Any]] = [
        {
            "experiment": baseline_row["experiment"],
            "model": baseline_row["model"],
            "features": baseline_row["features"],
            "original_feature_count": baseline_row["original_feature_count"],
            "encoded_feature_count": baseline_row["encoded_feature_count"],
            **{key: baseline_row[key] for key in ["mae", "rmse", "r2", "median_absolute_error", "rmsle"]},
            "parameters": json.dumps(HGB_PARAMETERS, sort_keys=True),
            "evaluation_split": baseline_row["evaluation_split"],
        }
    ]
    model_filenames = {
        "hist_gradient_boosting": "hist_gradient_boosting_safe.joblib",
        "extra_trees": "extratrees_safe.joblib",
        "random_forest": "random_forest_safe.joblib",
        "gradient_boosting": "gradient_boosting_safe.joblib",
        "catboost": "catboost_safe.joblib",
        "xgboost": "xgboost_safe.joblib",
    }
    final_family_payloads: dict[str, dict[str, Any]] = {}
    for family in sorted(required_families):
        spec = family_winner_specs[family]
        print(f"  Final family fit: {family} ({spec.name})", flush=True)
        fitted = SafeRegressor(spec, BASE_FEATURES).fit(train)
        prediction = fitted.predict(test)
        winner_validation = successful.loc[successful["experiment"].eq(spec.name)].iloc[0].to_dict()
        payload = model_payload(fitted, model_version=f"safe_improvement_v4_{family}", validation_metrics={key: winner_validation[key] for key in ["mae", "rmse", "r2", "median_absolute_error", "rmsle"]})
        joblib.dump(payload, MODEL_DIR / model_filenames[family])
        final_family_payloads[family] = payload
        final_rows.append(full_test_row(f"Tuned {family}", spec.name, "v3 safe", len(BASE_FEATURES), fitted.encoded_feature_count, test, prediction, spec.parameters))

    print("  Final best-feature global fit", flush=True)
    final_global_model = SafeRegressor(best_spec, best_features).fit(train)
    final_global_prediction = final_global_model.predict(test)
    final_global_payload = model_payload(final_global_model, model_version="safe_improvement_v4_best_features_global", validation_metrics={key: float(architecture_validation.loc[architecture_validation["architecture"].eq("Global model"), key].iloc[0]) for key in ["mae", "rmse", "r2", "median_absolute_error", "rmsle"]})
    joblib.dump(final_global_payload, MODEL_DIR / "best_features_global.joblib")
    final_rows.append(full_test_row("Engineered/aggregated global", best_spec.name, str(best_feature_row["feature_ablation"]), len(best_features), final_global_model.encoded_feature_count, test, final_global_prediction, best_spec.parameters))

    print("  Final institution-routed fit", flush=True)
    final_institution = fit_routed(best_spec, train, best_features, ["medical_institution_type"], 1000)
    final_institution_prediction, final_institution_used = predict_routed(final_institution, test)
    final_institution_payload = routed_payload(final_institution, model_version="safe_improvement_v4_institution_routed", validation_metrics={key: float(architecture_validation.loc[architecture_validation["architecture"].eq("Institution routed"), key].iloc[0]) for key in ["mae", "rmse", "r2", "median_absolute_error", "rmsle"]})
    joblib.dump(final_institution_payload, MODEL_DIR / "institution_routed.joblib")
    final_rows.append(full_test_row("Routed Public/Private/Other", f"{best_spec.name} subgroup models", str(best_feature_row["feature_ablation"]), len(best_features), "native/direct or per-route", test, final_institution_prediction, best_spec.parameters))

    institution_labels = {
        "public_model.joblib": "Government or public hospital",
        "private_model.joblib": "Private hospital",
        "other_institution_model.joblib": "Charitable, trust or NGO-run hospital",
    }
    for filename, label in institution_labels.items():
        estimator = final_institution["route_models"].get(label, final_institution["global_model"])
        joblib.dump(model_payload(estimator, model_version=f"safe_improvement_v4_{filename.removesuffix('.joblib')}"), MODEL_DIR / filename)

    print("  Final ailment-group-routed fit", flush=True)
    final_ailment = fit_routed(best_spec, train, best_features, ["ailment_group"], 1500)
    final_ailment_prediction, final_ailment_used = predict_routed(final_ailment, test)
    final_ailment_payload = routed_payload(final_ailment, model_version="safe_improvement_v4_ailment_routed", validation_metrics={key: float(architecture_validation.loc[architecture_validation["architecture"].eq("Ailment-group routed"), key].iloc[0]) for key in ["mae", "rmse", "r2", "median_absolute_error", "rmsle"]})
    joblib.dump(final_ailment_payload, MODEL_DIR / "ailment_routed.joblib")
    final_rows.append(full_test_row("Ailment Groups", f"{best_spec.name} group models", str(best_feature_row["feature_ablation"]), len(best_features), "native/direct or per-route", test, final_ailment_prediction, best_spec.parameters))

    print("  Final institution + ailment hierarchy fit", flush=True)
    final_hierarchy = fit_routed(best_spec, train, best_features, ["medical_institution_type", "ailment_group"], 900)
    final_hierarchy_prediction, final_hierarchy_used = predict_routed(final_hierarchy, test)
    final_hierarchy_payload = routed_payload(final_hierarchy, model_version="safe_improvement_v4_institution_ailment_hierarchy", validation_metrics={key: float(architecture_validation.loc[architecture_validation["architecture"].eq("Institution + ailment hierarchy"), key].iloc[0]) for key in ["mae", "rmse", "r2", "median_absolute_error", "rmsle"]})
    joblib.dump(final_hierarchy_payload, MODEL_DIR / "institution_ailment_hierarchy.joblib")
    final_rows.append(full_test_row("Institution + ailment hierarchy", f"{best_spec.name} intersection models", str(best_feature_row["feature_ablation"]), len(best_features), "native/direct or per-route", test, final_hierarchy_prediction, best_spec.parameters))

    print("  Final high-cost two-stage fit", flush=True)
    train_cutoff = float(train[TARGET].quantile(0.99))
    final_high = fit_high_cost_architecture(best_spec, train, best_features, train_cutoff)
    final_high_prediction, final_high_probability = predict_high_cost(final_high, test, best_high_threshold)
    high_validation_metrics = {key: float(best_high_row[key]) for key in ["mae", "rmse", "r2", "median_absolute_error", "rmsle"]}
    final_high_payload = high_cost_payload(final_high, threshold=best_high_threshold, model_version="safe_improvement_v4_high_cost_two_stage", validation_metrics=high_validation_metrics)
    joblib.dump(final_high_payload, MODEL_DIR / "high_cost_two_stage.joblib")
    final_rows.append(full_test_row("High-cost classifier + regressors", best_spec.name, str(best_feature_row["feature_ablation"]), len(best_features), "classifier + two regressors", test, final_high_prediction, {**best_spec.parameters, "high_cost_probability_threshold": best_high_threshold, "training_cutoff": train_cutoff}))

    final_architecture_payloads = {
        "Global model": final_global_payload,
        "Institution routed": final_institution_payload,
        "Ailment-group routed": final_ailment_payload,
        "Institution + ailment hierarchy": final_hierarchy_payload,
        best_high_row["architecture"]: final_high_payload,
    }
    best_payload = final_architecture_payloads[selected_architecture]
    best_prediction = predict_payload(best_payload, test)
    joblib.dump(best_payload, MODEL_DIR / "best_safe_model.joblib")

    best_test_metrics = metrics(test, best_prediction)
    best_weighted_metrics = weighted_metrics(test, best_prediction)
    final_rows.append(full_test_row("Best Combined", selected_architecture, str(best_feature_row["feature_ablation"]), len(best_features), best_payload.get("encoded_feature_count", "routed/multi-model"), test, best_prediction, best_spec.parameters))
    experiment_results = pd.DataFrame(final_rows)
    experiment_results.to_csv(REPORT_DIR / "experiment_results.csv", index=False)
    write_model_comparison(experiment_results, REPORT_DIR / "model_comparison.md")

    public_private_test_rows = subgroup_rows(test, final_institution_prediction, "medical_institution_type", "Institution routed", "saved v3 final test")
    public_private = pd.DataFrame([*public_private_validation_rows, *public_private_test_rows])
    public_private.to_csv(REPORT_DIR / "public_private_results.csv", index=False)
    ailment_test_rows = subgroup_rows(test, final_ailment_prediction, "ailment_group", "Ailment-group routed", "saved v3 final test")
    ailment_results = pd.DataFrame([*ailment_validation_rows, *ailment_test_rows])
    ailment_results.to_csv(REPORT_DIR / "ailment_group_results.csv", index=False)
    subgroup_results = pd.DataFrame(
        [
            *subgroup_rows(test, best_prediction, "gender", "Best Combined", "saved v3 final test"),
            *subgroup_rows(test, best_prediction, "medical_institution_type", "Best Combined", "saved v3 final test"),
            *subgroup_rows(test, best_prediction, "ailment_group", "Best Combined", "saved v3 final test"),
        ]
    )
    subgroup_results.to_csv(REPORT_DIR / "subgroup_results.csv", index=False)

    test_top_one_cutoff = float(test[TARGET].quantile(0.99))
    top_mask = test[TARGET].ge(test_top_one_cutoff).to_numpy()
    outlier_rows = [
        {"analysis": "target_min", "value": float(modelling[TARGET].min())},
        {"analysis": "target_median", "value": float(modelling[TARGET].median())},
        {"analysis": "target_mean", "value": float(modelling[TARGET].mean())},
        {"analysis": "target_p90", "value": float(modelling[TARGET].quantile(0.90))},
        {"analysis": "target_p95", "value": float(modelling[TARGET].quantile(0.95))},
        {"analysis": "target_p99", "value": float(modelling[TARGET].quantile(0.99))},
        {"analysis": "target_p999", "value": float(modelling[TARGET].quantile(0.999))},
        {"analysis": "target_max", "value": float(modelling[TARGET].max())},
        {"analysis": "test_top_1_percent_cutoff", "value": test_top_one_cutoff},
    ]
    outlier_metric_rows = []
    for label, mask in [("full_test", np.ones(len(test), dtype=bool)), ("top_1_percent", top_mask), ("non_top_1_percent", ~top_mask)]:
        subset = test.loc[mask]
        outlier_metric_rows.append(
            {
                "analysis": label,
                "records": int(mask.sum()),
                "target_cutoff_definition": "test empirical 99th percentile used only for post-model diagnostic grouping",
                "extreme_records_valid": True,
                "validity_basis": "non-negative targets and exact equality to official medical-component sum verified by corrected v3 preprocessing",
                **metrics(subset, best_prediction[mask]),
            }
        )
    pd.DataFrame(outlier_rows + outlier_metric_rows).to_csv(REPORT_DIR / "outlier_analysis.csv", index=False)

    id_columns = [
        "hospitalisation_record_id", "person_group_id", "household_id", "fsu_serial_no",
        "sample_household_no", "person_serial_no", "hospitalisation_case_serial_no",
    ]
    predictions = test[id_columns + ["medical_institution_type", "ailment_nature", "ailment_group", TARGET]].copy()
    predictions["predicted_total_medical_expenditure_rs"] = best_prediction
    predictions["residual_rs"] = predictions[TARGET] - best_prediction
    predictions["absolute_error_rs"] = np.abs(predictions["residual_rs"])
    predictions["test_top_1_percent_actual"] = top_mask.astype(int)
    predictions.to_csv(REPORT_DIR / "best_model_predictions.csv", index=False)

    importance = permutation_feature_importance(best_payload, test)
    importance.to_csv(REPORT_DIR / "best_model_feature_importance.csv", index=False)
    actual = test[TARGET].to_numpy(dtype=float)
    save_plot_target(modelling, REPORT_DIR / "target_distribution.png")
    save_plot_actual_predicted(actual, best_prediction, REPORT_DIR / "actual_vs_predicted.png")
    save_plot_residual(best_prediction, actual - best_prediction, REPORT_DIR / "residual_plot.png")

    reloaded = joblib.load(MODEL_DIR / "best_safe_model.joblib")
    reload_prediction = predict_payload(reloaded, test.iloc[:10])
    reload_matches = bool(np.allclose(reload_prediction, best_prediction[:10], rtol=1e-10, atol=1e-7))
    if not reload_matches or not np.isfinite(reload_prediction).all():
        raise RuntimeError("Reloaded best safe model did not reproduce finite predictions")

    binary_validation = {
        feature: sorted(pd.Series(modelling[feature]).dropna().astype(float).unique().tolist())
        for feature in sorted(BINARY_FEATURES & set(best_features))
    }
    nonbinary = {feature: values for feature, values in binary_validation.items() if not set(values).issubset({0.0, 1.0})}
    if nonbinary:
        raise RuntimeError(f"Best model binary fields are not 0/1: {nonbinary}")
    primary_hash_after = _sha256(PRODUCTION_MODEL)
    if primary_hash_before != primary_hash_after:
        raise RuntimeError("Production primary_model.joblib changed during v4 experiments")

    final_feature_count = len(best_features)
    encoded_count: int | str
    if best_payload["artifact_type"] == "global_safe_regressor":
        encoded_count = best_payload["encoded_feature_count"]
    else:
        encoded_count = best_payload["artifact"]["global_model"].encoded_feature_count
    best_metrics_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "Highest validation R2, with global fallback when a routed gain is below 0.005",
        "selected_architecture": selected_architecture,
        "selected_model_type": best_spec.family,
        "selected_candidate": best_spec.name,
        "target_transform": best_spec.target_transform,
        "hyperparameters": best_spec.parameters,
        "selected_feature_ablation": str(best_feature_row["feature_ablation"]),
        "features": best_features,
        "original_feature_count": final_feature_count,
        "encoded_feature_count_per_global_or_fallback_model": encoded_count,
        "metrics": best_test_metrics,
        "survey_weighted_metrics": best_weighted_metrics,
        "baseline_r2": float(baseline_row["r2"]),
        "r2_improvement": float(best_test_metrics["r2"] - baseline_row["r2"]),
        "achieved_r2_at_least_0_70": bool(best_test_metrics["r2"] >= 0.70),
        "split": split,
        "known_leakage_columns_in_model": sorted(set(best_features) & PROHIBITED_FEATURES),
        "l5_row_multiplication": l5_summary["row_multiplication"],
        "unresolved_categorical_800x_codes": preprocessing_audit["unresolved_categorical_800x_codes_detected"],
        "binary_values": binary_validation,
        "artifact_reload_prediction_match": reload_matches,
        "production_primary_model_sha256_before": primary_hash_before,
        "production_primary_model_sha256_after": primary_hash_after,
        "production_primary_model_replaced": False,
    }
    write_json(REPORT_DIR / "best_model_metrics.json", best_metrics_payload)

    routing_lines = [
        "# Routing architecture evaluation",
        "",
        f"Architecture selection was frozen on the inner person-grouped validation split. Selected: **{selected_architecture}**.",
        "",
        "- Institution route: public, private, and charitable/NGO models when each route has at least 1,000 training episodes; otherwise the global fallback.",
        "- Ailment route: official-code broad ailment groups with at least 1,500 training episodes; otherwise the global fallback.",
        "- Institution + ailment hierarchy: intersection models with at least 900 training episodes; otherwise the global fallback.",
        "- High-cost route: a leakage-safe classifier predicts whether the train-defined 99th-percentile threshold is exceeded, then routes to normal/high regressors.",
        "",
        "Every route uses only the final leakage-safe feature list. Full-test metrics are included in experiment_results.csv.",
    ]
    (REPORT_DIR / "routing_architecture.md").write_text("\n".join(routing_lines) + "\n", encoding="utf-8")

    final_summary = {
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline_row,
        "family_winners": {family: asdict(spec) for family, spec in family_winner_specs.items()},
        "selected_model": asdict(best_spec),
        "selected_feature_ablation": str(best_feature_row["feature_ablation"]),
        "selected_features": best_features,
        "selected_architecture": selected_architecture,
        "best_metrics": best_test_metrics,
        "r2_improvement_vs_baseline": float(best_test_metrics["r2"] - baseline_row["r2"]),
        "achieved_r2_at_least_0_70": bool(best_test_metrics["r2"] >= 0.70),
        "split": split,
        "l5_summary": l5_summary,
        "production_primary_model_replaced": False,
        "dependency_changes": {"xgboost": importlib.metadata.version("xgboost")},
        "software": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "catboost": importlib.metadata.version("catboost"),
            "xgboost": importlib.metadata.version("xgboost"),
            "joblib": importlib.metadata.version("joblib"),
        },
    }
    write_json(REPORT_DIR / "experiment_summary.json", final_summary)
    summary_lines = [
        "# NSS 80 leakage-safe improvement v4",
        "",
        "## Outcome",
        "",
        f"The validation-frozen best architecture is **{selected_architecture}** using **{best_spec.name}** and the **{best_feature_row['feature_ablation']}** feature set.",
        "",
        f"Final test: MAE {_markdown_metric(best_test_metrics['mae'])} INR; RMSE {_markdown_metric(best_test_metrics['rmse'])} INR; R² {_markdown_metric(best_test_metrics['r2'])}; Median AE {_markdown_metric(best_test_metrics['median_absolute_error'])} INR; RMSLE {_markdown_metric(best_test_metrics['rmsle'])}.",
        "",
        f"R² change versus the reproduced v3 baseline: {_markdown_metric(best_test_metrics['r2'] - baseline_row['r2'])}. R² ≥ 0.70 achieved: {'yes' if best_test_metrics['r2'] >= 0.70 else 'no'}.",
        "",
        "## Scientific safeguards",
        "",
        "- Exact saved v3 outer train/test episode IDs were reused, with zero person overlap.",
        "- Hyperparameters, feature blocks, and routing architecture were selected only on the inner person-grouped validation split.",
        "- L5 was aggregated to one row per person before joining; monetary/payment L5 fields were never loaded into the aggregation.",
        "- Valid high-cost cases were retained and the full test set remains the primary result.",
        "- The production primary model hash remained unchanged.",
        "",
        "See model_comparison.md, the CSV ablation/subgroup tables, and best_model_metrics.json for the detailed results.",
    ]
    (REPORT_DIR / "experiment_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected_architecture": selected_architecture, "selected_model": best_spec.name, "best_metrics": best_test_metrics, "r2_improvement": best_test_metrics["r2"] - baseline_row["r2"], "production_primary_model_replaced": False}, indent=2), flush=True)
    return final_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run one controlled candidate per family for a fast pipeline smoke test.")
    args = parser.parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
