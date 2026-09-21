"""Runtime service for the selected NSS 80 practical 15-input model."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from model_training.nss80.practical15_intervals import bounds
from model_training.nss80.production_practical15 import (
    ARTIFACT_MODEL_VERSION,
    ARTIFACT_TYPE,
    BINARY_FEATURES,
    FEATURES,
    MODEL_VERSION,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    NUMERIC_LIMITS,
    OPTIONAL_FIELDS,
    SERVICE_RECEIPT_FEATURES,
    TARGET,
    canonical_category,
    categories_from_estimator,
    model_safe_values,
    validate_consistency,
    validate_numeric,
)


LIMITATION = (
    "Primary research estimate from India's NSS 80th Round (January-December "
    "2025). Output is INR per inpatient case; it is not a Malaysian price, a "
    "currency conversion, clinical advice, or a guaranteed hospital bill."
)
def interval_description(level: float) -> str:
    percent = int(round(level * 100))
    return (
        f"Estimated {percent}% person-calibrated prediction range based on similar "
        "hospitalisation records in the NSS 80 dataset. Actual medical expenditure "
        "can vary even for patients with similar characteristics; marginal coverage "
        "across comparable people is not an individual bill guarantee."
    )


class NSS80PredictionService:
    def __init__(self, models_dir: str | Path | None = None) -> None:
        backend_root = Path(__file__).resolve().parents[2]
        self.models_dir = Path(models_dir) if models_dir is not None else backend_root / "models" / "nss80"
        point_path = self.models_dir / "primary_model.joblib"
        metadata_path = self.models_dir / "metadata.json"
        for path in (point_path, metadata_path):
            if not path.exists():
                raise FileNotFoundError(f"NSS 80 production artifact is missing: {path.name}")

        artifact = joblib.load(point_path)
        if not isinstance(artifact, dict):
            raise ValueError("NSS 80 practical 15 artifact must be a metadata dictionary")
        if artifact.get("artifact_type") != ARTIFACT_TYPE:
            raise ValueError("NSS 80 production artifact is not the selected global model")
        if artifact.get("model_version") != ARTIFACT_MODEL_VERSION:
            raise ValueError("NSS 80 production artifact version does not match practical 15")
        if artifact.get("target") != TARGET or artifact.get("target_transform") != "identity":
            raise ValueError("NSS 80 practical 15 target metadata is incompatible")
        if artifact.get("model_family") != "hist_gradient_boosting":
            raise ValueError("NSS 80 production artifact is not HistGradientBoosting")
        if artifact.get("known_leakage_columns"):
            raise ValueError("NSS 80 production artifact declares prohibited leakage columns")
        self.features = list(artifact.get("features", []))
        if self.features != FEATURES:
            raise ValueError("NSS 80 practical 15 feature names or order do not match runtime code")
        if artifact.get("encoded_feature_count") != 90:
            raise ValueError("NSS 80 practical 15 encoded feature count is incompatible")

        self.estimator = artifact.get("estimator")
        if self.estimator is None or getattr(self.estimator, "preprocessor_", None) is None:
            raise ValueError("NSS 80 practical 15 fitted preprocessing is missing")
        if not isinstance(getattr(self.estimator, "model_", None), HistGradientBoostingRegressor):
            raise ValueError("NSS 80 practical 15 estimator is not HistGradientBoostingRegressor")
        estimator_spec = getattr(self.estimator, "spec", None)
        if getattr(estimator_spec, "family", None) != "hist_gradient_boosting" or getattr(estimator_spec, "target_transform", None) != "identity":
            raise ValueError("NSS 80 practical 15 estimator specification is incompatible")
        if list(getattr(self.estimator, "features", [])) != FEATURES:
            raise ValueError("NSS 80 practical 15 estimator feature order is incompatible")
        if list(getattr(self.estimator, "numeric", [])) != NUMERIC_FEATURES:
            raise ValueError("NSS 80 practical 15 numeric feature declaration is incompatible")
        if list(getattr(self.estimator, "binary", [])) != BINARY_FEATURES:
            raise ValueError("NSS 80 practical 15 binary feature declaration is incompatible")
        if list(getattr(self.estimator, "nominal", [])) != NOMINAL_FEATURES:
            raise ValueError("NSS 80 practical 15 nominal feature declaration is incompatible")
        if getattr(self.estimator, "encoded_feature_count", None) != 90:
            raise ValueError("NSS 80 practical 15 preprocessor does not produce 90 features")

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("model_version") != MODEL_VERSION:
            raise ValueError("NSS 80 metadata does not identify practical 15")
        if self.metadata.get("target_transform") != "identity":
            raise ValueError("NSS 80 metadata has an invalid target transform")
        if self.metadata.get("input_features") != FEATURES:
            raise ValueError("NSS 80 metadata feature order does not match practical 15")
        point_sha = hashlib.sha256(point_path.read_bytes()).hexdigest()
        if point_sha != self.metadata.get("model_sha256"):
            raise ValueError("Production point artifact hash does not match metadata")

        self.categories = categories_from_estimator(self.estimator)
        interval_info = self.metadata["prediction_interval"]
        interval_path = self.models_dir / interval_info["artifact"]
        if hashlib.sha256(interval_path.read_bytes()).hexdigest() != interval_info["sha256"]:
            raise ValueError("Interval artifact hash does not match metadata")
        self.interval = joblib.load(interval_path)
        if self.interval.get("point_sha256") != point_sha or self.interval.get("features") != FEATURES:
            raise ValueError("Interval artifact does not match the practical 15 point model")
        if self.interval.get("nominal_coverage") != interval_info.get("nominal_coverage"):
            raise ValueError("Interval artifact coverage does not match production metadata")
        self.interval_description = interval_description(self.interval["nominal_coverage"])

    def _validate_and_transform(self, values: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        missing = sorted(set(FEATURES).difference(values))
        extra = sorted(set(values).difference(FEATURES))
        if missing or extra:
            raise ValueError(f"Expected exactly the 15 practical features; missing={missing}, extra={extra}")
        cleaned: dict[str, Any] = {}
        for field in NUMERIC_FEATURES:
            cleaned[field] = validate_numeric(field, values[field])
        for field in [*BINARY_FEATURES, *NOMINAL_FEATURES]:
            canonical = canonical_category(field, values[field])
            if canonical is None:
                if field not in OPTIONAL_FIELDS:
                    raise ValueError(f"{field} is required")
                cleaned[field] = None
                continue
            lookup = {choice.casefold(): choice for choice in self.categories[field]}
            fitted_value = lookup.get(canonical.casefold())
            if fitted_value is None:
                raise ValueError(f"Invalid {field}; select a value returned by /predict/nss80/options")
            cleaned[field] = fitted_value
        ordered_cleaned = {feature: cleaned[feature] for feature in FEATURES}
        validate_consistency(ordered_cleaned)
        return ordered_cleaned, model_safe_values(ordered_cleaned)

    def prediction_options(self) -> dict[str, Any]:
        test_metrics = self.metadata["metrics"]["test"]
        return {
            "component": "nss80_primary_research_model",
            "research_role": "primary",
            "currency": "INR",
            "target": "Hospitalisation-related total medical expenditure per inpatient case",
            "field_order": FEATURES,
            "optional_fields": OPTIONAL_FIELDS,
            "numeric_limits": NUMERIC_LIMITS,
            "categories": self.categories,
            "not_applicable_value": None,
            "service_receipt_policy": {
                "source_labels": "Only Received / Not received are accepted.",
                "model_representation": "Not received = 0; Received = 1.",
                "payment_status_detail_used": False,
            },
            "model": {
                "model_version": MODEL_VERSION,
                "model_type": "HistGradientBoostingRegressor",
                "architecture": "Global model",
                "target_transform": "identity",
                "feature_count": 15,
                "encoded_feature_count": 90,
                "r2_test": test_metrics["r2"],
                "mae_test_inr": test_metrics["mae"],
                "rmse_test_inr": test_metrics["rmse"],
                "median_absolute_error_test_inr": test_metrics["median_absolute_error"],
                "rmsle_test": test_metrics["rmsle"],
            },
            "prediction_interval": {
                "available": True,
                "reason": self.interval_description,
                "nominal_coverage": self.interval["nominal_coverage"],
            },
        }

    def validate_input(self, values: dict[str, Any]) -> dict[str, Any]:
        cleaned, _ = self._validate_and_transform(values)
        return cleaned

    def predict(self, values: dict[str, Any]) -> dict[str, Any]:
        cleaned, transformed = self._validate_and_transform(values)
        frame = pd.DataFrame([transformed], columns=FEATURES)
        raw_prediction = float(self.estimator.predict(frame)[0])
        if not np.isfinite(raw_prediction):
            raise ValueError("NSS 80 practical 15 returned a non-finite prediction")
        point = round(max(0.0, raw_prediction), 2)
        lower, upper = bounds(self.interval, frame, np.array([point]))
        interval_result = {
            "lower": round(float(lower[0]), 2),
            "upper": round(float(upper[0]), 2),
            "coverage_level": self.interval["nominal_coverage"],
            "currency": "INR",
        }
        test_metrics = self.metadata["metrics"]["test"]
        return {
            "component": "nss80_primary_research_model",
            "research_role": "primary",
            "model_version": MODEL_VERSION,
            "model_type": "HistGradientBoostingRegressor",
            "architecture": "Global model",
            "target_transform": "identity",
            "feature_count": 15,
            "encoded_feature_count": 90,
            "r2_test": test_metrics["r2"],
            "payment_status_detail_used": False,
            "preprocessing_included_in_artifact": True,
            "prediction_interval_available": True,
            "predicted_medical_expenditure": point,
            "predicted_cost": point,
            "central_prediction": point,
            "lower_bound": interval_result["lower"],
            "upper_bound": interval_result["upper"],
            "interval_level": interval_result["coverage_level"],
            "predicted_medical_expenditure_range": interval_result,
            "predicted_cost_range": interval_result,
            "currency": "INR",
            "currency_context": "source_dataset",
            "source_dataset_target": "NSS Schedule 25.0 Block 7 item 12",
            "source_reference_id": "DDI-IND-NSO-HSCHealth80R-Jan2025-Dec2025",
            "limitation": LIMITATION,
            "range_interpretation": self.interval_description,
            "input_received": cleaned,
        }


@lru_cache(maxsize=1)
def get_cached_nss80_service() -> NSS80PredictionService:
    return NSS80PredictionService()
