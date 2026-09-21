"""Production tests for the evaluated practical-15 expenditure range."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from app.schemas.prediction import NSS80CostPredictionInput
from app.services.nss80_prediction_service import NSS80PredictionService
from model_training.nss80.practical15_intervals import bounds
from model_training.nss80.production_practical15 import FEATURES, model_safe_values
from model_training.nss80.safe_improvement_v4 import PROHIBITED_FEATURES


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/nss80/practical_15_prediction_range"
REPORT_DIR = ROOT / "reports/nss80/practical_15_prediction_range"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def service() -> NSS80PredictionService:
    return NSS80PredictionService()


@pytest.fixture(scope="module")
def nss_request() -> NSS80CostPredictionInput:
    values = json.loads(
        (ROOT / "reports/nss80/practical_15_promotion/sample_api_request.json").read_text(encoding="utf-8")
    )
    return NSS80CostPredictionInput(**values)


def test_selected_interval_estimators_load_with_exact_safe_features() -> None:
    assert len(FEATURES) == 15
    assert not set(FEATURES) & PROHIBITED_FEATURES
    bundle = joblib.load(MODEL_DIR / "practical_15_interval_80.joblib")
    for artifact, quantile in [(bundle["lower"], 0.10), (bundle["upper"], 0.90)]:
        assert artifact.features == FEATURES
        assert artifact.encoded_feature_count == 90
        assert isinstance(artifact.model_, HistGradientBoostingRegressor)
        assert artifact.spec.parameters["loss"] == "quantile"
        assert artifact.spec.parameters["quantile"] == pytest.approx(quantile)


def test_production_range_matches_selected_direct_models(service, nss_request) -> None:
    bundle_path = MODEL_DIR / "practical_15_interval_80.joblib"
    bundle = joblib.load(bundle_path)
    assert bundle["features"] == FEATURES
    assert bundle["nominal_coverage"] == pytest.approx(0.80)
    assert bundle["point_sha256"] == sha256(ROOT / "models/nss80/primary_model.joblib")

    transformed = model_safe_values(nss_request.model_dump())
    frame = pd.DataFrame([transformed], columns=FEATURES)
    raw_point = float(service.estimator.predict(frame)[0])
    point = round(max(0, raw_point), 2)
    direct_lower, direct_upper = bounds(bundle, frame, np.array([point]))
    result = service.predict(nss_request.model_dump())

    assert result["currency"] == "INR"
    assert result["interval_level"] == pytest.approx(0.80)
    assert result["central_prediction"] == pytest.approx(point)
    assert result["lower_bound"] == pytest.approx(round(float(direct_lower[0]), 2))
    assert result["upper_bound"] == pytest.approx(round(float(direct_upper[0]), 2))
    assert 0 <= result["lower_bound"] <= result["central_prediction"] <= result["upper_bound"]
    assert np.isfinite([result["central_prediction"], result["lower_bound"], result["upper_bound"]]).all()


def test_range_serialization_reload_is_exact(nss_request) -> None:
    bundle_path = MODEL_DIR / "practical_15_interval_80.joblib"
    first = joblib.load(bundle_path)
    second = joblib.load(bundle_path)
    frame = pd.DataFrame([model_safe_values(nss_request.model_dump())], columns=FEATURES)
    first_bounds = bounds(first, frame)
    second_bounds = bounds(second, frame)
    np.testing.assert_array_equal(first_bounds[0], second_bounds[0])
    np.testing.assert_array_equal(first_bounds[1], second_bounds[1])


def test_production_options_expose_selected_level(service) -> None:
    options = service.prediction_options()
    assert options["field_order"] == FEATURES
    assert options["model"]["feature_count"] == 15
    assert options["prediction_interval"]["available"]
    assert options["prediction_interval"]["nominal_coverage"] == pytest.approx(0.80)
    assert "not an individual bill guarantee" in options["prediction_interval"]["reason"]


def test_experiment_reports_locked_test_without_using_it_for_calibration() -> None:
    comparison = json.loads((REPORT_DIR / "interval_comparison.json").read_text(encoding="utf-8"))
    protocol = json.loads((REPORT_DIR / "protocol.json").read_text(encoding="utf-8"))
    assert {row["interval"] for row in comparison} == {"80%", "90%"}
    assert protocol["known_leakage_columns"] == []
    assert protocol["all_relevant_person_overlaps"] == 0
    assert not protocol["locked_test_used_for_selection_or_calibration"]
    assert not protocol["point_estimator_retrained"]
    assert protocol["point_artifact_sha256_before"] == protocol["point_artifact_sha256_after"]
    assert all(row["empirical_coverage"] > 0 and row["mean_width_inr"] > 0 for row in comparison)
