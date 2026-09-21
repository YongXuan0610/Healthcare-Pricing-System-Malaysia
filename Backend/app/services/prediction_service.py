"""Secondary US Kaggle benchmark service (never used for NSS or Malaysia)."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]
VALID_CATEGORIES = {
    "sex": {"female", "male"},
    "smoker": {"no", "yes"},
    "region": {"northeast", "northwest", "southeast", "southwest"},
}
LIMITATION = (
    "Secondary benchmark only: prediction uses the scale and context of the supplied "
    "US Kaggle medical-charge dataset. It is not an NSS estimate, a Malaysian Ringgit "
    "estimate, or an exact hospital bill."
)


class USKaggleBenchmarkService:
    def __init__(
        self,
        model_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ) -> None:
        backend_root = Path(__file__).resolve().parents[2]
        self.model_path = (
            Path(model_path)
            if model_path
            else backend_root / "models" / "us_kaggle" / "healthcare_cost_model.joblib"
        )
        self.metadata_path = (
            Path(metadata_path)
            if metadata_path
            else self.model_path.with_name("model_metadata.json")
        )
        self.interval_models_path = self.model_path.with_name(
            "healthcare_cost_interval_models.joblib"
        )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained pipeline not found at {self.model_path}. Run model_training/train_model.py first."
            )
        self.pipeline = joblib.load(self.model_path)
        if not self.interval_models_path.exists():
            raise FileNotFoundError(
                f"Prediction interval models not found at {self.interval_models_path}"
            )
        interval_models = joblib.load(self.interval_models_path)
        self.lower_pipeline = interval_models["lower"]
        self.upper_pipeline = interval_models["upper"]
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Model metadata not found at {self.metadata_path}")
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        interval = metadata.get("prediction_interval")
        if not interval:
            raise ValueError(
                "Model metadata has no calibrated prediction interval. Retrain the model."
            )
        self.interval = interval

    @staticmethod
    def validate_input(values: dict[str, Any]) -> dict[str, Any]:
        missing = sorted(set(FEATURES).difference(values))
        extra = sorted(set(values).difference(FEATURES))
        if missing or extra:
            raise ValueError(f"Expected exactly {FEATURES}; missing={missing}, extra={extra}")

        cleaned = dict(values)
        for field in VALID_CATEGORIES:
            if not isinstance(cleaned[field], str):
                raise ValueError(f"{field} must be a string")
            cleaned[field] = cleaned[field].strip().lower()
            if cleaned[field] not in VALID_CATEGORIES[field]:
                allowed = ", ".join(sorted(VALID_CATEGORIES[field]))
                raise ValueError(f"Invalid {field!r}; expected one of: {allowed}")

        limits = {"age": (18, 64), "bmi": (15.96, 53.13), "children": (0, 5)}
        for field, (minimum, maximum) in limits.items():
            value = cleaned[field]
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f"{field} must be numeric")
            if not np.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{field} must be between {minimum} and {maximum}")
        if int(cleaned["age"]) != cleaned["age"] or int(cleaned["children"]) != cleaned["children"]:
            raise ValueError("age and children must be whole numbers")
        cleaned["age"] = int(cleaned["age"])
        cleaned["children"] = int(cleaned["children"])
        cleaned["bmi"] = float(cleaned["bmi"])
        return cleaned

    def predict(self, values: dict[str, Any]) -> dict[str, Any]:
        cleaned = self.validate_input(values)
        prediction = float(self.pipeline.predict(pd.DataFrame([cleaned], columns=FEATURES))[0])
        if not np.isfinite(prediction):
            raise RuntimeError("Model produced a non-finite prediction")
        prediction = round(max(0.0, prediction), 2)
        input_frame = pd.DataFrame([cleaned], columns=FEATURES)
        raw_lower = float(self.lower_pipeline.predict(input_frame)[0])
        raw_upper = float(self.upper_pipeline.predict(input_frame)[0])
        smoker_adjustments = self.interval["conformal_adjustment_by_smoker"]
        adjustment = float(
            smoker_adjustments.get(
                cleaned["smoker"], self.interval["global_conformal_adjustment"]
            )
        )
        range_lower = round(max(0.0, raw_lower - adjustment), 2)
        range_upper = round(max(range_lower, raw_upper + adjustment), 2)
        prediction_range = {
            "lower": range_lower,
            "upper": range_upper,
            "coverage_level": float(self.interval["nominal_coverage"]),
            "method": self.interval["method"],
            "currency": "USD",
        }
        return {
            "component": "us_kaggle_secondary_benchmark",
            "research_role": "secondary_benchmark",
            "predicted_charge": prediction,
            "predicted_cost": prediction,
            "predicted_charge_range": prediction_range,
            "predicted_cost_range": prediction_range,
            "currency": "USD",
            "currency_context": "source_dataset",
            "source_dataset_target": "charges_usd",
            "limitation": LIMITATION,
            "range_interpretation": self.interval["interpretation"],
            "input_received": cleaned,
        }


HealthcareCostPredictionService = USKaggleBenchmarkService


@lru_cache(maxsize=4)
def _cached_service(model_path: str | None) -> USKaggleBenchmarkService:
    return USKaggleBenchmarkService(model_path=model_path)


def predict_healthcare_charge(
    *,
    age: int,
    sex: str,
    bmi: float,
    children: int,
    smoker: str,
    region: str,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Standalone validated prediction function for scripts and integration tests."""

    resolved_path = str(Path(model_path).resolve()) if model_path is not None else None
    service = _cached_service(resolved_path)
    return service.predict(
        {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
        }
    )


predict_us_kaggle_benchmark = predict_healthcare_charge
