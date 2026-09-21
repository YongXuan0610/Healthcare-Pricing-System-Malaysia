"""Production contract for the frozen practical 15-input NSS estimator.

Historical v4 and compact-25 contracts remain unchanged for reproducibility.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from . import production_v4 as v4


MODEL_VERSION = "nss80_practical_15"
ARTIFACT_MODEL_VERSION = "isolated_practical_15_full_data"
ARTIFACT_TYPE = "global_safe_regressor"
TARGET = v4.TARGET

FEATURES = [
    "age_years",
    "gender",
    "chronic_ailment",
    "pregnant",
    "communicable_disease",
    "other_ailment_last_15_days",
    "number_of_hospitalisations",
    "length_of_stay_days",
    "ailment_nature",
    "hospitalisation_treatment_nature",
    "medical_institution_type",
    "ward_type",
    "place_of_hospitalisation",
    "surgery",
    "medicine",
]

NUMERIC_FEATURES = [feature for feature in FEATURES if feature in v4.NUMERIC_FEATURES]
BINARY_FEATURES = [feature for feature in FEATURES if feature in v4.BINARY_FEATURES]
NOMINAL_FEATURES = [feature for feature in FEATURES if feature in v4.NOMINAL_FEATURES]
NUMERIC_LIMITS = {feature: v4.NUMERIC_LIMITS[feature] for feature in NUMERIC_FEATURES}
OPTIONAL_FIELDS = ["pregnant"]
SERVICE_RECEIPT_FEATURES = ["surgery", "medicine"]
YES_NO_FEATURES = [feature for feature in FEATURES if feature in v4.YES_NO_FEATURES]


def canonical_category(field: str, value: object) -> str | None:
    if field in SERVICE_RECEIPT_FEATURES:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            for label in ["Received", "Not received"]:
                if value.strip().casefold() == label.casefold():
                    return label
        raise ValueError(f"{field} must be Received or Not received")
    return v4.canonical_category(field, value)


def validate_numeric(field: str, value: object) -> int | None:
    return v4.validate_numeric(field, value)


def validate_consistency(values: Mapping[str, Any]) -> None:
    age = values.get("age_years")
    pregnancy_applicable = (
        values.get("gender") == "Female"
        and isinstance(age, int)
        and 15 <= age <= 49
    )
    if pregnancy_applicable and values.get("pregnant") is None:
        raise ValueError("pregnant is required for female respondents aged 15 to 49")
    if not pregnancy_applicable and values.get("pregnant") is not None:
        raise ValueError(
            "pregnant must be null when the NSS pregnancy question is not applicable"
        )


def model_safe_values(values: Mapping[str, Any]) -> dict[str, Any]:
    output = {feature: values[feature] for feature in FEATURES}
    output["gender"] = int(values["gender"] == "Male")
    for feature in YES_NO_FEATURES:
        output[feature] = (
            np.nan if values[feature] is None else int(values[feature] == "Yes")
        )
    output["communicable_disease"] = int(
        values["communicable_disease"] != "Not suffered"
    )
    for feature in SERVICE_RECEIPT_FEATURES:
        output[feature] = int(values[feature] == "Received")
    for feature in NOMINAL_FEATURES:
        if output[feature] is None:
            output[feature] = np.nan
    return output


def categories_from_estimator(estimator: Any) -> dict[str, list[str]]:
    categories = v4.categories_from_estimator(estimator)
    categories = {
        feature: categories[feature]
        for feature in FEATURES
        if feature not in NUMERIC_FEATURES
    }
    for feature in SERVICE_RECEIPT_FEATURES:
        categories[feature] = ["Not received", "Received"]
    return categories
