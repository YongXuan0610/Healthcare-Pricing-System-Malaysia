"""Production contract for the selected NSS 80 v4 leakage-safe model.

The saved estimator consumes the decoded, human-readable NSS categories used by
the v3/v4 preparation code.  This module keeps the API vocabulary and the
model-safe binary transformations in one place so the web form never needs to
invent survey codes or expose expenditure-component leakage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .schema_v3 import (
    AILMENT,
    COMMUNICABLE_DISEASE,
    EDUCATION,
    INSURANCE_COVERAGE,
    LEVEL_OF_CARE,
    MARITAL_STATUS,
    MEDICAL_INSTITUTION,
    PLACE,
    REASON_NOT_GOVERNMENT,
    RELATION,
    RURAL_HOUSEHOLD_TYPE,
    SERVICE_RECEIPT,
    STATE,
    TREATMENT_NATURE,
    URBAN_HOUSEHOLD_TYPE,
    WARD_TYPE,
    YES_NO,
)


MODEL_VERSION = "nss80_v4"
ARTIFACT_MODEL_VERSION = "safe_improvement_v4_best_features_global"
ARTIFACT_TYPE = "global_safe_regressor"
TARGET = "total_medical_expenditure_rs"
MISSING_CATEGORY = "Missing or not applicable"

FEATURES = [
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
    "nss_region",
    "district",
    "place_of_hospitalisation",
    "treatment_state_code",
    "community_communicable_disease_outbreak",
    "pregnant",
    "communicable_disease",
    "other_ailment_last_15_days",
    "other_ailment_previous_day",
]

NUMERIC_FEATURES = [
    "household_size",
    "household_usual_consumer_expenditure_rs",
    "age_years",
    "number_of_hospitalisations",
    "length_of_stay_days",
    "medical_insurance_premium_rs",
    "pre_hospitalisation_treatment_duration_days",
]

BINARY_FEATURES = [
    "gender",
    "chronic_ailment",
    "health_financing_or_insurance_coverage",
    "surgery",
    "medicine",
    "xray_ecg_eeg_scan",
    "other_diagnostic_tests",
    "treated_on_medical_advice_before_hospitalisation",
    "community_communicable_disease_outbreak",
    "pregnant",
    "communicable_disease",
    "other_ailment_last_15_days",
    "other_ailment_previous_day",
]

NOMINAL_FEATURES = [
    feature
    for feature in FEATURES
    if feature not in set(NUMERIC_FEATURES) | set(BINARY_FEATURES)
]

SERVICE_RECEIPT_FEATURES = [
    "surgery",
    "medicine",
    "xray_ecg_eeg_scan",
    "other_diagnostic_tests",
]

YES_NO_FEATURES = [
    "chronic_ailment",
    "treated_on_medical_advice_before_hospitalisation",
    "community_communicable_disease_outbreak",
    "pregnant",
    "other_ailment_last_15_days",
    "other_ailment_previous_day",
]

OPTIONAL_FIELDS = [
    "reason_not_using_government_public_hospital",
    "treated_on_medical_advice_before_hospitalisation",
    "pre_hospitalisation_treatment_nature",
    "pre_hospitalisation_level_of_care",
    "pre_hospitalisation_treatment_duration_days",
    "treatment_state_code",
    "pregnant",
    "other_ailment_previous_day",
]

# These are the observed v4 modeling ranges, not invented clinical thresholds.
NUMERIC_LIMITS: dict[str, dict[str, int]] = {
    "household_size": {"minimum": 1, "maximum": 30, "step": 1},
    "household_usual_consumer_expenditure_rs": {
        "minimum": 580,
        "maximum": 280_773,
        "step": 1,
    },
    "age_years": {"minimum": 0, "maximum": 120, "step": 1},
    "number_of_hospitalisations": {"minimum": 1, "maximum": 24, "step": 1},
    "length_of_stay_days": {"minimum": 1, "maximum": 365, "step": 1},
    "medical_insurance_premium_rs": {"minimum": 0, "maximum": 216_000, "step": 1},
    "pre_hospitalisation_treatment_duration_days": {
        "minimum": 1,
        "maximum": 999,
        "step": 1,
    },
}


def _values(mapping: Mapping[str, str]) -> list[str]:
    return list(dict.fromkeys(mapping.values()))


STATIC_CATEGORIES: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "chronic_ailment": _values(YES_NO),
    "health_financing_or_insurance_coverage": _values(INSURANCE_COVERAGE),
    **{feature: _values(SERVICE_RECEIPT) for feature in SERVICE_RECEIPT_FEATURES},
    "sector": ["Rural", "Urban"],
    "state": _values(STATE),
    "ailment_nature": _values(AILMENT),
    "hospitalisation_treatment_nature": _values(TREATMENT_NATURE),
    "medical_institution_type": _values(MEDICAL_INSTITUTION),
    "ward_type": _values(WARD_TYPE),
    "household_type": list(
        dict.fromkeys(
            [*RURAL_HOUSEHOLD_TYPE.values(), *URBAN_HOUSEHOLD_TYPE.values()]
        )
    ),
    "relation_to_household_head": _values(RELATION),
    "marital_status": _values(MARITAL_STATUS),
    "highest_education_level": _values(EDUCATION),
    "reason_not_using_government_public_hospital": _values(REASON_NOT_GOVERNMENT),
    "treated_on_medical_advice_before_hospitalisation": _values(YES_NO),
    "pre_hospitalisation_treatment_nature": _values(TREATMENT_NATURE),
    "pre_hospitalisation_level_of_care": _values(LEVEL_OF_CARE),
    "place_of_hospitalisation": _values(PLACE),
    "treatment_state_code": _values(STATE),
    "community_communicable_disease_outbreak": _values(YES_NO),
    "pregnant": _values(YES_NO),
    "communicable_disease": _values(COMMUNICABLE_DISEASE),
    "other_ailment_last_15_days": _values(YES_NO),
    "other_ailment_previous_day": _values(YES_NO),
}


def canonical_category(field: str, value: object) -> str | None:
    """Canonicalize an API category using only the official decoded vocabulary."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    stripped = value.strip()
    if not stripped:
        return None
    choices = STATIC_CATEGORIES.get(field)
    if choices is None:
        return stripped
    lookup = {choice.casefold(): choice for choice in choices}
    canonical = lookup.get(stripped.casefold())
    if canonical is None:
        raise ValueError(f"Invalid {field}; expected one of: {', '.join(choices)}")
    return canonical


def validate_numeric(field: str, value: object) -> int | None:
    """Validate a source-survey whole-number value against the observed v4 range."""

    if value is None:
        if field in OPTIONAL_FIELDS:
            return None
        raise ValueError(f"{field} is required")
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{field} must be numeric")
    numeric = float(value)
    limit = NUMERIC_LIMITS[field]
    if not np.isfinite(numeric) or not limit["minimum"] <= numeric <= limit["maximum"]:
        raise ValueError(
            f"{field} must be between {limit['minimum']} and {limit['maximum']}"
        )
    if not numeric.is_integer():
        raise ValueError(f"{field} must be a whole number")
    return int(numeric)


def validate_consistency(values: Mapping[str, Any]) -> None:
    """Reject internally contradictory conditional NSS answers."""

    sector = values.get("sector")
    household_type = values.get("household_type")
    household_choices = (
        set(RURAL_HOUSEHOLD_TYPE.values())
        if sector == "Rural"
        else set(URBAN_HOUSEHOLD_TYPE.values())
    )
    if household_type not in household_choices:
        raise ValueError(f"household_type is not valid for the selected {sector} sector")

    institution = values.get("medical_institution_type")
    reason = values.get("reason_not_using_government_public_hospital")
    if institution == "Government or public hospital" and reason is not None:
        raise ValueError(
            "reason_not_using_government_public_hospital must be null for a government or public hospital"
        )
    if institution != "Government or public hospital" and reason is None:
        raise ValueError(
            "reason_not_using_government_public_hospital is required for a non-government hospital"
        )

    treated = values.get("treated_on_medical_advice_before_hospitalisation")
    pre_fields = (
        values.get("pre_hospitalisation_treatment_nature"),
        values.get("pre_hospitalisation_level_of_care"),
        values.get("pre_hospitalisation_treatment_duration_days"),
    )
    if treated == "Yes" and any(value is None for value in pre_fields):
        raise ValueError(
            "pre-hospitalisation treatment nature, level of care, and duration are required when prior treatment was received"
        )
    if treated != "Yes" and any(value is not None for value in pre_fields):
        raise ValueError(
            "pre-hospitalisation treatment details must be null unless prior treatment was received"
        )

    place = values.get("place_of_hospitalisation")
    treatment_state = values.get("treatment_state_code")
    if place == "Other state":
        if treatment_state is None:
            raise ValueError("treatment_state_code is required when hospitalisation was in another state")
        if treatment_state == values.get("state"):
            raise ValueError("treatment_state_code must differ from state for an other-state hospitalisation")
    elif treatment_state is not None:
        raise ValueError("treatment_state_code must be null unless hospitalisation was in another state")

    gender = values.get("gender")
    age = values.get("age_years")
    pregnant = values.get("pregnant")
    pregnancy_answer_required = gender == "Female" and isinstance(age, int) and 15 <= age <= 49
    if pregnancy_answer_required and pregnant is None:
        raise ValueError("pregnant is required for female respondents aged 15 to 49")
    if not pregnancy_answer_required and pregnant is not None:
        raise ValueError("pregnant must be null when the NSS pregnancy question is not applicable")

    recent_ailment = values.get("other_ailment_last_15_days")
    previous_day = values.get("other_ailment_previous_day")
    if recent_ailment == "Yes" and previous_day is None:
        raise ValueError("other_ailment_previous_day is required when another ailment was reported")
    if recent_ailment != "Yes" and previous_day is not None:
        raise ValueError(
            "other_ailment_previous_day must be null when no other recent ailment was reported"
        )

    state = values.get("state")
    district = values.get("district")
    if isinstance(district, str) and not district.startswith(f"{state} - district code "):
        raise ValueError("district is not valid for the selected state")

    region = values.get("nss_region")
    if isinstance(region, str):
        code = region.removeprefix("NSS region code ")
        state_code = code[:2]
        if STATE.get(state_code) != state:
            raise ValueError("nss_region is not valid for the selected state")


def model_safe_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """Convert official labels to the exact 0/1 representation fitted by v4."""

    output = {feature: values.get(feature) for feature in FEATURES}
    output["gender"] = None if values.get("gender") is None else int(values["gender"] == "Male")
    for field in YES_NO_FEATURES:
        output[field] = None if values.get(field) is None else int(values[field] == "Yes")
    output["health_financing_or_insurance_coverage"] = int(
        values["health_financing_or_insurance_coverage"] != "Not covered"
    )
    output["communicable_disease"] = int(values["communicable_disease"] != "Not suffered")
    for field in SERVICE_RECEIPT_FEATURES:
        # Official source 1 -> 0; source categories 2/3/4 -> 1.  Payment detail
        # is deliberately collapsed before the estimator sees the feature.
        output[field] = int(values[field] != "Not received")
    return output


def categories_from_estimator(estimator: Any) -> dict[str, list[str]]:
    """Return fitted category choices plus safe display labels for binary inputs."""

    categories = {key: list(value) for key, value in STATIC_CATEGORIES.items()}
    preprocessor = getattr(estimator, "preprocessor_", None)
    if preprocessor is None:
        raise ValueError("The v4 production estimator does not contain fitted preprocessing")
    nominal_pipeline = preprocessor.named_transformers_["nominal"]
    encoder = nominal_pipeline.named_steps["onehot"]
    fitted = {
        field: [str(value) for value in choices if str(value) != MISSING_CATEGORY]
        for field, choices in zip(estimator.nominal, encoder.categories_, strict=True)
    }
    categories.update(fitted)
    categories["state"] = [
        state for state in categories["state"] if state != "Unverified state code 99"
    ]
    return categories


def dependent_options(categories: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    """Build state/sector subsets used to keep the long browser form understandable."""

    districts_by_state = {
        state: [
            district
            for district in categories["district"]
            if district.startswith(f"{state} - district code ")
        ]
        for state in categories["state"]
    }
    state_code_by_label = {label: code for code, label in STATE.items()}
    regions_by_state = {
        state: [
            region
            for region in categories["nss_region"]
            if region.removeprefix("NSS region code ").startswith(state_code_by_label[state])
        ]
        for state in categories["state"]
        if state in state_code_by_label
    }
    return {
        "household_type_by_sector": {
            "Rural": list(RURAL_HOUSEHOLD_TYPE.values()),
            "Urban": list(URBAN_HOUSEHOLD_TYPE.values()),
        },
        "districts_by_state": districts_by_state,
        "nss_regions_by_state": regions_by_state,
    }
