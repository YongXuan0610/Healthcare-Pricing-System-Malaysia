"""Production contract for the frozen compact 25-input NSS estimator.

Historical v4 contracts remain unchanged for reproducible research artifacts.
"""
from typing import Any
from collections.abc import Mapping
import numpy as np
from . import production_v4 as v4

MODEL_VERSION = 'nss80_compact_25'
ARTIFACT_MODEL_VERSION = 'isolated_full_data_25'
ARTIFACT_TYPE = 'global_safe_regressor'
TARGET = v4.TARGET
FEATURES = [
    'age_years', 'gender', 'relation_to_household_head', 'marital_status',
    'highest_education_level', 'chronic_ailment', 'health_financing_or_insurance_coverage',
    'medical_insurance_premium_rs', 'pregnant', 'communicable_disease',
    'other_ailment_last_15_days', 'other_ailment_previous_day',
    'number_of_hospitalisations', 'length_of_stay_days', 'ailment_nature',
    'hospitalisation_treatment_nature', 'medical_institution_type',
    'reason_not_using_government_public_hospital', 'ward_type', 'place_of_hospitalisation',
    'treatment_state_code', 'surgery', 'medicine', 'xray_ecg_eeg_scan', 'other_diagnostic_tests',
]
NUMERIC_FEATURES = [f for f in FEATURES if f in v4.NUMERIC_FEATURES]
BINARY_FEATURES = [f for f in FEATURES if f in v4.BINARY_FEATURES]
NOMINAL_FEATURES = [f for f in FEATURES if f in v4.NOMINAL_FEATURES]
NUMERIC_LIMITS = {f: v4.NUMERIC_LIMITS[f] for f in NUMERIC_FEATURES}
OPTIONAL_FIELDS = [f for f in FEATURES if f in v4.OPTIONAL_FIELDS]
SERVICE_RECEIPT_FEATURES = v4.SERVICE_RECEIPT_FEATURES
YES_NO_FEATURES = [f for f in FEATURES if f in v4.YES_NO_FEATURES]


def canonical_category(field: str, value: object) -> str | None:
    if field in SERVICE_RECEIPT_FEATURES:
        if value is None or value == '':
            return None
        if isinstance(value, str):
            for label in ['Received', 'Not received']:
                if value.strip().casefold() == label.casefold():
                    return label
        raise ValueError(f'{field} must be Received or Not received')
    return v4.canonical_category(field, value)


def validate_numeric(field: str, value: object) -> int | None:
    return v4.validate_numeric(field, value)


def validate_consistency(values: Mapping[str, Any]) -> None:
    institution = values.get('medical_institution_type')
    reason = values.get('reason_not_using_government_public_hospital')
    if institution == 'Government or public hospital' and reason is not None:
        raise ValueError('reason_not_using_government_public_hospital must be null for a government or public hospital')
    if institution != 'Government or public hospital' and reason is None:
        raise ValueError('reason_not_using_government_public_hospital is required for a non-government hospital')
    other_state = values.get('place_of_hospitalisation') == 'Other state'
    if other_state and values.get('treatment_state_code') is None:
        raise ValueError('treatment_state_code is required when hospitalisation was in another state')
    if not other_state and values.get('treatment_state_code') is not None:
        raise ValueError('treatment_state_code must be null unless hospitalisation was in another state')
    age = values.get('age_years')
    applicable = values.get('gender') == 'Female' and isinstance(age, int) and 15 <= age <= 49
    if applicable and values.get('pregnant') is None:
        raise ValueError('pregnant is required for female respondents aged 15 to 49')
    if not applicable and values.get('pregnant') is not None:
        raise ValueError('pregnant must be null when the NSS pregnancy question is not applicable')
    recent = values.get('other_ailment_last_15_days') == 'Yes'
    if recent and values.get('other_ailment_previous_day') is None:
        raise ValueError('other_ailment_previous_day is required when another ailment was reported')
    if not recent and values.get('other_ailment_previous_day') is not None:
        raise ValueError('other_ailment_previous_day must be null when no other recent ailment was reported')


def model_safe_values(values: Mapping[str, Any]) -> dict[str, Any]:
    output = {f: values[f] for f in FEATURES}
    output['gender'] = int(values['gender'] == 'Male')
    for f in YES_NO_FEATURES:
        output[f] = np.nan if values[f] is None else int(values[f] == 'Yes')
    output['health_financing_or_insurance_coverage'] = int(values['health_financing_or_insurance_coverage'] != 'Not covered')
    output['communicable_disease'] = int(values['communicable_disease'] != 'Not suffered')
    for f in SERVICE_RECEIPT_FEATURES:
        output[f] = int(values[f] == 'Received')
    for f in NOMINAL_FEATURES:
        if output[f] is None:
            output[f] = np.nan
    return output


def categories_from_estimator(estimator: Any) -> dict[str, list[str]]:
    categories = v4.categories_from_estimator(estimator)
    categories = {f: categories[f] for f in FEATURES if f not in NUMERIC_FEATURES}
    for f in SERVICE_RECEIPT_FEATURES:
        categories[f] = ['Not received', 'Received']
    return categories
