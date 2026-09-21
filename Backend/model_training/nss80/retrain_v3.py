"""Reproducible NSS 80 L1+L2+L4 leakage comparison.

This module creates a readable episode table, filters the modelling population
to officially coded Male/Female records, makes one person-grouped split, and
fits a leakage-safe model plus a clearly labelled leakage-included experiment.
It never overwrites the production NSS model.

Run from ``Backend``::

    .\.venv\Scripts\python.exe -m model_training.nss80.retrain_v3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Callable, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .evaluate import regression_metrics
from .schema_v3 import (
    CATEGORY_MAPPINGS,
    COLUMN_DEFINITIONS,
    GENDER,
    HOUSEHOLD_KEY,
    OFFICIAL_SOURCE,
    RURAL_HOUSEHOLD_TYPE,
    URBAN_HOUSEHOLD_TYPE,
    column_mapping_rows,
    decode_communicable_binary,
    decode_coverage_binary,
    decode_series,
    decode_service_received_binary,
    decode_yes_no_binary,
    normalize_code,
    value_mapping_rows,
)


RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_TRAIN = 0.20
TARGET = "total_medical_expenditure_rs"
GENDER_LIMITATION = (
    "The NSS model was trained and evaluated using records coded as Male or "
    "Female in the source survey. Other gender-category records were preserved "
    "in the source data but excluded from the defined modelling population. "
    "Therefore, the model was not validated for excluded gender groups."
)
LEAKAGE_WARNING = (
    "This model includes variables that are observed during or after the "
    "expenditure event and/or variables that directly contribute to total "
    "medical expenditure. Its metrics are expected to be optimistic and are "
    "shown only to demonstrate the effect of target leakage. It is not suitable "
    "as the final predictive model."
)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = BACKEND_ROOT / "datasets" / "nss80" / "raw"
DOCUMENTATION_PATH = (
    BACKEND_ROOT / "datasets" / "nss80" / "documentation" / "nss80_health_schedule.pdf"
)
REPORT_DIR = BACKEND_ROOT / "reports" / "nss80" / "retraining_v3"
READABLE_LEVEL_DIR = BACKEND_ROOT / "reports" / "nss80" / "processed"
MODEL_DIR = BACKEND_ROOT / "models" / "nss80" / "retraining_v3"
PRODUCTION_MODEL_DIR = BACKEND_ROOT / "models" / "nss80"


CORE_NUMERIC_FEATURES = [
    "household_size",
    "household_usual_consumer_expenditure_rs",
    "age_years",
    "number_of_hospitalisations",
    "length_of_stay_days",
]
CORE_BINARY_FEATURES = [
    "gender",
    "chronic_ailment",
    "health_financing_or_insurance_coverage",
    "surgery",
    "medicine",
    "xray_ecg_eeg_scan",
    "other_diagnostic_tests",
]
CORE_NOMINAL_FEATURES = [
    "sector",
    "state",
    "ailment_nature",
    "hospitalisation_treatment_nature",
    "medical_institution_type",
    "ward_type",
]
CORE_FEATURES = (
    CORE_NUMERIC_FEATURES + CORE_BINARY_FEATURES + CORE_NOMINAL_FEATURES
)

# Approximation of the deployed v2 scientific feature set on the new cleaned
# population.  It is retained as ablation A, not as production output.
CURRENT_SAFE_FEATURES = [
    "age_years",
    "length_of_stay_days",
    "gender",
    "sector",
    "state",
    "ailment_nature",
    "hospitalisation_treatment_nature",
    "medical_institution_type",
    "ward_class",
    "surgery",
    "medicine",
    "xray_ecg_eeg_scan",
    "other_diagnostic_tests",
]

SOCIOECONOMIC_FEATURES = [
    "household_type",
    "medical_insurance_premium_rs",
    "relation_to_household_head",
    "marital_status",
    "highest_education_level",
]
ADDITIONAL_HEALTH_FEATURES = [
    "pregnant",
    "communicable_disease",
    "other_ailment_last_15_days",
    "other_ailment_previous_day",
]
PRE_HOSPITALISATION_FEATURES = [
    "reason_not_using_government_public_hospital",
    "treated_on_medical_advice_before_hospitalisation",
    "pre_hospitalisation_treatment_nature",
    "pre_hospitalisation_level_of_care",
    "pre_hospitalisation_treatment_duration_days",
]
EXTRA_GEOGRAPHY_FEATURES = [
    "nss_region",
    "district",
    "place_of_hospitalisation",
    "treatment_state_code",
]

LEAKAGE_FEATURES = [
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
]

NUMERIC_FEATURES = set(CORE_NUMERIC_FEATURES) | {
    "medical_insurance_premium_rs",
    "pre_hospitalisation_treatment_duration_days",
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
    "household_income_loss_due_to_hospitalisation_rs",
}
BINARY_FEATURES = set(CORE_BINARY_FEATURES) | {
    "pregnant",
    "communicable_disease",
    "other_ailment_last_15_days",
    "other_ailment_previous_day",
    "treated_on_medical_advice_before_hospitalisation",
}
NOMINAL_FEATURES = set(CORE_NOMINAL_FEATURES) | {
    "ward_class",
    "household_type",
    "relation_to_household_head",
    "marital_status",
    "highest_education_level",
    "reason_not_using_government_public_hospital",
    "pre_hospitalisation_treatment_nature",
    "pre_hospitalisation_level_of_care",
    "nss_region",
    "district",
    "place_of_hospitalisation",
    "treatment_state_code",
    "medical_service_free_fully_or_partly",
    "major_source_of_finance",
}

FEATURE_SOURCE_CODES = {
    "sector": "sec",
    "state": "st",
    "nss_region": "nssreg",
    "district": "dist",
    "household_size": "hhsz",
    "household_type": "b5i4",
    "medical_insurance_premium_rs": "b5i6",
    "household_usual_consumer_expenditure_rs": "umce",
    "relation_to_household_head": "b3c3",
    "gender": "b3c4",
    "age_years": "b3c5/b6i3",
    "marital_status": "b3c6",
    "highest_education_level": "b3c7",
    "number_of_hospitalisations": "b3c10",
    "pregnant": "b3c11",
    "communicable_disease": "b3c13",
    "chronic_ailment": "b3c14",
    "other_ailment_last_15_days": "b3c15",
    "other_ailment_previous_day": "b3c16",
    "health_financing_or_insurance_coverage": "b3c17",
    "ailment_nature": "b6i5",
    "hospitalisation_treatment_nature": "b6i6",
    "medical_institution_type": "b6i7",
    "reason_not_using_government_public_hospital": "b6i8",
    "ward_type": "b6i9",
    "ward_class": "derived from b6i9",
    "length_of_stay_days": "b6i12",
    "surgery": "b6i13",
    "medicine": "b6i14",
    "xray_ecg_eeg_scan": "b6i15",
    "other_diagnostic_tests": "b6i16",
    "treated_on_medical_advice_before_hospitalisation": "b6i17",
    "pre_hospitalisation_treatment_nature": "b6i18",
    "pre_hospitalisation_level_of_care": "b6i19",
    "pre_hospitalisation_treatment_duration_days": "b6i20",
    "medical_service_free_fully_or_partly": "b7i5",
    "package_component_rs": "b7i6",
    "doctor_surgeon_fee_rs": "b7i7",
    "medicines_rs": "b7i8",
    "diagnostic_tests_rs": "b7i9",
    "bed_charges_rs": "b7i10",
    "other_medical_expenses_rs": "b7i11",
    TARGET: "b7i12",
    "patient_transport_rs": "b7i13",
    "other_non_medical_household_expenses_rs": "b7i14",
    "total_expenditure_rs": "b7i15",
    "insurance_or_employer_reimbursement_rs": "b7i16",
    "major_source_of_finance": "b7i17",
    "place_of_hospitalisation": "b7i18",
    "treatment_state_code": "b7i19",
    "household_income_loss_due_to_hospitalisation_rs": "b7i20",
}

KNOWN_LEAKAGE = set(LEAKAGE_FEATURES) | {TARGET}
SPECIAL_CODE_PATTERN = r"^800[1-9]$"


@dataclass
class PreparedData:
    readable: pd.DataFrame
    modelling: pd.DataFrame
    audit: dict[str, Any]
    raw_headers: dict[str, list[str]]


@dataclass(frozen=True)
class ModelCandidate:
    name: str
    target_transform: str
    factory: Callable[[], Any]
    parameters: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_level(level: int, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    path = RAW_DIR / f"hhscsL{level}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, usecols=usecols, low_memory=False)


def _numeric(series: pd.Series, field: str, *, required: bool = False) -> pd.Series:
    text = series.astype("string").str.strip()
    output = pd.to_numeric(text, errors="coerce")
    invalid = text.notna() & text.ne("") & output.isna()
    if invalid.any():
        raise ValueError(
            f"Invalid numeric value(s) in {field}: "
            f"{sorted(text.loc[invalid].unique().tolist())[:20]}"
        )
    if required and output.isna().any():
        raise ValueError(f"Required numeric field {field} contains missing values")
    return output.astype(float)


def _id(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame[columns].fillna("<missing>").astype(str).agg("|".join, axis=1)


def _decoded_or_missing(source_code: str, series: pd.Series, **kwargs: Any) -> pd.Series:
    return decode_series(source_code, series, **kwargs).where(lambda values: values.notna(), np.nan)


def prepare_hospitalisation_data() -> PreparedData:
    """Join L1+L2+L4 and preserve a readable all-gender episode table."""

    raw_headers = {
        f"L{level}": list(_read_level(level, usecols=None).columns)
        for level in range(1, 8)
    }
    l1 = _read_level(1)
    l2 = _read_level(2)
    l3 = _read_level(3, usecols=HOUSEHOLD_KEY + ["suno", "b4c1"])
    l4 = _read_level(4)

    if l1.duplicated(HOUSEHOLD_KEY).any():
        raise ValueError("L1 contains duplicate verified household keys")
    if l2.duplicated(HOUSEHOLD_KEY + ["b3c1"]).any():
        raise ValueError("L2 contains duplicate verified household/person keys")
    if l3.duplicated(HOUSEHOLD_KEY + ["b4c1"]).any():
        raise ValueError("L3 contains duplicate verified household/person keys")

    l2_suno_alignment = l4[HOUSEHOLD_KEY + ["b6i2", "suno"]].merge(
        l2[HOUSEHOLD_KEY + ["b3c1", "suno"]],
        left_on=HOUSEHOLD_KEY + ["b6i2"],
        right_on=HOUSEHOLD_KEY + ["b3c1"],
        how="left",
        validate="many_to_one",
        indicator="suno_person_join_status",
        suffixes=("_l4", "_l2"),
    )
    l2_suno_comparable = l2_suno_alignment["suno_person_join_status"].eq("both")
    l2_suno_mismatch = l2_suno_comparable & l2_suno_alignment[
        "suno_l4"
    ].fillna("<missing>").ne(l2_suno_alignment["suno_l2"].fillna("<missing>"))

    l2_columns = HOUSEHOLD_KEY + [
        "b3c1",
        "b3c3",
        "b3c4",
        "b3c5",
        "b3c6",
        "b3c7",
        "b3c10",
        "b3c11",
        "b3c13",
        "b3c14",
        "b3c15",
        "b3c16",
        "b3c17",
    ]
    l1_columns = HOUSEHOLD_KEY + ["hhsz", "b5i4", "b5i6", "umce"]
    joined_l2 = l4.merge(
        l2[l2_columns],
        left_on=HOUSEHOLD_KEY + ["b6i2"],
        right_on=HOUSEHOLD_KEY + ["b3c1"],
        how="left",
        validate="many_to_one",
        indicator="person_join_status",
    )
    person_unmatched = joined_l2["person_join_status"].ne("both")
    person_unmatched_codes = (
        joined_l2.loc[person_unmatched, "b6i2"].value_counts().to_dict()
    )
    unmatched_l4 = joined_l2.loc[person_unmatched, l4.columns].copy()
    deceased_reconciliation = unmatched_l4.merge(
        l3,
        left_on=HOUSEHOLD_KEY + ["b6i2"],
        right_on=HOUSEHOLD_KEY + ["b4c1"],
        how="left",
        validate="many_to_one",
        indicator="deceased_join_status",
        suffixes=("_l4", "_l3"),
    )
    deceased_reconciled = deceased_reconciliation["deceased_join_status"].eq("both")
    if int(deceased_reconciled.sum()) != int(person_unmatched.sum()):
        raise ValueError(
            "One or more L4 episodes missing from L2 could not be verified against the L3 deceased roster"
        )
    living_person_episodes = joined_l2.loc[~person_unmatched].copy()

    l1_suno_alignment = living_person_episodes[
        HOUSEHOLD_KEY + ["suno"]
    ].merge(
        l1[HOUSEHOLD_KEY + ["suno"]],
        on=HOUSEHOLD_KEY,
        how="left",
        validate="many_to_one",
        indicator="suno_household_join_status",
        suffixes=("_l4", "_l1"),
    )
    if l1_suno_alignment["suno_household_join_status"].ne("both").any():
        raise ValueError("Verified living-person episode has no L1 household match")
    l1_suno_mismatch = l1_suno_alignment["suno_l4"].fillna("<missing>").ne(
        l1_suno_alignment["suno_l1"].fillna("<missing>")
    )
    joined = living_person_episodes.merge(
        l1[l1_columns],
        on=HOUSEHOLD_KEY,
        how="left",
        validate="many_to_one",
        indicator="household_join_status",
    )
    household_unmatched = joined["household_join_status"].ne("both")
    if household_unmatched.any():
        raise ValueError("Verified living-person episode has no L1 household match")
    joined = joined.loc[~household_unmatched].copy()

    case_id = _id(joined, HOUSEHOLD_KEY + ["b6i1"])
    person_id = _id(joined, HOUSEHOLD_KEY + ["b3c1"])
    household_id = _id(joined, HOUSEHOLD_KEY)
    if case_id.duplicated().any():
        raise ValueError("Joined data contains duplicate hospitalisation record IDs")

    episode_age = _numeric(joined["b6i3"], "b6i3", required=True)
    roster_age = _numeric(joined["b3c5"], "b3c5", required=True)
    age_mismatch = episode_age.ne(roster_age)
    if age_mismatch.any():
        raise ValueError(
            f"Age mismatch between L2 and L4 for {int(age_mismatch.sum())} rows"
        )

    readable = pd.DataFrame(index=joined.index)
    readable["hospitalisation_record_id"] = case_id
    readable["person_group_id"] = person_id
    readable["household_id"] = household_id
    readable["fsu_serial_no"] = joined["fsu"]
    readable["sample_household_no"] = joined["hhd"]
    readable["person_serial_no"] = joined["b3c1"]
    readable["hospitalisation_case_serial_no"] = joined["b6i1"]
    readable["sector"] = _decoded_or_missing("sec", joined["sec"])
    readable["state"] = _decoded_or_missing("st", joined["st"])
    readable["nss_region"] = _decoded_or_missing("nssreg", joined["nssreg"])
    district_code = joined["dist"].map(normalize_code)
    readable["district"] = readable["state"].astype("string") + " - district code " + district_code.astype("string")
    readable["household_size"] = _numeric(joined["hhsz"], "hhsz", required=True)
    readable["household_type"] = _decoded_or_missing(
        "b5i4", joined["b5i4"], sector=joined["sec"]
    )
    readable["medical_insurance_premium_rs"] = _numeric(joined["b5i6"], "b5i6")
    readable["household_usual_consumer_expenditure_rs"] = _numeric(
        joined["umce"], "umce", required=True
    )
    readable["relation_to_household_head"] = _decoded_or_missing("b3c3", joined["b3c3"])
    readable["gender"] = _decoded_or_missing(
        "b3c4", joined["b3c4"], strict=False
    )
    readable["age_years"] = episode_age
    readable["marital_status"] = _decoded_or_missing("b3c6", joined["b3c6"])
    readable["highest_education_level"] = _decoded_or_missing("b3c7", joined["b3c7"])
    readable["number_of_hospitalisations"] = _numeric(joined["b3c10"], "b3c10")
    readable["pregnant"] = _decoded_or_missing("b3c11", joined["b3c11"])
    readable["communicable_disease"] = _decoded_or_missing("b3c13", joined["b3c13"])
    readable["chronic_ailment"] = _decoded_or_missing("b3c14", joined["b3c14"])
    readable["other_ailment_last_15_days"] = _decoded_or_missing("b3c15", joined["b3c15"])
    readable["other_ailment_previous_day"] = _decoded_or_missing("b3c16", joined["b3c16"])
    readable["health_financing_or_insurance_coverage"] = _decoded_or_missing("b3c17", joined["b3c17"])
    readable["ailment_nature"] = _decoded_or_missing("b6i5", joined["b6i5"])
    readable["hospitalisation_treatment_nature"] = _decoded_or_missing("b6i6", joined["b6i6"])
    readable["medical_institution_type"] = _decoded_or_missing("b6i7", joined["b6i7"])
    readable["reason_not_using_government_public_hospital"] = _decoded_or_missing("b6i8", joined["b6i8"])
    readable["ward_type"] = _decoded_or_missing("b6i9", joined["b6i9"])
    readable["ward_class"] = readable["ward_type"].map(
        {
            "Free ward": "Standard or free ward",
            "Paying general ward": "Standard or free ward",
            "Paying special ward": "Special ward",
        }
    )
    readable["length_of_stay_days"] = _numeric(joined["b6i12"], "b6i12", required=True)
    for output, source in {
        "surgery": "b6i13",
        "medicine": "b6i14",
        "xray_ecg_eeg_scan": "b6i15",
        "other_diagnostic_tests": "b6i16",
    }.items():
        readable[output] = _decoded_or_missing(source, joined[source])
    readable["treated_on_medical_advice_before_hospitalisation"] = _decoded_or_missing(
        "b6i17", joined["b6i17"]
    )
    readable["pre_hospitalisation_treatment_nature"] = _decoded_or_missing("b6i18", joined["b6i18"])
    readable["pre_hospitalisation_level_of_care"] = _decoded_or_missing("b6i19", joined["b6i19"])
    readable["pre_hospitalisation_treatment_duration_days"] = _numeric(joined["b6i20"], "b6i20")
    readable["medical_service_free_fully_or_partly"] = _decoded_or_missing("b7i5", joined["b7i5"])
    for output, source in {
        "package_component_rs": "b7i6",
        "doctor_surgeon_fee_rs": "b7i7",
        "medicines_rs": "b7i8",
        "diagnostic_tests_rs": "b7i9",
        "bed_charges_rs": "b7i10",
        "other_medical_expenses_rs": "b7i11",
        TARGET: "b7i12",
        "patient_transport_rs": "b7i13",
        "other_non_medical_household_expenses_rs": "b7i14",
        "total_expenditure_rs": "b7i15",
        "insurance_or_employer_reimbursement_rs": "b7i16",
        "household_income_loss_due_to_hospitalisation_rs": "b7i20",
    }.items():
        readable[output] = _numeric(joined[source], source, required=True)
    readable["major_source_of_finance"] = _decoded_or_missing("b7i17", joined["b7i17"])
    readable["place_of_hospitalisation"] = _decoded_or_missing("b7i18", joined["b7i18"])
    readable["treatment_state_code"] = _decoded_or_missing("b7i19", joined["b7i19"])
    readable["survey_multiplier"] = _numeric(joined["mult"], "mult", required=True)
    readable = readable.reset_index(drop=True)
    joined = joined.reset_index(drop=True)

    gender_counts = readable["gender"].fillna("Missing or invalid").value_counts()
    gender_audit_rows = []
    for category in ["Male", "Female", "Transgender", "Other", "Missing or invalid"]:
        count = int(gender_counts.get(category, 0))
        included = category in {"Male", "Female"}
        gender_audit_rows.append(
            {
                "gender_category": category,
                "original_record_count": count,
                "included_in_modelling": "Yes" if included else "No",
                "reason": (
                    "Included by the defined Male/Female modelling population."
                    if included
                    else "Preserved in readable data but excluded because the model was not defined or validated for this group."
                ),
            }
        )

    modelling = readable.loc[readable["gender"].isin(["Male", "Female"])].copy()
    modelling["gender"] = modelling["gender"].map({"Female": 0, "Male": 1}).astype(int)
    modelling["pregnant"] = decode_yes_no_binary(
        joined.loc[modelling.index, "b3c11"], "b3c11"
    ).astype(float)
    modelling["communicable_disease"] = decode_communicable_binary(
        joined.loc[modelling.index, "b3c13"]
    ).astype(float)
    modelling["chronic_ailment"] = decode_yes_no_binary(
        joined.loc[modelling.index, "b3c14"], "b3c14"
    ).astype(float)
    modelling["other_ailment_last_15_days"] = decode_yes_no_binary(
        joined.loc[modelling.index, "b3c15"], "b3c15"
    ).astype(float)
    modelling["other_ailment_previous_day"] = decode_yes_no_binary(
        joined.loc[modelling.index, "b3c16"], "b3c16"
    ).astype(float)
    modelling["health_financing_or_insurance_coverage"] = decode_coverage_binary(
        joined.loc[modelling.index, "b3c17"]
    ).astype(float)
    for output, source in {
        "surgery": "b6i13",
        "medicine": "b6i14",
        "xray_ecg_eeg_scan": "b6i15",
        "other_diagnostic_tests": "b6i16",
    }.items():
        modelling[output] = decode_service_received_binary(
            joined.loc[modelling.index, source], source
        ).astype(float)
    modelling["treated_on_medical_advice_before_hospitalisation"] = decode_yes_no_binary(
        joined.loc[modelling.index, "b6i17"], "b6i17"
    ).astype(float)
    if set(modelling["gender"].unique()) != {0, 1}:
        raise ValueError("Final ML gender field must contain exactly Female=0 and Male=1")

    components = readable[
        [
            "package_component_rs",
            "doctor_surgeon_fee_rs",
            "medicines_rs",
            "diagnostic_tests_rs",
            "bed_charges_rs",
            "other_medical_expenses_rs",
        ]
    ].sum(axis=1)
    component_mismatch = int(components.ne(readable[TARGET]).sum())
    if component_mismatch:
        raise ValueError(
            f"Medical component sum disagrees with target for {component_mismatch} rows"
        )

    person_episode_counts = readable["person_group_id"].value_counts()
    numeric_800x_counts: dict[str, int] = {}
    unresolved_categorical_800x_counts: dict[str, int] = {}
    for level in range(1, 8):
        raw = _read_level(level)
        for column in raw.columns:
            count = int(
                raw[column]
                .astype("string")
                .str.strip()
                .str.match(SPECIAL_CODE_PATTERN, na=False)
                .sum()
            )
            if count:
                definition = COLUMN_DEFINITIONS.get(column)
                if definition and definition.data_type == "numeric":
                    numeric_800x_counts[f"L{level}.{column}"] = count
                else:
                    unresolved_categorical_800x_counts[f"L{level}.{column}"] = count

    audit = {
        "source": OFFICIAL_SOURCE,
        "source_files": {path.name: _sha256(path) for path in sorted(RAW_DIR.glob("hhscsL*.csv"))},
        "source_level_rows": {f"hhscsL{level}": int(len(_read_level(level))) for level in range(1, 8)},
        "raw_l4_hospitalisation_records": int(len(l4)),
        "after_l2_living_person_join": int(len(living_person_episodes)),
        "l2_unmatched_or_deceased_records_excluded": int(person_unmatched.sum()),
        "l2_unmatched_member_codes": {str(key): int(value) for key, value in person_unmatched_codes.items()},
        "l2_excluded_records_verified_in_l3_deceased_roster": int(deceased_reconciled.sum()),
        "l2_excluded_records_not_verified_as_deceased": int((~deceased_reconciled).sum()),
        "after_l1_household_join_readable_records": int(len(readable)),
        "l1_unmatched_records_excluded": int(household_unmatched.sum()),
        "suno_cross_level_join_policy": (
            "Sample sub-unit number (suno) is retained for reference but excluded from the "
            "cross-level entity key. The supplied public-use files contain missing L1/L3 "
            "values and three L2/L4 disagreements; all remaining key fields are unique."
        ),
        "suno_missing_records_by_level": {
            "L1": int(l1["suno"].isna().sum()),
            "L2": int(l2["suno"].isna().sum()),
            "L3": int(l3["suno"].isna().sum()),
            "L4": int(l4["suno"].isna().sum()),
        },
        "suno_l4_l2_mismatched_living_episode_records": int(l2_suno_mismatch.sum()),
        "suno_l4_l1_mismatched_living_episode_records": int(l1_suno_mismatch.sum()),
        "suno_l4_l1_mismatched_distinct_households": int(
            l1_suno_alignment.loc[l1_suno_mismatch, HOUSEHOLD_KEY]
            .drop_duplicates()
            .shape[0]
        ),
        "readable_duplicate_record_ids": int(readable["hospitalisation_record_id"].duplicated().sum()),
        "persons_with_multiple_episodes": int(person_episode_counts.gt(1).sum()),
        "maximum_episodes_per_person": int(person_episode_counts.max()),
        "gender_counts": {
            "male": int(gender_counts.get("Male", 0)),
            "female": int(gender_counts.get("Female", 0)),
            "transgender_or_other": int(gender_counts.get("Transgender", 0) + gender_counts.get("Other", 0)),
            "missing_or_invalid": int(gender_counts.get("Missing or invalid", 0)),
        },
        "gender_filtering_rows": gender_audit_rows,
        "ml_records_after_gender_filter": int(len(modelling)),
        "target_missing": int(readable[TARGET].isna().sum()),
        "target_negative": int(readable[TARGET].lt(0).sum()),
        "target_zero": int(readable[TARGET].eq(0).sum()),
        "component_sum_mismatch": component_mismatch,
        "unverified_state_code_99_records": int(
            readable["state"].eq("Unverified state code 99").sum()
        ),
        "unresolved_categorical_800x_codes_detected": unresolved_categorical_800x_counts,
        "numeric_800x_values_retained": numeric_800x_counts,
        "numeric_800x_policy": (
            "Values in officially defined continuous rupee/day/count fields are "
            "retained as measurements. The same patterns in categorical or binary "
            "fields are rejected as unresolved codes."
        ),
        "numeric_9999_policy": (
            "Schedule 25.0 defines monetary entries as whole rupees and does not "
            "define 9999 as a missing-value sentinel. An exact monetary value of "
            "Rs. 9,999 is therefore retained as numeric rather than guessed to be missing."
        ),
        "gender_limitation": GENDER_LIMITATION,
    }
    return PreparedData(readable, modelling.reset_index(drop=True), audit, raw_headers)


def export_readable_level_files() -> list[Path]:
    """Create optional L1-L7 readable reference copies without changing raw data."""

    READABLE_LEVEL_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for level in range(1, 8):
        raw = _read_level(level)
        decoded = raw.copy()
        raw_sector = raw["sec"] if "sec" in raw else None
        if "b5i4" in decoded:
            decoded["b5i4"] = decode_series(
                "b5i4", decoded["b5i4"], sector=raw_sector
            )
        for column in list(decoded.columns):
            if column == "b5i4":
                continue
            if column in CATEGORY_MAPPINGS or column in {"nssreg", "dist"}:
                decoded[column] = decode_series(column, decoded[column])
            definition = COLUMN_DEFINITIONS.get(column)
            if definition and definition.data_type == "numeric":
                decoded[column] = _numeric(decoded[column], column)
        rename = {
            column: COLUMN_DEFINITIONS[column].clean_name
            if column in COLUMN_DEFINITIONS
            else f"unverified_public_use_field_{column}"
            for column in decoded.columns
        }
        decoded = decoded.rename(columns=rename)
        if decoded.columns.duplicated().any():
            duplicates = decoded.columns[decoded.columns.duplicated()].tolist()
            raise ValueError(f"Readable L{level} column-name collision: {duplicates}")
        path = READABLE_LEVEL_DIR / f"hhscsL{level}_readable.csv"
        decoded.to_csv(path, index=False)
        paths.append(path)
    return paths


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _feature_types(features: list[str]) -> tuple[list[str], list[str], list[str]]:
    numeric = [feature for feature in features if feature in NUMERIC_FEATURES]
    binary = [feature for feature in features if feature in BINARY_FEATURES]
    nominal = [feature for feature in features if feature in NOMINAL_FEATURES]
    classified = set(numeric) | set(binary) | set(nominal)
    missing = sorted(set(features) - classified)
    if missing:
        raise ValueError(f"Feature type not declared for: {missing}")
    return numeric, binary, nominal


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    numeric, binary, nominal = _feature_types(features)
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            ("numeric", SimpleImputer(strategy="median"), numeric)
        )
    if binary:
        transformers.append(
            ("binary", SimpleImputer(strategy="most_frequent"), binary)
        )
    if nominal:
        transformers.append(
            (
                "nominal",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="Missing or not applicable",
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                dtype=np.float32,
                            ),
                        ),
                    ]
                ),
                nominal,
            )
        )
    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def _hgb_factory() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=600,
        max_leaf_nodes=127,
        min_samples_leaf=40,
        l2_regularization=10.0,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=25,
        random_state=RANDOM_STATE,
    )


HGB_PARAMETERS = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 600,
    "max_leaf_nodes": 127,
    "min_samples_leaf": 40,
    "l2_regularization": 10.0,
    "early_stopping": True,
    "validation_fraction": 0.10,
    "n_iter_no_change": 25,
    "random_state": RANDOM_STATE,
}


class CatBoostSklearnAdapter(RegressorMixin, BaseEstimator):
    """Expose CatBoost through sklearn 1.9's estimator-tag protocol.

    The CatBoost version installed in the project predates sklearn's current
    ``__sklearn_tags__`` requirement.  Keeping the third-party estimator behind
    this minimal adapter lets Pipeline perform its normal fitted-state checks
    without changing CatBoost's training or prediction behaviour.
    """

    def __init__(
        self,
        iterations: int = 250,
        depth: int = 8,
        learning_rate: float = 0.06,
        l2_leaf_reg: float = 8.0,
        loss_function: str = "RMSE",
        random_seed: int = RANDOM_STATE,
        verbose: bool = False,
        allow_writing_files: bool = False,
        thread_count: int = -1,
    ) -> None:
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.loss_function = loss_function
        self.random_seed = random_seed
        self.verbose = verbose
        self.allow_writing_files = allow_writing_files
        self.thread_count = thread_count

    def fit(
        self, X: Any, y: Any, sample_weight: Any | None = None
    ) -> "CatBoostSklearnAdapter":
        from catboost import CatBoostRegressor

        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function=self.loss_function,
            random_seed=self.random_seed,
            verbose=self.verbose,
            allow_writing_files=self.allow_writing_files,
            thread_count=self.thread_count,
        )
        self.model_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(X), dtype=float)


def model_candidates(include_extended: bool) -> list[ModelCandidate]:
    candidates = [
        ModelCandidate("hist_gradient_boosting", "identity", _hgb_factory, HGB_PARAMETERS),
        ModelCandidate("hist_gradient_boosting_log1p", "log1p", _hgb_factory, HGB_PARAMETERS),
    ]
    if not include_extended:
        return candidates
    candidates.extend(
        [
            ModelCandidate(
                "random_forest",
                "identity",
                lambda: RandomForestRegressor(
                    n_estimators=160,
                    max_depth=22,
                    max_features=0.8,
                    min_samples_leaf=8,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
                {
                    "n_estimators": 160,
                    "max_depth": 22,
                    "max_features": 0.8,
                    "min_samples_leaf": 8,
                    "random_state": RANDOM_STATE,
                },
            ),
            ModelCandidate(
                "gradient_boosting",
                "identity",
                lambda: GradientBoostingRegressor(
                    loss="huber",
                    learning_rate=0.05,
                    n_estimators=140,
                    max_depth=3,
                    min_samples_leaf=25,
                    subsample=0.8,
                    random_state=RANDOM_STATE,
                ),
                {
                    "loss": "huber",
                    "learning_rate": 0.05,
                    "n_estimators": 140,
                    "max_depth": 3,
                    "min_samples_leaf": 25,
                    "subsample": 0.8,
                    "random_state": RANDOM_STATE,
                },
            ),
        ]
    )
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        return candidates
    candidates.append(
        ModelCandidate(
            "catboost_log1p",
            "log1p",
            lambda: CatBoostSklearnAdapter(
                iterations=250,
                depth=8,
                learning_rate=0.06,
                l2_leaf_reg=8.0,
                loss_function="RMSE",
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            ),
            {
                "iterations": 250,
                "depth": 8,
                "learning_rate": 0.06,
                "l2_leaf_reg": 8.0,
                "loss_function": "RMSE",
                "random_seed": RANDOM_STATE,
            },
        )
    )
    return candidates


def _weights(series: pd.Series) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Survey multipliers must be finite and positive")
    return values / values.mean()


def _fit_candidate(
    candidate: ModelCandidate,
    frame: pd.DataFrame,
    features: list[str],
) -> Pipeline:
    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor(features)),
            ("model", candidate.factory()),
        ]
    )
    y = frame[TARGET].to_numpy(dtype=float)
    fit_target = np.log1p(y) if candidate.target_transform == "log1p" else y
    pipeline.fit(
        frame[features],
        fit_target,
        model__sample_weight=_weights(frame["survey_multiplier"]),
    )
    return pipeline


def _predict(pipeline: Pipeline, transform: str, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    values = np.asarray(pipeline.predict(frame[features]), dtype=float)
    if transform == "log1p":
        values = np.expm1(values)
    return np.maximum(0.0, values)


def _metric_payload(
    frame: pd.DataFrame, prediction: np.ndarray
) -> tuple[dict[str, float], dict[str, float]]:
    actual = frame[TARGET].to_numpy(dtype=float)
    return (
        regression_metrics(actual, prediction),
        regression_metrics(actual, prediction, sample_weight=_weights(frame["survey_multiplier"])),
    )


def _encoded_feature_names(pipeline: Pipeline) -> list[str]:
    return [
        str(name)
        for name in pipeline.named_steps["preprocessor"].get_feature_names_out()
    ]


def _validate_encoded_matrix(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    features: list[str],
) -> dict[str, Any]:
    transformed = np.asarray(
        pipeline.named_steps["preprocessor"].transform(frame[features].iloc[: min(len(frame), 10_000)]),
        dtype=float,
    )
    names = _encoded_feature_names(pipeline)
    if not np.isfinite(transformed).all():
        raise ValueError("Encoded feature matrix contains non-finite values")
    onehot_indices = [index for index, name in enumerate(names) if name.startswith("nominal__")]
    binary_indices = [index for index, name in enumerate(names) if name.startswith("binary__")]
    if onehot_indices and not np.isin(transformed[:, onehot_indices], [0.0, 1.0]).all():
        raise ValueError("One-hot columns contain values other than 0/1")
    if binary_indices and not np.isin(transformed[:, binary_indices], [0.0, 1.0]).all():
        raise ValueError("Binary columns contain values other than 0/1")
    gender_name = next((name for name in names if name == "binary__gender"), None)
    if gender_name is None:
        raise ValueError("Encoded design matrix does not contain binary__gender")
    gender_index = names.index(gender_name)
    if not np.isin(transformed[:, gender_index], [0.0, 1.0]).all():
        raise ValueError("Encoded gender is not strictly 0/1")
    return {
        "sample_rows_checked": int(len(transformed)),
        "encoded_column_count": int(len(names)),
        "one_hot_column_count": int(len(onehot_indices)),
        "binary_column_count": int(len(binary_indices)),
        "all_encoded_values_finite": True,
        "one_hot_columns_only_0_or_1": True,
        "binary_columns_only_0_or_1": True,
        "gender_only_0_or_1": True,
    }


def _evaluate_validation_candidate(
    candidate: ModelCandidate,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    *,
    label: str,
) -> tuple[dict[str, Any], Pipeline]:
    started = time.perf_counter()
    pipeline = _fit_candidate(candidate, train, features)
    prediction = _predict(pipeline, candidate.target_transform, validation, features)
    metrics, weighted = _metric_payload(validation, prediction)
    row = {
        "label": label,
        "model": candidate.name,
        "target_transform": candidate.target_transform,
        "selected_original_feature_count": len(features),
        "encoded_column_count": len(_encoded_feature_names(pipeline)),
        "validation_records": len(validation),
        "fit_seconds": time.perf_counter() - started,
        **metrics,
    }
    row.update({f"weighted_{key}": value for key, value in weighted.items()})
    return row, pipeline


def _archive_existing_primary() -> dict[str, Any]:
    archive_dir = MODEL_DIR / "baseline_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "primary_model.joblib",
        "interval_models.joblib",
        "metadata.json",
        "evaluation.json",
        "model_comparison.csv",
        "feature_importance.csv",
        "largest_prediction_errors.csv",
    ]
    rows = []
    for name in names:
        source = PRODUCTION_MODEL_DIR / name
        if not source.exists():
            continue
        destination = archive_dir / name
        if not destination.exists():
            shutil.copy2(source, destination)
        if _sha256(source) != _sha256(destination):
            raise RuntimeError(f"Baseline archive hash mismatch for {name}")
        rows.append(
            {
                "file": name,
                "sha256": _sha256(source),
                "bytes": source.stat().st_size,
            }
        )
    manifest = {
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(PRODUCTION_MODEL_DIR),
        "production_primary_was_not_modified": True,
        "files": rows,
    }
    (archive_dir / "baseline_archive_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def _permutation_importance(
    pipeline: Pipeline,
    transform: str,
    frame: pd.DataFrame,
    features: list[str],
    *,
    sample_size: int = 5_000,
    repeats: int = 2,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    if len(frame) > sample_size:
        positions = rng.choice(len(frame), size=sample_size, replace=False)
        sample = frame.iloc[positions].reset_index(drop=True)
    else:
        sample = frame.reset_index(drop=True)
    baseline = regression_metrics(
        sample[TARGET].to_numpy(), _predict(pipeline, transform, sample, features)
    )["mae"]
    rows = []
    for feature in features:
        increases = []
        for _ in range(repeats):
            shuffled = sample.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            mae = regression_metrics(
                sample[TARGET].to_numpy(),
                _predict(pipeline, transform, shuffled, features),
            )["mae"]
            increases.append(mae - baseline)
        rows.append(
            {
                "feature": feature,
                "source_code": FEATURE_SOURCE_CODES.get(feature, "derived"),
                "mae_increase_mean_rs": float(np.mean(increases)),
                "mae_increase_std_rs": float(np.std(increases)),
                "permutation_repeats": repeats,
                "sample_records": len(sample),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mae_increase_mean_rs", ascending=False, ignore_index=True
    )


def _plot_actual_vs_predicted(
    actual: np.ndarray, prediction: np.ndarray, path: Path, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(RANDOM_STATE)
    if len(actual) > 6_000:
        positions = rng.choice(len(actual), size=6_000, replace=False)
        actual = actual[positions]
        prediction = prediction[positions]
    cap = max(float(np.quantile(actual, 0.99)), float(np.quantile(prediction, 0.99)), 1.0)
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.scatter(
        np.minimum(actual, cap),
        np.minimum(prediction, cap),
        s=8,
        alpha=0.22,
        color="#2563eb",
        edgecolors="none",
    )
    axis.plot([0, cap], [0, cap], color="#dc2626", linewidth=1.5)
    axis.set_title(title)
    axis.set_xlabel("Actual total medical expenditure (Rs.; capped at test P99)")
    axis.set_ylabel("Predicted total medical expenditure (Rs.; capped at test P99)")
    axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_residuals(
    prediction: np.ndarray, residual: np.ndarray, path: Path, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(RANDOM_STATE)
    if len(prediction) > 6_000:
        positions = rng.choice(len(prediction), size=6_000, replace=False)
        prediction = prediction[positions]
        residual = residual[positions]
    x_cap = max(float(np.quantile(prediction, 0.99)), 1.0)
    y_cap = max(float(np.quantile(np.abs(residual), 0.99)), 1.0)
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.scatter(
        np.minimum(prediction, x_cap),
        np.clip(residual, -y_cap, y_cap),
        s=8,
        alpha=0.22,
        color="#0f766e",
        edgecolors="none",
    )
    axis.axhline(0, color="#dc2626", linewidth=1.5)
    axis.set_title(title)
    axis.set_xlabel("Predicted expenditure (Rs.; capped at test P99)")
    axis.set_ylabel("Residual: actual - predicted (Rs.; clipped at residual P99)")
    axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_target(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = frame[TARGET].to_numpy(dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    cap = float(np.quantile(values, 0.99))
    axes[0].hist(np.minimum(values, cap), bins=60, color="#2563eb", alpha=0.85)
    axes[0].set_title("Original target (capped at P99)")
    axes[0].set_xlabel("Total medical expenditure (Rs.)")
    axes[0].set_ylabel("Records")
    axes[1].hist(np.log1p(values), bins=60, color="#0f766e", alpha=0.85)
    axes[1].set_title("log1p target")
    axes[1].set_xlabel("log1p(total medical expenditure)")
    axes[1].set_ylabel("Records")
    figure.suptitle("NSS 80 Male/Female modelling population target distribution")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _subgroup_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mapping = {
        "Public": "Government or public hospital",
        "Private": "Private hospital",
    }
    for label, category in mapping.items():
        mask = frame["medical_institution_type"].eq(category).to_numpy()
        if not mask.any():
            continue
        metrics = regression_metrics(frame.loc[mask, TARGET].to_numpy(), prediction[mask])
        rows.append(
            {
                "institution_subgroup": label,
                "records": int(mask.sum()),
                **metrics,
            }
        )
    return rows


def _mapping_for_source(source_code: str) -> dict[str, str]:
    if source_code == "derived from b6i9":
        return {
            "1 or 2": "Standard or free ward",
            "3": "Special ward",
        }
    if "/" in source_code:
        return {}
    if source_code == "b5i4":
        return {
            **{
                f"Rural sector, code {code}": label
                for code, label in RURAL_HOUSEHOLD_TYPE.items()
            },
            **{
                f"Urban sector, code {code}": label
                for code, label in URBAN_HOUSEHOLD_TYPE.items()
            },
        }
    return CATEGORY_MAPPINGS.get(source_code, {})


def _encoding_report_rows(
    safe_pipeline: Pipeline,
    safe_features: list[str],
    leakage_pipeline: Pipeline,
    leakage_features: list[str],
) -> list[dict[str, str]]:
    names_by_set = {
        "Leakage-safe": _encoded_feature_names(safe_pipeline),
        "Leakage-Included Experimental Model": _encoded_feature_names(leakage_pipeline),
    }
    rows = []
    for feature_set, features in {
        "Leakage-safe": safe_features,
        "Leakage-Included Experimental Model": leakage_features,
    }.items():
        encoded_names = names_by_set[feature_set]
        for feature in features:
            source_code = FEATURE_SOURCE_CODES.get(feature, "derived")
            mapping = _mapping_for_source(source_code)
            if feature in NUMERIC_FEATURES:
                method = "Numeric; median imputation; no category ordering"
                missing = "Median imputation fitted on training records only"
                final_columns = [f"numeric__{feature}"]
            elif feature in BINARY_FEATURES:
                if feature == "gender":
                    method = "Official decode; Male/Female filter; Female=0, Male=1"
                elif source_code in {"b6i13", "b6i14", "b6i15", "b6i16"}:
                    method = "Official service receipt decode; not received=0, any received=1"
                elif source_code == "b3c17":
                    method = "Official coverage decode; not covered=0, any listed coverage=1"
                elif source_code == "b3c13":
                    method = "Official disease decode; not suffered=0, listed disease=1"
                else:
                    method = "Official yes/no decode; No=0, Yes=1"
                missing = "Most-frequent imputation fitted on training records only"
                final_columns = [f"binary__{feature}"]
            else:
                method = (
                    "Official sector-conditional rural/urban decode followed by one-hot encoding"
                    if source_code == "b5i4"
                    else "Official readable label followed by one-hot encoding"
                )
                missing = "Explicit 'Missing or not applicable' category fitted on training records only"
                prefix = f"nominal__{feature}_"
                final_columns = [name for name in encoded_names if name.startswith(prefix)]
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature": feature,
                    "source_code": source_code,
                    "readable_name": feature.replace("_", " "),
                    "original_categories": json.dumps(list(mapping.keys()), ensure_ascii=False),
                    "decoded_categories": json.dumps(list(mapping.values()), ensure_ascii=False),
                    "encoding_method": method,
                    "final_ml_columns": json.dumps(final_columns, ensure_ascii=False),
                    "missing_value_strategy": missing,
                }
            )
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _format_metric(value: float, metric: str) -> str:
    return f"{value:.4f}" if metric in {"r2", "rmsle"} else f"{value:,.2f}"


def _write_comparison_markdown(comparison: pd.DataFrame, path: Path) -> None:
    columns = ["model", "feature_set", "mae", "rmse", "r2", "median_absolute_error", "rmsle"]
    lines = [
        "# NSS 80 Retraining v3 Model Comparison",
        "",
        GENDER_LIMITATION,
        "",
        "| Model | Feature set | MAE (Rs.) | RMSE (Rs.) | R-squared | Median AE (Rs.) | RMSLE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison[columns].itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.feature_set} | {row.mae:,.2f} | {row.rmse:,.2f} | "
            f"{row.r2:.4f} | {row.median_absolute_error:,.2f} | {row.rmsle:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Leakage warning",
            "",
            LEAKAGE_WARNING,
            "",
            "Historical v1/v2 rows, when present, use the earlier FSU-grouped population and are context only. "
            "All v3 safe/experimental rows use the same Male/Female-only person-grouped split.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_preprocessing_summary(
    audit: dict[str, Any],
    safe_features: list[str],
    leakage_features: list[str],
    split: dict[str, Any],
    path: Path,
) -> None:
    gender = audit["gender_counts"]
    content = f"""# NSS 80 Retraining v3 Preprocessing Summary

## Source and modelling unit

- Official source: NSS 80 Schedule 25.0, January-December 2025.
- Raw files inspected: `hhscsL1.csv` through `hhscsL7.csv`.
- Main join: L1 household + L2 living-person roster + L4 hospitalisation episode.
- Join keys: `{', '.join(HOUSEHOLD_KEY)}` plus L2 `b3c1` = L4 `b6i2` for the person.
- `suno` is retained as a source identifier but excluded from the cross-level join because it is missing or inconsistent for some otherwise uniquely matched records in the supplied public-use files.
- Modelling unit: one hospitalisation episode per record.
- Target: `b7i12`, renamed `total_medical_expenditure_rs`.

## Cohort and gender filter

- Raw L4 episodes: {audit['raw_l4_hospitalisation_records']:,}.
- After L2 living-person join: {audit['after_l2_living_person_join']:,}; {audit['l2_excluded_records_verified_in_l3_deceased_roster']:,} excluded episodes were verified against the L3 deceased-member roster.
- Readable L1+L2+L4 episodes: {audit['after_l1_household_join_readable_records']:,}.
- Living-person episodes lost at the L1 join: {audit['l1_unmatched_records_excluded']:,}.
- `suno` discrepancies retained for audit: {audit['suno_l4_l2_mismatched_living_episode_records']:,} L4/L2 episodes and {audit['suno_l4_l1_mismatched_living_episode_records']:,} L4/L1 episodes across {audit['suno_l4_l1_mismatched_distinct_households']:,} households.
- Male records: {gender['male']:,}.
- Female records: {gender['female']:,}.
- Transgender/other records: {gender['transgender_or_other']:,}.
- Missing/invalid gender records: {gender['missing_or_invalid']:,}.
- Final Male/Female modelling records: {audit['ml_records_after_gender_filter']:,}.

{GENDER_LIMITATION}

The readable dataset retains every officially decoded gender category. Only the ML population is filtered. Gender is then encoded as Female=0 and Male=1.

## Decoding and missing values

Every categorical value used by the model is decoded from the official Schedule 25.0 before ML encoding. Genuine binary fields are converted to 0/1. Nominal categories are one-hot encoded. Real quantities remain numeric. Blank conditional fields remain missing; numeric fields use training-only median imputation, binary fields use training-only most-frequent imputation, and nominal fields use an explicit missing/not-applicable category. Unresolved 8001-8009-style values in categorical/binary fields cause the pipeline to fail. Values in officially defined continuous rupee/day/count fields are retained as measurements and audited.

## Split

- Random state: {split['random_state']}.
- Training records: {split['train_records']:,}.
- Test records: {split['test_records']:,}.
- Training people: {split['train_people']:,}.
- Test people: {split['test_people']:,}.
- Person overlap: {split['person_overlap_count']}.

The same saved train/test record IDs are used for the safe and leakage-included models.

## Final feature sets

Leakage-safe ({len(safe_features)} original features): {', '.join(safe_features)}.

Leakage-Included Experimental Model ({len(leakage_features)} original features): {', '.join(leakage_features)}.

## Leakage interpretation

{LEAKAGE_WARNING}
"""
    path.write_text(content, encoding="utf-8")


def _feature_selection_rows(
    selected_safe_features: list[str], leakage_features: list[str]
) -> list[dict[str, Any]]:
    groups = {
        **{feature: "core" for feature in CORE_FEATURES},
        **{feature: "current_v2_derived" for feature in CURRENT_SAFE_FEATURES},
        **{feature: "socioeconomic_test" for feature in SOCIOECONOMIC_FEATURES},
        **{feature: "additional_health_test" for feature in ADDITIONAL_HEALTH_FEATURES},
        **{feature: "pre_hospitalisation_test" for feature in PRE_HOSPITALISATION_FEATURES},
        **{feature: "extra_geography_test" for feature in EXTRA_GEOGRAPHY_FEATURES},
        **{feature: "leakage_experimental_only" for feature in LEAKAGE_FEATURES},
    }
    rows = []
    for feature in _ordered_unique([*groups, TARGET]):
        rows.append(
            {
                "feature": feature,
                "source_code": FEATURE_SOURCE_CODES.get(feature, "derived"),
                "feature_group": "target" if feature == TARGET else groups.get(feature, "derived"),
                "data_type": (
                    "numeric"
                    if feature in NUMERIC_FEATURES or feature == TARGET
                    else "binary"
                    if feature in BINARY_FEATURES
                    else "nominal"
                ),
                "selected_in_safe_model": "Yes" if feature in selected_safe_features else "No",
                "selected_in_leakage_experiment": "Yes" if feature in leakage_features else "No",
                "exclusion_reason": (
                    "Prediction target; never included in X."
                    if feature == TARGET
                    else "Observed during/after expenditure or directly target-contributing; experimental only."
                    if feature in LEAKAGE_FEATURES
                    else "Not selected by the controlled validation ablation."
                    if feature not in selected_safe_features
                    else ""
                ),
            }
        )
    return rows


def _historical_rows() -> list[dict[str, Any]]:
    path = PRODUCTION_MODEL_DIR / "evaluation.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for source in payload.get("test_comparison", []):
        if source.get("model") not in {
            "original_deployed_v1_log_hgb",
            "selected_payment_safe_v2_log_hgb",
        }:
            continue
        rows.append(
            {
                "model": source["model"],
                "feature_set": "Historical context; earlier FSU-grouped population/split",
                "evaluation_split": "historical_not_directly_comparable",
                "training_records": "",
                "test_records": "",
                "original_selected_features": "",
                "final_encoded_columns": "",
                **{key: source[key] for key in ["mae", "rmse", "r2", "median_absolute_error", "rmsle"]},
            }
        )
    return rows


def run(
    *,
    export_all_levels: bool = True,
    include_extended_models: bool = True,
    reuse_ablation_results: bool = False,
) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    baseline_hash_before = _sha256(PRODUCTION_MODEL_DIR / "primary_model.joblib")
    baseline_archive = _archive_existing_primary()

    prepared = prepare_hospitalisation_data()
    readable = prepared.readable
    modelling = prepared.modelling
    audit = prepared.audit

    readable_path = REPORT_DIR / "nss80_hospitalisation_readable.csv"
    readable.to_csv(readable_path, index=False)
    pd.DataFrame(audit["gender_filtering_rows"]).to_csv(
        REPORT_DIR / "gender_filtering_audit.csv", index=False
    )
    pd.DataFrame(column_mapping_rows(prepared.raw_headers)).to_csv(
        REPORT_DIR / "nss80_column_mapping.csv", index=False
    )
    pd.DataFrame(value_mapping_rows()).to_csv(
        REPORT_DIR / "nss80_value_mapping.csv", index=False
    )

    readable_level_paths: list[Path] = []
    if export_all_levels:
        readable_level_paths = export_readable_level_files()
    else:
        readable_level_paths = [
            READABLE_LEVEL_DIR / f"hhscsL{level}_readable.csv"
            for level in range(1, 8)
            if (READABLE_LEVEL_DIR / f"hhscsL{level}_readable.csv").exists()
        ]

    groups = modelling["person_group_id"]
    outer = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_index, test_index = next(
        outer.split(modelling, modelling[TARGET], groups)
    )
    train = modelling.iloc[train_index].reset_index(drop=True)
    test = modelling.iloc[test_index].reset_index(drop=True)
    train_people = set(train["person_group_id"])
    test_people = set(test["person_group_id"])
    person_overlap = train_people & test_people
    if person_overlap:
        raise RuntimeError("Person leakage detected between training and test")

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE_WITHIN_TRAIN,
        random_state=RANDOM_STATE + 1,
    )
    selection_index, validation_index = next(
        inner.split(train, train[TARGET], train["person_group_id"])
    )
    selection_train = train.iloc[selection_index].reset_index(drop=True)
    validation = train.iloc[validation_index].reset_index(drop=True)
    validation_overlap = set(selection_train["person_group_id"]) & set(
        validation["person_group_id"]
    )
    if validation_overlap:
        raise RuntimeError("Person leakage detected in model-selection split")

    id_columns = [
        "hospitalisation_record_id",
        "person_group_id",
        "household_id",
        "fsu_serial_no",
        "sample_household_no",
        "person_serial_no",
        "hospitalisation_case_serial_no",
    ]
    train[id_columns].to_csv(REPORT_DIR / "train_record_ids.csv", index=False)
    test[id_columns].to_csv(REPORT_DIR / "test_record_ids.csv", index=False)

    split = {
        "strategy": "GroupShuffleSplit by verified household/person identifier",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_records": int(len(train)),
        "test_records": int(len(test)),
        "train_people": int(train["person_group_id"].nunique()),
        "test_people": int(test["person_group_id"].nunique()),
        "person_overlap_count": 0,
        "selection_train_records": int(len(selection_train)),
        "validation_records": int(len(validation)),
        "selection_validation_person_overlap_count": 0,
    }

    ablation_candidate = ModelCandidate(
        "hist_gradient_boosting_log1p",
        "log1p",
        _hgb_factory,
        HGB_PARAMETERS,
    )
    ablation_specs: list[tuple[str, str, list[str]]] = [
        ("A", "Current existing leakage-safe feature semantics", CURRENT_SAFE_FEATURES),
        ("B", "New 18-feature core", CORE_FEATURES),
        ("C", "Core plus socioeconomic variables", _ordered_unique([*CORE_FEATURES, *SOCIOECONOMIC_FEATURES])),
        ("D", "Core plus additional L2 health variables", _ordered_unique([*CORE_FEATURES, *ADDITIONAL_HEALTH_FEATURES])),
        ("E", "Core plus pre-hospitalisation variables", _ordered_unique([*CORE_FEATURES, *PRE_HOSPITALISATION_FEATURES])),
        ("F", "Core plus extra geography", _ordered_unique([*CORE_FEATURES, *EXTRA_GEOGRAPHY_FEATURES])),
    ]
    ablation_path = REPORT_DIR / "ablation_results.csv"
    if reuse_ablation_results:
        if not ablation_path.exists():
            raise FileNotFoundError(
                "--reuse-ablation-results requested but ablation_results.csv is missing"
            )
        ablation_results = pd.read_csv(ablation_path, dtype={"label": "string"})
        required_labels = {"A", "B", "C", "D", "E", "F", "G"}
        if set(ablation_results["label"].astype(str)) != required_labels:
            raise RuntimeError("Existing ablation results are incomplete or incompatible")
        ablation_features = {
            str(row.label): json.loads(row.features)
            for row in ablation_results.itertuples(index=False)
        }
        print("Reusing completed person-grouped ablation results A-G", flush=True)
    else:
        ablation_rows: list[dict[str, Any]] = []
        ablation_features: dict[str, list[str]] = {}
        for code, description, features in ablation_specs:
            print(f"Ablation {code}: {description}", flush=True)
            row, _ = _evaluate_validation_candidate(
                ablation_candidate,
                selection_train,
                validation,
                features,
                label=code,
            )
            row["description"] = description
            row["features"] = json.dumps(features)
            row["selection_data"] = "person-grouped validation"
            ablation_rows.append(row)
            ablation_features[code] = features

        core_mae = next(row["mae"] for row in ablation_rows if row["label"] == "B")
        block_by_code = {
            "C": SOCIOECONOMIC_FEATURES,
            "D": ADDITIONAL_HEALTH_FEATURES,
            "E": PRE_HOSPITALISATION_FEATURES,
            "F": EXTRA_GEOGRAPHY_FEATURES,
        }
        helpful_blocks = [
            code
            for code in block_by_code
            if next(row["mae"] for row in ablation_rows if row["label"] == code) < core_mae
        ]
        g_features = _ordered_unique(
            [*CORE_FEATURES, *(feature for code in helpful_blocks for feature in block_by_code[code])]
        )
        if g_features == CORE_FEATURES:
            g_row = dict(next(row for row in ablation_rows if row["label"] == "B"))
            g_row.update(
                {
                    "label": "G",
                    "description": "Validation-selected combination; no expansion block improved core MAE",
                    "features": json.dumps(g_features),
                    "fit_seconds": 0.0,
                }
            )
        else:
            print(f"Ablation G: validated combination of blocks {helpful_blocks}", flush=True)
            g_row, _ = _evaluate_validation_candidate(
                ablation_candidate,
                selection_train,
                validation,
                g_features,
                label="G",
            )
            g_row["description"] = f"Core plus individually helpful blocks: {', '.join(helpful_blocks)}"
            g_row["features"] = json.dumps(g_features)
            g_row["selection_data"] = "person-grouped validation"
        ablation_rows.append(g_row)
        ablation_features["G"] = g_features
        ablation_results = pd.DataFrame(ablation_rows).sort_values("mae", ignore_index=True)
        ablation_results.to_csv(ablation_path, index=False)
    selected_ablation = str(ablation_results.iloc[0]["label"])
    selected_safe_features = ablation_features[selected_ablation]

    validation_rows: list[dict[str, Any]] = []
    candidates = model_candidates(include_extended_models)
    for candidate in candidates:
        print(f"Safe model selection: {candidate.name}", flush=True)
        row, _ = _evaluate_validation_candidate(
            candidate,
            selection_train,
            validation,
            selected_safe_features,
            label=candidate.name,
        )
        row["selection_data"] = "person-grouped validation"
        row["parameters"] = json.dumps(candidate.parameters, sort_keys=True)
        validation_rows.append(row)
        pd.DataFrame(validation_rows).sort_values("mae", ignore_index=True).to_csv(
            REPORT_DIR / "safe_model_selection.csv", index=False
        )
    validation_models = pd.DataFrame(validation_rows).sort_values("mae", ignore_index=True)
    validation_models.to_csv(REPORT_DIR / "safe_model_selection.csv", index=False)
    selected_candidate_name = str(validation_models.iloc[0]["model"])
    selected_candidate = next(
        candidate for candidate in candidates if candidate.name == selected_candidate_name
    )

    dummy = DummyRegressor(strategy="median")
    dummy.fit(
        np.zeros((len(train), 1)),
        train[TARGET].to_numpy(dtype=float),
        sample_weight=_weights(train["survey_multiplier"]),
    )
    dummy_prediction = np.maximum(
        0.0, dummy.predict(np.zeros((len(test), 1)))
    )
    dummy_metrics, dummy_weighted = _metric_payload(test, dummy_prediction)
    comparison_rows: list[dict[str, Any]] = [
        {
            "model": "DummyRegressor median",
            "feature_set": "Baseline",
            "evaluation_split": "v3_same_person_grouped_test",
            "training_records": len(train),
            "test_records": len(test),
            "original_selected_features": 0,
            "final_encoded_columns": 0,
            **dummy_metrics,
        }
    ]

    required_test_candidates = [
        next(candidate for candidate in candidates if candidate.name == "hist_gradient_boosting"),
        next(candidate for candidate in candidates if candidate.name == "hist_gradient_boosting_log1p"),
    ]
    if selected_candidate.name not in {candidate.name for candidate in required_test_candidates}:
        required_test_candidates.append(selected_candidate)
    fitted_safe: dict[str, Pipeline] = {}
    safe_test_results: dict[str, tuple[dict[str, float], dict[str, float], np.ndarray]] = {}
    for candidate in required_test_candidates:
        print(f"Final safe test fit: {candidate.name}", flush=True)
        pipeline = _fit_candidate(candidate, train, selected_safe_features)
        prediction = _predict(
            pipeline, candidate.target_transform, test, selected_safe_features
        )
        metrics, weighted = _metric_payload(test, prediction)
        fitted_safe[candidate.name] = pipeline
        safe_test_results[candidate.name] = (metrics, weighted, prediction)
        comparison_rows.append(
            {
                "model": candidate.name,
                "feature_set": f"Leakage-safe ({selected_ablation})",
                "evaluation_split": "v3_same_person_grouped_test",
                "training_records": len(train),
                "test_records": len(test),
                "original_selected_features": len(selected_safe_features),
                "final_encoded_columns": len(_encoded_feature_names(pipeline)),
                **metrics,
            }
        )

    safe_pipeline = fitted_safe[selected_candidate.name]
    safe_metrics, safe_weighted_metrics, safe_prediction = safe_test_results[
        selected_candidate.name
    ]
    leakage_experiment_features = _ordered_unique(
        [*selected_safe_features, *LEAKAGE_FEATURES]
    )
    if TARGET in leakage_experiment_features:
        raise RuntimeError("Target leaked into experimental X")
    print("Final leakage-included experimental fit", flush=True)
    leakage_pipeline = _fit_candidate(
        selected_candidate, train, leakage_experiment_features
    )
    leakage_prediction = _predict(
        leakage_pipeline,
        selected_candidate.target_transform,
        test,
        leakage_experiment_features,
    )
    leakage_metrics, leakage_weighted_metrics = _metric_payload(
        test, leakage_prediction
    )
    comparison_rows.append(
        {
            "model": f"{selected_candidate.name} - Leakage-Included Experimental Model",
            "feature_set": "Leakage-Included Experimental Model",
            "evaluation_split": "v3_same_person_grouped_test",
            "training_records": len(train),
            "test_records": len(test),
            "original_selected_features": len(leakage_experiment_features),
            "final_encoded_columns": len(_encoded_feature_names(leakage_pipeline)),
            **leakage_metrics,
        }
    )
    comparison_rows.extend(_historical_rows())
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    _write_comparison_markdown(comparison, REPORT_DIR / "model_comparison.md")

    safe_encoded_validation = _validate_encoded_matrix(
        safe_pipeline, test, selected_safe_features
    )
    leakage_encoded_validation = _validate_encoded_matrix(
        leakage_pipeline, test, leakage_experiment_features
    )
    safe_leakage_intersection = sorted(
        set(selected_safe_features) & KNOWN_LEAKAGE
    )
    if safe_leakage_intersection:
        raise RuntimeError(
            f"Leakage-safe feature set contains prohibited fields: {safe_leakage_intersection}"
        )

    safe_predictions = test[id_columns + ["gender", "medical_institution_type", TARGET]].copy()
    safe_predictions["predicted_total_medical_expenditure_rs"] = safe_prediction
    safe_predictions["residual_rs"] = safe_predictions[TARGET] - safe_prediction
    safe_predictions["absolute_error_rs"] = np.abs(safe_predictions["residual_rs"])
    safe_predictions.to_csv(REPORT_DIR / "safe_test_predictions.csv", index=False)
    leakage_predictions = test[id_columns + ["gender", "medical_institution_type", TARGET]].copy()
    leakage_predictions["predicted_total_medical_expenditure_rs"] = leakage_prediction
    leakage_predictions["residual_rs"] = leakage_predictions[TARGET] - leakage_prediction
    leakage_predictions["absolute_error_rs"] = np.abs(leakage_predictions["residual_rs"])
    leakage_predictions.to_csv(
        REPORT_DIR / "leakage_test_predictions.csv", index=False
    )

    safe_importance = _permutation_importance(
        safe_pipeline,
        selected_candidate.target_transform,
        test,
        selected_safe_features,
    )
    leakage_importance = _permutation_importance(
        leakage_pipeline,
        selected_candidate.target_transform,
        test,
        leakage_experiment_features,
    )
    safe_importance.to_csv(REPORT_DIR / "safe_feature_importance.csv", index=False)
    leakage_importance.to_csv(
        REPORT_DIR / "leakage_feature_importance.csv", index=False
    )

    pd.DataFrame(
        _encoding_report_rows(
            safe_pipeline,
            selected_safe_features,
            leakage_pipeline,
            leakage_experiment_features,
        )
    ).to_csv(REPORT_DIR / "nss80_feature_encoding_report.csv", index=False)
    pd.DataFrame(
        _feature_selection_rows(
            selected_safe_features, leakage_experiment_features
        )
    ).to_csv(REPORT_DIR / "nss80_feature_selection.csv", index=False)

    actual = test[TARGET].to_numpy(dtype=float)
    _plot_actual_vs_predicted(
        actual,
        safe_prediction,
        REPORT_DIR / "safe_actual_vs_predicted.png",
        "NSS 80 leakage-safe model: actual vs predicted",
    )
    _plot_actual_vs_predicted(
        actual,
        leakage_prediction,
        REPORT_DIR / "leakage_actual_vs_predicted.png",
        "NSS 80 Leakage-Included Experimental Model: actual vs predicted",
    )
    _plot_residuals(
        safe_prediction,
        actual - safe_prediction,
        REPORT_DIR / "safe_residual_plot.png",
        "NSS 80 leakage-safe model residuals",
    )
    _plot_residuals(
        leakage_prediction,
        actual - leakage_prediction,
        REPORT_DIR / "leakage_residual_plot.png",
        "NSS 80 Leakage-Included Experimental Model residuals",
    )
    _plot_target(modelling, REPORT_DIR / "target_distribution.png")

    safe_subgroups = _subgroup_metrics(test, safe_prediction)
    leakage_subgroups = _subgroup_metrics(test, leakage_prediction)
    safe_payload = {
        "label": "Leakage-Safe Model",
        "selected_ablation": selected_ablation,
        "selected_model": selected_candidate.name,
        "target_transform": selected_candidate.target_transform,
        "parameters": selected_candidate.parameters,
        "target": TARGET,
        "target_unit": "Indian rupees per hospitalisation episode",
        "training_record_count": len(train),
        "test_record_count": len(test),
        "original_selected_feature_count": len(selected_safe_features),
        "final_encoded_column_count": len(_encoded_feature_names(safe_pipeline)),
        "features": selected_safe_features,
        "metrics": safe_metrics,
        "survey_weighted_metrics": safe_weighted_metrics,
        "public_private_subgroups": safe_subgroups,
        "gender_limitation": GENDER_LIMITATION,
    }
    leakage_payload = {
        "label": "Leakage-Included Experimental Model",
        "warning": LEAKAGE_WARNING,
        "selected_model": selected_candidate.name,
        "target_transform": selected_candidate.target_transform,
        "parameters": selected_candidate.parameters,
        "target": TARGET,
        "training_record_count": len(train),
        "test_record_count": len(test),
        "original_selected_feature_count": len(leakage_experiment_features),
        "final_encoded_column_count": len(_encoded_feature_names(leakage_pipeline)),
        "features": leakage_experiment_features,
        "metrics": leakage_metrics,
        "survey_weighted_metrics": leakage_weighted_metrics,
        "public_private_subgroups": leakage_subgroups,
        "gender_limitation": GENDER_LIMITATION,
    }
    _write_json(REPORT_DIR / "safe_model_metrics.json", safe_payload)
    _write_json(REPORT_DIR / "leakage_model_metrics.json", leakage_payload)

    joblib.dump(
        {
            "pipeline": safe_pipeline,
            "target_transform": selected_candidate.target_transform,
            "features": selected_safe_features,
            "target": TARGET,
            "model_version": "retraining_v3_candidate",
            "gender_encoding": {"Female": 0, "Male": 1},
            "gender_limitation": GENDER_LIMITATION,
        },
        MODEL_DIR / "safe_model.joblib",
    )
    joblib.dump(
        {
            "pipeline": leakage_pipeline,
            "target_transform": selected_candidate.target_transform,
            "features": leakage_experiment_features,
            "target": TARGET,
            "model_version": "retraining_v3_leakage_experiment",
            "warning": LEAKAGE_WARNING,
            "gender_encoding": {"Female": 0, "Male": 1},
            "gender_limitation": GENDER_LIMITATION,
        },
        MODEL_DIR / "leakage_experimental_model.joblib",
    )

    reloaded_safe = joblib.load(MODEL_DIR / "safe_model.joblib")
    smoke = _predict(
        reloaded_safe["pipeline"],
        reloaded_safe["target_transform"],
        test.iloc[:3],
        reloaded_safe["features"],
    )
    if not np.isfinite(smoke).all():
        raise RuntimeError("Reloaded safe model produced non-finite predictions")

    baseline_hash_after = _sha256(PRODUCTION_MODEL_DIR / "primary_model.joblib")
    if baseline_hash_before != baseline_hash_after:
        raise RuntimeError("Production primary_model.joblib changed during v3 retraining")

    validation_summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_files_unchanged": all(
            _sha256(RAW_DIR / name) == digest
            for name, digest in audit["source_files"].items()
        ),
        "readable_record_ids_unique": audit["readable_duplicate_record_ids"] == 0,
        "target_numeric_no_missing": audit["target_missing"] == 0,
        "target_non_negative": audit["target_negative"] == 0,
        "target_equals_component_sum": audit["component_sum_mismatch"] == 0,
        "unresolved_categorical_800x_codes": audit[
            "unresolved_categorical_800x_codes_detected"
        ],
        "numeric_800x_values_retained": audit["numeric_800x_values_retained"],
        "numeric_800x_policy": audit["numeric_800x_policy"],
        "gender_filter_applied_before_split": True,
        "gender_encoding": {"Female": 0, "Male": 1},
        "gender_counts": audit["gender_counts"],
        "l2_excluded_records_verified_in_l3_deceased_roster": audit[
            "l2_excluded_records_verified_in_l3_deceased_roster"
        ],
        "l2_excluded_records_not_verified_as_deceased": audit[
            "l2_excluded_records_not_verified_as_deceased"
        ],
        "l1_unmatched_living_episode_records": audit[
            "l1_unmatched_records_excluded"
        ],
        "suno_cross_level_join_policy": audit["suno_cross_level_join_policy"],
        "suno_missing_records_by_level": audit["suno_missing_records_by_level"],
        "suno_l4_l2_mismatched_living_episode_records": audit[
            "suno_l4_l2_mismatched_living_episode_records"
        ],
        "suno_l4_l1_mismatched_living_episode_records": audit[
            "suno_l4_l1_mismatched_living_episode_records"
        ],
        "suno_l4_l1_mismatched_distinct_households": audit[
            "suno_l4_l1_mismatched_distinct_households"
        ],
        "person_group_overlap_train_test": 0,
        "same_test_record_ids_for_safe_and_leakage": safe_predictions["hospitalisation_record_id"].equals(
            leakage_predictions["hospitalisation_record_id"]
        ),
        "safe_target_not_in_X": TARGET not in selected_safe_features,
        "safe_known_leakage_columns_in_X": safe_leakage_intersection,
        "identifier_columns_used_as_predictors": sorted(
            set(id_columns) & set(selected_safe_features)
        ),
        "safe_encoded_matrix": safe_encoded_validation,
        "leakage_encoded_matrix": leakage_encoded_validation,
        "production_primary_model_sha256_before": baseline_hash_before,
        "production_primary_model_sha256_after": baseline_hash_after,
        "production_primary_model_replaced": False,
        "baseline_archive": baseline_archive,
        "numeric_9999_policy": audit["numeric_9999_policy"],
    }
    _write_json(REPORT_DIR / "validation_summary.json", validation_summary)
    _write_preprocessing_summary(
        audit,
        selected_safe_features,
        leakage_experiment_features,
        split,
        REPORT_DIR / "preprocessing_summary.md",
    )

    experiment_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_document": str(DOCUMENTATION_PATH.relative_to(BACKEND_ROOT)),
        "raw_files": [str(path.relative_to(BACKEND_ROOT)) for path in sorted(RAW_DIR.glob("hhscsL*.csv"))],
        "audit": audit,
        "split": split,
        "selected_ablation": selected_ablation,
        "selected_safe_features": selected_safe_features,
        "selected_model": selected_candidate.name,
        "target_transform": selected_candidate.target_transform,
        "safe_metrics": safe_metrics,
        "leakage_metrics": leakage_metrics,
        "readable_level_files": [str(path.relative_to(BACKEND_ROOT)) for path in readable_level_paths],
        "production_primary_model_replaced": False,
        "software": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "joblib": importlib.metadata.version("joblib"),
        },
        "gender_limitation": GENDER_LIMITATION,
        "leakage_warning": LEAKAGE_WARNING,
    }
    _write_json(REPORT_DIR / "experiment_metadata.json", experiment_metadata)
    print(json.dumps({
        "safe_metrics": safe_metrics,
        "leakage_metrics": leakage_metrics,
        "selected_ablation": selected_ablation,
        "selected_model": selected_candidate.name,
        "train_records": len(train),
        "test_records": len(test),
        "production_primary_model_replaced": False,
    }, indent=2))
    return experiment_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-readable-levels",
        action="store_true",
        help="Skip the large optional L1-L7 readable reference exports.",
    )
    parser.add_argument(
        "--minimum-models",
        action="store_true",
        help="Run only Dummy and required HistGradientBoosting candidates.",
    )
    parser.add_argument(
        "--reuse-ablation-results",
        action="store_true",
        help="Reuse a complete A-G ablation table from this deterministic split.",
    )
    args = parser.parse_args()
    run(
        export_all_levels=not args.skip_readable_levels,
        include_extended_models=not args.minimum_models,
        reuse_ablation_results=args.reuse_ablation_results,
    )


if __name__ == "__main__":
    main()
