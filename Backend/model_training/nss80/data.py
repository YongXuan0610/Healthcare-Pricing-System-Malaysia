"""Prepare an immutable, auditable inpatient modeling table from NSS 80 levels."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from .codebook import (
    AILMENT,
    FREE_MEDICAL_SERVICE,
    GENDER,
    MEDICAL_INSTITUTION,
    SECTOR,
    SERVICE_RECEIPT,
    STATE,
    TREATMENT_SYSTEM,
    WARD_TYPE,
)


TARGET = "medical_expenditure_inr"
NUMERIC_FEATURES = ["age_years", "length_of_stay_days"]
CATEGORICAL_FEATURES = [
    "gender",
    "sector",
    "state",
    "ailment",
    "treatment_system",
    "medical_institution",
    "ward_type",
    "surgery",
    "medicine",
    "imaging",
    "other_diagnostics",
    "free_medical_service",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
HOUSEHOLD_KEY = [
    "rnd",
    "sch",
    "fsu",
    "samp",
    "sec",
    "st",
    "nssreg",
    "dist",
    "strm",
    "sstrm",
    "subrnd",
    "sro",
    "suno",
    "sd",
    "sss",
    "hhd",
]


@dataclass
class NSS80DatasetAudit:
    raw_level4: pd.DataFrame
    modeling: pd.DataFrame
    report: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_level(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required NSS source file not found: {path}")
    return pd.read_csv(path, dtype=str, low_memory=False)


def _map_required(series: pd.Series, mapping: dict[str, str], field: str) -> pd.Series:
    mapped = series.map(mapping)
    unknown = sorted(series[mapped.isna()].dropna().unique().tolist())
    if unknown:
        raise ValueError(f"Unknown NSS code(s) for {field}: {unknown}")
    return mapped


def load_and_prepare_dataset(raw_dir: str | Path) -> NSS80DatasetAudit:
    """Join the member roster to inpatient cases without editing raw data.

    The target is Block 7 item 12: total medical expenditure (items 6-11), in
    whole Indian rupees. Component costs, total expenditure, reimbursement, and
    other post-outcome fields are deliberately excluded from the predictors.
    """

    raw_dir = Path(raw_dir)
    level2_path = raw_dir / "hhscsL2.csv"
    level3_path = raw_dir / "hhscsL3.csv"
    level4_path = raw_dir / "hhscsL4.csv"
    level2 = _read_level(level2_path)
    level3 = _read_level(level3_path)
    level4 = _read_level(level4_path)

    level4_required = set(
        HOUSEHOLD_KEY
        + [
            "b6i1",
            "b6i2",
            "b6i3",
            "b6i5",
            "b6i6",
            "b6i7",
            "b6i9",
            "b6i12",
            "b6i13",
            "b6i14",
            "b6i15",
            "b6i16",
            "b7i5",
            "b7i12",
            "mult",
        ]
    )
    missing = sorted(level4_required.difference(level4.columns))
    if missing:
        raise ValueError(f"NSS Level 4 is missing required columns: {missing}")

    living_roster = level2[HOUSEHOLD_KEY + ["b3c1", "b3c4", "b3c5"]].rename(
        columns={"b3c1": "member_serial", "b3c4": "gender_code", "b3c5": "roster_age"}
    )
    deceased_roster = level3[HOUSEHOLD_KEY + ["b4c1", "b4c3", "b4c4"]].rename(
        columns={"b4c1": "member_serial", "b4c3": "gender_code", "b4c4": "roster_age"}
    )
    roster = pd.concat([living_roster, deceased_roster], ignore_index=True)
    roster_key = HOUSEHOLD_KEY + ["member_serial"]
    if roster.duplicated(roster_key).any():
        raise ValueError("NSS member roster contains duplicate household/member keys")

    joined = level4.merge(
        roster,
        left_on=HOUSEHOLD_KEY + ["b6i2"],
        right_on=roster_key,
        how="left",
        validate="many_to_one",
    )
    unmatched_gender = int(joined["gender_code"].isna().sum())
    joined = joined.loc[joined["gender_code"].notna()].copy()

    age = pd.to_numeric(joined["b6i3"], errors="coerce")
    roster_age = pd.to_numeric(joined["roster_age"], errors="coerce")
    age_mismatches = int(((age - roster_age).abs() > 0).fillna(False).sum())
    if age_mismatches:
        raise ValueError(f"NSS age mismatch between inpatient and roster levels: {age_mismatches}")

    frame = pd.DataFrame(index=joined.index)
    frame["case_id"] = joined[HOUSEHOLD_KEY + ["b6i1"]].astype(str).agg("|".join, axis=1)
    frame["household_id"] = joined[HOUSEHOLD_KEY].astype(str).agg("|".join, axis=1)
    frame["fsu_id"] = joined["fsu"].astype(str)
    frame["age_years"] = age
    frame["length_of_stay_days"] = pd.to_numeric(joined["b6i12"], errors="coerce")
    frame["gender"] = _map_required(joined["gender_code"], GENDER, "gender")
    frame["sector"] = _map_required(joined["sec"], SECTOR, "sector")
    frame["state"] = _map_required(joined["st"], STATE, "state")
    frame["ailment"] = _map_required(joined["b6i5"], AILMENT, "ailment")
    frame["treatment_system"] = _map_required(
        joined["b6i6"], TREATMENT_SYSTEM, "treatment_system"
    )
    frame["medical_institution"] = _map_required(
        joined["b6i7"], MEDICAL_INSTITUTION, "medical_institution"
    )
    frame["ward_type"] = _map_required(joined["b6i9"], WARD_TYPE, "ward_type")
    for output, source in {
        "surgery": "b6i13",
        "medicine": "b6i14",
        "imaging": "b6i15",
        "other_diagnostics": "b6i16",
    }.items():
        frame[output] = _map_required(joined[source], SERVICE_RECEIPT, output)
    frame["free_medical_service"] = _map_required(
        joined["b7i5"], FREE_MEDICAL_SERVICE, "free_medical_service"
    )
    frame[TARGET] = pd.to_numeric(joined["b7i12"], errors="coerce")
    frame["survey_multiplier"] = pd.to_numeric(joined["mult"], errors="coerce")

    numeric_required = NUMERIC_FEATURES + [TARGET, "survey_multiplier"]
    invalid_numeric = {
        column: int(frame[column].isna().sum()) for column in numeric_required
    }
    if any(invalid_numeric.values()):
        raise ValueError(f"NSS modeling data has invalid numeric values: {invalid_numeric}")
    if not frame["case_id"].is_unique:
        raise ValueError("NSS inpatient case IDs are not unique")
    if not frame["age_years"].between(0, 120).all():
        raise ValueError("NSS age values fall outside 0-120")
    if not frame["length_of_stay_days"].between(1, 365).all():
        raise ValueError("NSS length of stay values fall outside 1-365 days")
    if (frame[TARGET] < 0).any():
        raise ValueError("NSS medical expenditure contains negative values")
    if (frame["survey_multiplier"] <= 0).any():
        raise ValueError("NSS survey multiplier contains non-positive values")

    raw_hashes = {
        path.name: sha256_file(path)
        for path in sorted(raw_dir.glob("hhscsL*.csv"))
    }
    report = {
        "source_level_rows": {
            "hhscsL2": int(len(level2)),
            "hhscsL3": int(len(level3)),
            "hhscsL4": int(len(level4)),
        },
        "modeling_rows": int(len(frame)),
        "unmatched_roster_rows_excluded": unmatched_gender,
        "duplicate_raw_rows": int(level4.duplicated().sum()),
        "duplicate_case_ids": int(frame["case_id"].duplicated().sum()),
        "fsu_count": int(frame["fsu_id"].nunique()),
        "household_count": int(frame["household_id"].nunique()),
        "zero_target_rows": int((frame[TARGET] == 0).sum()),
        "target_summary_inr": {
            key: float(value)
            for key, value in frame[TARGET]
            .describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            .items()
        },
        "source_sha256": raw_hashes,
        "target_definition": (
            "NSS Schedule 25.0 Block 7 item 12: total medical expenditure "
            "(items 6-11), whole Indian rupees, per inpatient case."
        ),
        "excluded_leakage_fields": [
            "b7i6-b7i11 component expenditures",
            "b7i13-b7i15 target-related total/non-medical expenditures",
            "b7i16 reimbursement",
            "b7i20 income loss",
            "b7i5 free-medical-service status (excluded from fitted v2 features)",
            "b6i13-b6i16 payment mode (collapsed to service received yes/no)",
            "b6i9 free versus paying-general detail (collapsed to standard_or_free)",
        ],
    }
    return NSS80DatasetAudit(raw_level4=level4, modeling=frame, report=report)
