from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd

from app.scrapers.public_pricing_common import PUBLIC_PRICE_COLUMNS


TRUSTED_PATIENT_CLASSES = {"all", "citizen", "foreigner", "unhcr", "resident"}
FREE_PRICE_PATTERN = re.compile(r"\b(?:free|percuma|tiada\s+caj)\b", re.IGNORECASE)

HOSPITALS = (
    {"hospital_code": "HKL", "hospital_name": "Hospital Kuala Lumpur", "state": "Kuala Lumpur", "source_url": "https://hkl.moh.gov.my/awam/caj-hospital", "source_name": "Hospital Kuala Lumpur Official Website"},
    {"hospital_code": "HRC", "hospital_name": "Hospital Rehabilitasi Cheras", "state": "Kuala Lumpur", "source_url": "https://jknkl.moh.gov.my/hrc/ward-and-treatment-charges/", "source_name": "Hospital Rehabilitasi Cheras Official Website"},
    {"hospital_code": "HCJ", "hospital_name": "Hospital Cyberjaya", "state": "Selangor", "source_url": "https://jknselangor.moh.gov.my/hcj/index.php/awam/caj-hospital", "source_name": "Hospital Cyberjaya Official Website"},
    {"hospital_code": "HSB", "hospital_name": "Hospital Sungai Buloh", "state": "Selangor", "source_url": "https://jknselangor.moh.gov.my/hsgbuloh/index.php/ms/lihat-caj-bayaran", "source_name": "Hospital Sungai Buloh Official Website"},
    {"hospital_code": "HPSF", "hospital_name": "Hospital Pakar Sultanah Fatimah, Muar", "state": "Johor", "source_url": "https://jknjohor.moh.gov.my/hpsf/caj-bayaran/", "source_name": "Hospital Pakar Sultanah Fatimah Official Website"},
)
HOSPITAL_BY_CODE = {item["hospital_code"]: item for item in HOSPITALS}


def _clean_string_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace("\xa0", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()


def normalize_public_pricing_frame(frame: pd.DataFrame, *, default_hospital_code: str = "HKL") -> pd.DataFrame:
    """Normalize legacy HKL rows and current shared-schema records."""

    normalized = frame.copy()
    for column in PUBLIC_PRICE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""
    for column in PUBLIC_PRICE_COLUMNS:
        if column != "price_rm":
            normalized[column] = _clean_string_series(normalized[column])

    default_metadata = HOSPITAL_BY_CODE.get(default_hospital_code)
    normalized.loc[normalized["source_type"].eq(""), "source_type"] = "hospital"
    if default_metadata is not None:
        missing = normalized["source_type"].eq("hospital") & normalized["hospital_code"].eq("")
        normalized.loc[missing, "hospital_code"] = default_hospital_code
    normalized["hospital_code"] = normalized["hospital_code"].str.upper()

    for hospital_code, metadata in HOSPITAL_BY_CODE.items():
        hospital_mask = normalized["hospital_code"].eq(hospital_code)
        for field in ("hospital_name", "state", "source_url", "source_name"):
            normalized.loc[hospital_mask & normalized[field].eq(""), field] = metadata[field]
        normalized.loc[hospital_mask & normalized["source_code"].eq(""), "source_code"] = hospital_code

    normalized["patient_class"] = normalized["patient_class"].str.casefold()
    normalized.loc[normalized["patient_class"].eq(""), "patient_class"] = "all"
    normalized["category"] = normalized["category"].map(
        lambda value: value.title() if value == value.casefold() else value
    )
    normalized["price_rm"] = pd.to_numeric(normalized["price_rm"], errors="coerce")
    free_mask = normalized["original_price_text"].str.contains(FREE_PRICE_PATTERN, na=False) | normalized["notes"].str.contains(FREE_PRICE_PATTERN, na=False)
    trusted_price = normalized["price_rm"].gt(0) | (normalized["price_rm"].eq(0) & free_mask)
    normalized = normalized[
        normalized["source_type"].isin({"hospital", "national_reference"})
        & normalized["patient_class"].isin(TRUSTED_PATIENT_CLASSES)
        & normalized["price_rm"].notna()
        & trusted_price
        & normalized["category"].ne("")
        & normalized["service_name"].ne("")
        & normalized["source_url"].ne("")
    ].copy()
    # Legacy rows have no IDs, so retain them by their full normalized values.
    with_ids = normalized["id"].ne("")
    current = normalized[with_ids].drop_duplicates(subset="id", keep="last")
    legacy = normalized[~with_ids].drop_duplicates()
    return pd.concat([current, legacy], ignore_index=True)[PUBLIC_PRICE_COLUMNS]


def public_category_names(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    return sorted({str(value).strip() for value in frame["category"].dropna() if str(value).strip()}, key=str.casefold)


def filter_public_pricing_frame(
    frame: pd.DataFrame,
    *,
    category: str | None = None,
    service_name: str | None = None,
    patient_class: str | None = None,
    charge_type: str | None = None,
    hospital_code: str | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = pd.Series(True, index=frame.index)
    if category:
        mask &= frame["category"].astype(str).str.strip().str.casefold().eq(str(category).strip().casefold())
    if hospital_code and str(hospital_code).strip().casefold() not in {"all", "*"}:
        selected = str(hospital_code).strip().casefold()
        mask &= frame["hospital_code"].astype(str).str.strip().str.casefold().eq(selected) | frame["source_type"].eq("national_reference")
    for column, query in (("service_name", service_name), ("charge_type", charge_type)):
        if query:
            mask &= frame[column].astype(str).str.contains(re.escape(str(query).strip()), case=False, na=False)
    if patient_class:
        requested = str(patient_class).strip().casefold()
        values = frame["patient_class"].astype(str).str.strip().str.casefold()
        if requested == "foreigner":
            mask &= values.isin({"foreigner", "unhcr", "resident", "all"})
        else:
            mask &= values.eq(requested) | values.isin({"all", "any"})
    return frame.loc[mask].copy()


class PublicPricingService:
    def __init__(self, csv_path: str | Path | None = None) -> None:
        backend_root = Path(__file__).resolve().parents[2]
        self.csv_path = Path(csv_path) if csv_path is not None else backend_root / "app" / "data" / "processed" / "public_hospital_prices.csv"
        self._cache: pd.DataFrame | None = None

    def load_dataframe(self, force_reload: bool = False) -> pd.DataFrame:
        if self._cache is not None and not force_reload:
            return self._cache.copy()
        self._cache = normalize_public_pricing_frame(pd.read_csv(self.csv_path)) if self.csv_path.exists() else pd.DataFrame(columns=PUBLIC_PRICE_COLUMNS)
        return self._cache.copy()

    def reload(self) -> pd.DataFrame:
        return self.load_dataframe(force_reload=True)

    def list_categories(self) -> list[str]:
        return public_category_names(self.load_dataframe())

    def list_hospitals(self) -> list[dict[str, str]]:
        frame = self.load_dataframe()
        available_codes = set(frame.loc[frame["source_type"].eq("hospital"), "hospital_code"])
        return [{key: str(metadata[key]) for key in ("hospital_code", "hospital_name", "state", "source_url", "source_name")} for metadata in HOSPITALS if metadata["hospital_code"] in available_codes]

    def find_matching_charges(self, category: str | None = None, service_name: str | None = None, patient_class: str | None = None, charge_type: str | None = None, hospital_code: str | None = None) -> pd.DataFrame:
        return filter_public_pricing_frame(self.load_dataframe(), category=category, service_name=service_name, patient_class=patient_class, charge_type=charge_type, hospital_code=hospital_code)

    def get_public_charge(self, category: str | None = None, service_name: str | None = None, patient_class: str | None = None, charge_type: str | None = None) -> dict[str, Any] | None:
        matches = self.find_matching_charges(category, service_name, patient_class, charge_type)
        if matches.empty:
            return None
        result = matches.iloc[0].to_dict()
        result["price_rm"] = float(result["price_rm"])
        return result

    @staticmethod
    def summarize_matches(matches: pd.DataFrame) -> dict[str, Any]:
        if matches.empty or "price_rm" not in matches.columns:
            return {"estimate_available": False, "pricing_type": "unavailable", "published_cost": None, "lower_estimate": None, "typical_estimate": None, "upper_estimate": None, "records_used": 0, "range_method": None}
        prices = pd.to_numeric(matches["price_rm"], errors="coerce").dropna()
        prices = prices[prices >= 0]
        if prices.empty:
            return PublicPricingService.summarize_matches(matches.iloc[0:0])
        records_used = int(len(prices))
        if records_used == 1:
            value = round(float(prices.iloc[0]), 2)
            return {"estimate_available": True, "pricing_type": "exact", "published_cost": value, "lower_estimate": None, "typical_estimate": value, "upper_estimate": None, "records_used": 1, "range_method": "single_published_charge"}
        if records_used <= 4:
            lower, typical, upper, method = float(prices.min()), float(prices.median()), float(prices.max()), "min_max"
        else:
            lower, typical, upper, method = float(prices.quantile(0.25)), float(prices.median()), float(prices.quantile(0.75)), "p25_median_p75"
        return {"estimate_available": True, "pricing_type": "range", "published_cost": None, "lower_estimate": round(lower, 2), "typical_estimate": round(typical, 2), "upper_estimate": round(upper, 2), "records_used": records_used, "range_method": method}

    @staticmethod
    def _serialize_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        records = frame.where(pd.notna(frame), "").to_dict(orient="records")
        for record in records:
            record["price_rm"] = float(record["price_rm"])
        return records

    @classmethod
    def build_hospital_comparison(cls, matches: pd.DataFrame, hospital_code: str | None = None) -> list[dict[str, Any]]:
        normalized = normalize_public_pricing_frame(matches)
        hospital_matches = normalized[normalized["source_type"].eq("hospital")]
        selected = (hospital_code or "all").upper()
        hospitals: list[dict[str, Any]] = []
        for metadata in HOSPITALS:
            if selected not in {"ALL", "*"} and metadata["hospital_code"] != selected:
                continue
            rows = hospital_matches[hospital_matches["hospital_code"].eq(metadata["hospital_code"])].copy()
            if rows.empty and selected in {"ALL", "*"}:
                continue
            source_metadata = dict(metadata)
            if not rows.empty:
                source_metadata["source_url"] = str(rows.iloc[0]["source_url"])
                source_metadata["source_name"] = str(rows.iloc[0]["source_name"])
            hospitals.append({**source_metadata, "records": cls._serialize_records(rows), "statistics": cls.summarize_matches(rows)})
        return hospitals

    @classmethod
    def build_national_references(cls, matches: pd.DataFrame) -> list[dict[str, Any]]:
        normalized = normalize_public_pricing_frame(matches)
        references = normalized[normalized["source_type"].eq("national_reference")]
        result: list[dict[str, Any]] = []
        for source_code, rows in references.groupby("source_code", sort=False):
            first = rows.iloc[0]
            result.append({"source_code": source_code, "source_name": first["source_name"], "state": first["state"], "source_url": first["source_url"], "source_updated_at": first["source_updated_at"], "records": cls._serialize_records(rows)})
        return result

    def get_pricing_reference(self, category: str | None = None, service_name: str | None = None, patient_class: str | None = None, charge_type: str | None = None, hospital_code: str | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
        matches = self.find_matching_charges(category, service_name, patient_class, charge_type, hospital_code)
        hospitals = matches[matches["source_type"].eq("hospital")]
        summary_matches = hospitals[hospitals["hospital_code"].eq("HKL")]
        if summary_matches.empty and not hospitals.empty:
            first_code = hospitals.iloc[0]["hospital_code"]
            summary_matches = hospitals[hospitals["hospital_code"].eq(first_code)]
        return self.summarize_matches(summary_matches), matches

    def estimate_public_cost(self, category: str | None = None, service_name: str | None = None, patient_class: str | None = None, charge_type: str | None = None) -> tuple[float | None, str, list[dict[str, Any]]]:
        matches = self.find_matching_charges(category, service_name, patient_class, charge_type)
        matches = matches[matches["source_type"].eq("hospital")]
        if matches.empty:
            return None, "unavailable", []
        return float(matches.iloc[0]["price_rm"]), "processed_csv", self._serialize_records(matches.head(5))


public_pricing_service = PublicPricingService()
