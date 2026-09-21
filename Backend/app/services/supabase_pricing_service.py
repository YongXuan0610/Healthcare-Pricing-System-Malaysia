"""Supabase-backed pricing repository for Malaysian hospital pricing data.

The repository uses Supabase's Data API with the project's publishable key.
Only SELECT access is required here, so Row Level Security remains enforced.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from app.services.public_pricing_service import (
    PUBLIC_PRICE_COLUMNS,
    PublicPricingService,
    filter_public_pricing_frame,
    normalize_public_pricing_frame,
    public_category_names,
)


PUBLIC_CHARGE_COLUMNS = [
    "category",
    "service_name",
    "patient_class",
    "charge_type",
    "price_rm",
    "source_url",
]


class PricingDataStoreError(RuntimeError):
    """Raised when the Supabase pricing data store cannot be queried."""


class SupabasePricingDataAPI:
    """Small read-only wrapper around the Supabase PostgREST Data API."""

    def __init__(
        self,
        supabase_url: str | None = None,
        publishable_key: str | None = None,
    ) -> None:
        backend_root = Path(__file__).resolve().parents[2]
        load_dotenv(backend_root / ".env")

        self.supabase_url = (
            supabase_url or os.getenv("SUPABASE_URL", "")
        ).strip().rstrip("/")
        self.publishable_key = (
            publishable_key or os.getenv("SUPABASE_PUBLISHABLE_KEY", "")
        ).strip()

        self.session = requests.Session()
        self.timeout_seconds = 12

    def _ensure_configured(self) -> None:
        missing: list[str] = []

        if not self.supabase_url:
            missing.append("SUPABASE_URL")
        if not self.publishable_key:
            missing.append("SUPABASE_PUBLISHABLE_KEY")

        if missing:
            raise PricingDataStoreError(
                "Missing backend Supabase environment variable(s): "
                + ", ".join(missing)
                + ". Create Backend/.env before starting FastAPI."
            )

    def select(
        self,
        table: str,
        *,
        select: str = "*",
        filters: dict[str, str] | None = None,
        order: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_configured()

        params: dict[str, str] = {"select": select}

        if filters:
            params.update(filters)
        if order:
            params["order"] = order
        if limit is not None:
            params["limit"] = str(limit)

        try:
            response = self.session.get(
                f"{self.supabase_url}/rest/v1/{table}",
                headers={
                    "apikey": self.publishable_key,
                    "Accept": "application/json",
                },
                params=params,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise PricingDataStoreError(
                "Unable to connect to the Supabase pricing database."
            ) from exc

        if not response.ok:
            detail = response.text.strip()
            if len(detail) > 300:
                detail = detail[:300] + "..."
            raise PricingDataStoreError(
                f"Supabase pricing query failed ({response.status_code}): {detail}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise PricingDataStoreError(
                "Supabase returned an invalid pricing response."
            ) from exc

        if not isinstance(payload, list):
            raise PricingDataStoreError(
                "Supabase returned an unexpected pricing response."
            )

        return payload


class SupabasePublicPricingService:
    """Public-hospital pricing service backed by public.public_charges."""

    def __init__(
        self,
        api: SupabasePricingDataAPI | None = None,
        *,
        hrc_csv_path: str | Path | None = None,
        local_csv_path: str | Path | None = None,
    ) -> None:
        self.api = api or SupabasePricingDataAPI()
        backend_root = Path(__file__).resolve().parents[2]
        self.local_csv_path = Path(local_csv_path) if local_csv_path else (
            Path(hrc_csv_path) if hrc_csv_path else
            backend_root
            / "app"
            / "data"
            / "processed"
            / "public_hospital_prices.csv"
        )

    def load_dataframe(self) -> pd.DataFrame:
        rows = self.api.select(
            "public_charges",
            select="category,service_name,patient_class,charge_type,price_rm,source_url",
            order="id.asc",
        )

        frame = pd.DataFrame(rows)

        for column in PUBLIC_CHARGE_COLUMNS:
            if column not in frame.columns:
                frame[column] = ""

        hkl_frame = normalize_public_pricing_frame(
            frame[PUBLIC_CHARGE_COLUMNS].copy(),
            default_hospital_code="HKL",
        )

        if self.local_csv_path.exists():
            local_frame = normalize_public_pricing_frame(
                pd.read_csv(self.local_csv_path),
                default_hospital_code="HRC",
            )
            # Supabase remains the runtime source for HKL. The committed combined
            # dataset supplies the additional hospitals and national references.
            local_frame = local_frame[
                ~local_frame["hospital_code"].eq("HKL")
            ]
        else:
            local_frame = pd.DataFrame(columns=PUBLIC_PRICE_COLUMNS)

        return normalize_public_pricing_frame(
            pd.concat([hkl_frame, local_frame], ignore_index=True)
        )

    def find_matching_charges(
        self,
        category: str | None = None,
        service_name: str | None = None,
        patient_class: str | None = None,
        charge_type: str | None = None,
        hospital_code: str | None = None,
    ) -> pd.DataFrame:
        return filter_public_pricing_frame(
            self.load_dataframe(),
            category=category,
            service_name=service_name,
            patient_class=patient_class,
            charge_type=charge_type,
            hospital_code=hospital_code,
        )

    def get_pricing_reference(
        self,
        category: str | None = None,
        service_name: str | None = None,
        patient_class: str | None = None,
        charge_type: str | None = None,
        hospital_code: str | None = None,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        matches = self.find_matching_charges(
            category=category,
            service_name=service_name,
            patient_class=patient_class,
            charge_type=charge_type,
            hospital_code=hospital_code,
        )

        hospital_matches = matches[matches["source_type"].eq("hospital")]
        hkl_matches = hospital_matches[hospital_matches["hospital_code"].eq("HKL")]
        summary_matches = hkl_matches if not hkl_matches.empty else hospital_matches
        if not summary_matches.empty and hkl_matches.empty:
            first_code = summary_matches.iloc[0]["hospital_code"]
            summary_matches = summary_matches[summary_matches["hospital_code"].eq(first_code)]
        return PublicPricingService.summarize_matches(summary_matches), matches

    def list_categories(self) -> list[str]:
        return public_category_names(self.load_dataframe())

    def list_hospitals(self) -> list[dict[str, str]]:
        frame = self.load_dataframe()
        available = set(
            frame.loc[frame["source_type"].eq("hospital"), "hospital_code"]
        )
        from app.services.public_pricing_service import HOSPITALS
        return [
            {
                key: str(metadata[key])
                for key in (
                    "hospital_code", "hospital_name", "state",
                    "source_url", "source_name",
                )
            }
            for metadata in HOSPITALS
            if metadata["hospital_code"] in available
        ]


class SupabasePrivatePricingService:
    """Private package and ward pricing backed by Supabase PostgreSQL."""

    def __init__(self, api: SupabasePricingDataAPI | None = None) -> None:
        self.api = api or SupabasePricingDataAPI()

    @staticmethod
    def _package_to_frontend(row: dict[str, Any]) -> dict[str, Any]:
        gender = str(row.get("gender_target") or "Both")
        age = str(row.get("age_target") or "Adult")

        return {
            "id": str(row["id"]),
            "hospital": row["hospital"],
            "name": row["package_name"],
            "price": float(row["price_rm"]),
            "description": f"Target: {gender} / {age}",
            "gender_target": gender,
            "age_target": age,
        }

    def list_hospitals(self) -> list[str]:
        rows = self.api.select(
            "private_packages",
            select="hospital",
            order="hospital.asc",
        )

        hospitals = {
            str(row.get("hospital", "")).strip()
            for row in rows
            if str(row.get("hospital", "")).strip()
        }

        return sorted(hospitals, key=str.casefold)

    def list_packages(
        self,
        hospital: str,
        *,
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        rows = self.api.select(
            "private_packages",
            select=(
                "id,hospital,package_name,price_rm,"
                "gender_target,age_target"
            ),
            filters={"hospital": f"eq.{hospital}"},
            order="package_name.asc",
            limit=top_n,
        )

        return [self._package_to_frontend(row) for row in rows]

    def get_package(
        self,
        hospital: str,
        package_name: str,
    ) -> dict[str, Any] | None:
        rows = self.api.select(
            "private_packages",
            select=(
                "id,hospital,package_name,price_rm,"
                "gender_target,age_target"
            ),
            filters={
                "hospital": f"eq.{hospital}",
                "package_name": f"eq.{package_name}",
            },
            limit=1,
        )

        if not rows:
            return None

        return self._package_to_frontend(rows[0])

    def list_wards(self, hospital_key: str) -> list[dict[str, Any]]:
        rows = self.api.select(
            "private_ward_rates",
            select=(
                "id,hospital_key,hospital,ward_type_original,"
                "normalised_category,price_rm,rate_basis,notes"
            ),
            filters={"hospital_key": f"eq.{hospital_key}"},
            order="price_rm.asc",
        )

        return [
            {
                "id": f"ward-{row['id']}",
                "name": row["ward_type_original"],
                "dailyRate": float(row["price_rm"]),
                "description": row.get("normalised_category") or "",
                "rateBasis": row.get("rate_basis") or "",
                "notes": row.get("notes") or "",
            }
            for row in rows
        ]

    def get_ward(
        self,
        hospital_key: str,
        ward_type: str,
    ) -> dict[str, Any] | None:
        rows = self.api.select(
            "private_ward_rates",
            select=(
                "id,hospital_key,hospital,ward_type_original,"
                "normalised_category,price_rm,rate_basis,notes"
            ),
            filters={
                "hospital_key": f"eq.{hospital_key}",
                "ward_type_original": f"eq.{ward_type}",
            },
            limit=1,
        )

        if not rows:
            return None

        row = rows[0]
        return {
            "id": f"ward-{row['id']}",
            "name": row["ward_type_original"],
            "dailyRate": float(row["price_rm"]),
            "description": row.get("normalised_category") or "",
            "rateBasis": row.get("rate_basis") or "",
            "notes": row.get("notes") or "",
        }
