"""Read-only Malaysian LIAM published price-range reference service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "procedure_code",
    "body_system",
    "procedure_name",
    "care_setting",
    "segmentation_type",
    "segment",
    "typical_bill_amount_rm",
    "typical_bill_p25_rm",
    "typical_bill_p75_rm",
    "number_of_discharges_band",
    "data_status",
    "source_pdf_page",
    "source_url",
}

NUMERIC_COLUMNS = (
    "typical_bill_amount_rm",
    "typical_bill_p25_rm",
    "typical_bill_p75_rm",
)
REFERENCE_KEY_COLUMNS = (
    "procedure_code",
    "care_setting",
    "segmentation_type",
    "segment",
)
INSUFFICIENT_DATA_STATUS = "Insufficient credible data"


class LIAMPricingReferenceService:
    def __init__(self, dataset_path: str | Path | None = None) -> None:
        backend_root = Path(__file__).resolve().parents[2]
        self.dataset_path = (
            Path(dataset_path)
            if dataset_path is not None
            else backend_root
            / "datasets"
            / "malaysia_liam"
            / "reference"
            / "Medical_Dataset_Malaysia_LIAM_Price_Ranges.csv"
        )
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"LIAM reference dataset not found: {self.dataset_path}")
        frame = pd.read_csv(self.dataset_path)
        missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(f"LIAM dataset is missing required columns: {missing}")
        for column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        duplicates = frame.duplicated(list(REFERENCE_KEY_COLUMNS), keep=False)
        if duplicates.any():
            duplicate_keys = frame.loc[
                duplicates,
                list(REFERENCE_KEY_COLUMNS),
            ].drop_duplicates()
            raise ValueError(
                "LIAM dataset contains duplicate procedure/segment rows: "
                f"{duplicate_keys.to_dict(orient='records')}"
            )

        published = frame["data_status"].eq("Published")
        valid_range = (
            frame["typical_bill_p25_rm"].le(frame["typical_bill_amount_rm"])
            & frame["typical_bill_amount_rm"].le(frame["typical_bill_p75_rm"])
        )
        if not valid_range[published].all():
            raise ValueError("LIAM published rows contain an invalid P25/typical/P75 range")

        insufficient = frame["data_status"].eq(INSUFFICIENT_DATA_STATUS)
        if frame.loc[insufficient, list(NUMERIC_COLUMNS)].notna().any(axis=None):
            raise ValueError(
                "LIAM insufficient-data rows must not contain numeric bill values"
            )

        facility_states = frame[
            frame["segmentation_type"].eq("Facility State")
        ]
        normalized_states = facility_states["segment"].astype(str).str.strip()
        if normalized_states.eq("").any() or not normalized_states.eq(
            facility_states["segment"].astype(str)
        ).all():
            raise ValueError("LIAM Facility State names must be non-empty and normalized")

        self.frame = frame

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if pd.isna(value):
            return None
        text = str(value).strip()
        return text or None

    def list_procedures(self) -> list[dict[str, Any]]:
        records = (
            self.frame[["procedure_code", "procedure_name", "body_system", "care_setting"]]
            .drop_duplicates()
            .sort_values(["body_system", "procedure_name", "care_setting"])
        )
        procedures: list[dict[str, Any]] = []

        for procedure in records.to_dict(orient="records"):
            state_rows = self.frame[
                self.frame["procedure_code"].astype(str).eq(
                    str(procedure["procedure_code"])
                )
                & self.frame["care_setting"].astype(str).eq(
                    str(procedure["care_setting"])
                )
                & self.frame["segmentation_type"].eq("Facility State")
            ][["segment", "data_status"]].sort_values("segment")

            procedure["facility_states"] = [
                {
                    "state": str(row["segment"]),
                    "data_status": str(row["data_status"]),
                }
                for row in state_rows.to_dict(orient="records")
            ]
            procedures.append(procedure)

        return procedures

    def get_reference(
        self,
        *,
        procedure_code: str,
        care_setting: str | None = None,
        segmentation_type: str = "Overall",
        segment: str = "All",
    ) -> dict[str, Any]:
        matches = self.frame[
            self.frame["procedure_code"].astype(str).str.casefold().eq(
                procedure_code.strip().casefold()
            )
        ]
        if care_setting:
            matches = matches[
                matches["care_setting"].astype(str).str.casefold().eq(
                    care_setting.strip().casefold()
                )
            ]
        matches = matches[
            matches["segmentation_type"].astype(str).str.casefold().eq(
                segmentation_type.strip().casefold()
            )
            & matches["segment"].astype(str).str.casefold().eq(segment.strip().casefold())
        ]
        if len(matches) > 1:
            raise ValueError(
                "LIAM query is ambiguous; specify care_setting to select one published row"
            )
        if matches.empty:
            return {
                "component": "malaysia_liam_pricing_reference",
                "research_role": "pricing_reference_only",
                "estimate_available": False,
                "pricing_type": "unavailable",
                "currency": "MYR",
                "segmentation_type": segmentation_type,
                "segment": segment,
                "message": "No LIAM reference row matches this exact selection.",
            }

        row = matches.iloc[0]
        number_of_discharges_band = self._optional_text(
            row["number_of_discharges_band"]
        )
        if row["data_status"] != "Published" or pd.isna(row["typical_bill_amount_rm"]):
            return {
                "component": "malaysia_liam_pricing_reference",
                "research_role": "pricing_reference_only",
                "estimate_available": False,
                "pricing_type": "insufficient_credible_data",
                "currency": "MYR",
                "procedure_code": row["procedure_code"],
                "procedure_name": row["procedure_name"],
                "care_setting": row["care_setting"],
                "segmentation_type": row["segmentation_type"],
                "segment": row["segment"],
                "data_status": row["data_status"],
                "number_of_discharges_band": number_of_discharges_band,
                "source_pdf_page": int(row["source_pdf_page"]),
                "source_url": row["source_url"],
                "message": "LIAM marks this segment as insufficient credible data.",
            }

        return {
            "component": "malaysia_liam_pricing_reference",
            "research_role": "pricing_reference_only",
            "estimate_available": True,
            "pricing_type": "published_percentile_range",
            "currency": "MYR",
            "procedure_code": row["procedure_code"],
            "body_system": row["body_system"],
            "procedure_name": row["procedure_name"],
            "care_setting": row["care_setting"],
            "segmentation_type": row["segmentation_type"],
            "segment": row["segment"],
            "data_status": row["data_status"],
            "typical_bill_amount": float(row["typical_bill_amount_rm"]),
            "lower_reference": float(row["typical_bill_p25_rm"]),
            "upper_reference": float(row["typical_bill_p75_rm"]),
            "number_of_discharges_band": number_of_discharges_band,
            "range_method": "published_p25_typical_p75",
            "source_pdf_page": int(row["source_pdf_page"]),
            "source_url": row["source_url"],
            "message": (
                "Published LIAM reference only; no machine-learning model or currency "
                "conversion is applied."
            ),
        }
