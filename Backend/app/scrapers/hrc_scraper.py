from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

from app.scrapers.public_pricing_common import (
    PUBLIC_PRICE_COLUMNS,
    PublicPricingSource,
    citizenship_from_patient_class,
    extract_source_updated_at,
    make_record,
)


SOURCE_URL = "https://jknkl.moh.gov.my/hrc/ward-and-treatment-charges/"
SOURCE_NAME = "Hospital Rehabilitasi Cheras Official Website"
HOSPITAL_NAME = "Hospital Rehabilitasi Cheras"
HOSPITAL_CODE = "HRC"
SOURCE = PublicPricingSource(
    source_type="hospital",
    source_code=HOSPITAL_CODE,
    source_name=SOURCE_NAME,
    source_url=SOURCE_URL,
    state="Kuala Lumpur",
    hospital_code=HOSPITAL_CODE,
    hospital_name=HOSPITAL_NAME,
)
OUTPUT_COLUMNS = PUBLIC_PRICE_COLUMNS

CATEGORY_BY_CLINICAL_SERVICE = {
    "physiotherapy": "Physiotherapy",
    "occupational therapy": "Occupational Therapy",
    "speech therapist": "Speech Therapy",
    "audiology": "Audiology",
    "traditional and complementary medicine": (
        "Traditional and Complementary Medicine"
    ),
}


class HRCHospitalChargesScraper:
    """Scrape and normalize HRC's official ward and treatment charge tables."""

    def __init__(
        self,
        source_url: str = SOURCE_URL,
        *,
        raw_output_path: str | Path | None = None,
        output_path: str | Path | None = None,
        combined_output_path: str | Path | None = None,
        hkl_csv_path: str | Path | None = None,
    ) -> None:
        data_root = Path(__file__).resolve().parents[1] / "data"
        self.source_url = source_url
        self.raw_output_path = Path(raw_output_path) if raw_output_path else (
            data_root / "raw" / "hrc_ward_and_treatment_charges_raw.html"
        )
        self.output_path = Path(output_path) if output_path else (
            data_root / "processed" / "hrc_public_charges.csv"
        )
        self.combined_output_path = (
            Path(combined_output_path)
            if combined_output_path
            else data_root / "processed" / "public_hospital_prices.csv"
        )
        self.hkl_csv_path = Path(hkl_csv_path) if hkl_csv_path else (
            data_root / "processed" / "hkl_public_charges.csv"
        )

    def scrape(
        self,
        html: str | None = None,
        *,
        scraped_at: str | None = None,
        save: bool = True,
    ) -> pd.DataFrame:
        page_html = html if html is not None else self._fetch_page()
        timestamp = scraped_at or datetime.now(timezone.utc).isoformat()
        frame = self.parse_html(page_html, scraped_at=timestamp)

        if frame.empty:
            raise ValueError(
                "No hospital charge data could be extracted from the HRC page."
            )

        if save:
            self._save_outputs(page_html, frame)

        return frame

    def _fetch_page(self) -> str:
        try:
            response = requests.get(
                self.source_url,
                timeout=30,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36 MyCareCost/1.0"
                    )
                },
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to fetch HRC ward and treatment charges page: {exc}"
            ) from exc

        response.encoding = response.apparent_encoding or response.encoding
        return response.text

    def parse_html(self, html: str, *, scraped_at: str) -> pd.DataFrame:
        soup = BeautifulSoup(html, "html.parser")
        records: list[dict[str, Any]] = []

        for table in soup.find_all("table"):
            grid = self._table_grid(table)
            if not grid or not grid[0]:
                continue

            table_name = self._clean_text(grid[0][0]).casefold()
            if table_name == "ward deposit":
                records.extend(self._parse_simple_class_table(grid, "Ward Deposit"))
            elif table_name == "ward charges":
                records.extend(self._parse_simple_class_table(grid, "Ward Charges"))
            elif table_name == "treatment charges":
                records.extend(
                    self._parse_simple_class_table(grid, "Treatment Charges")
                )
            elif table_name.startswith("specialist clinic treatment charges"):
                records.extend(self._parse_specialist_clinic_table(grid))
            elif table_name == "clinical services":
                records.extend(self._parse_clinical_services_table(table, grid))

        source_updated_at = extract_source_updated_at(html)
        normalized_records: list[dict[str, Any]] = []
        runtime_source = PublicPricingSource(
            **{**SOURCE.__dict__, "source_url": self.source_url}
        )
        for record in records:
            normalized_records.append(
                make_record(
                    runtime_source,
                    category=record["category"],
                    service_name=record["service_name"],
                    original_service_name=record["service_name"],
                    charge_type=record["charge_type"],
                    patient_class=record["patient_class"],
                    ward_class=record.get("ward_class", ""),
                    room_type=record.get("room_type", ""),
                    price_rm=float(record["price_rm"]),
                    price_unit=record.get("price_unit", ""),
                    original_price_text=record.get("original_price_text", ""),
                    notes=record.get("notes", ""),
                    scraped_at=scraped_at,
                    source_updated_at=source_updated_at,
                )
            )

        frame = pd.DataFrame(normalized_records)
        for column in OUTPUT_COLUMNS:
            if column not in frame.columns:
                frame[column] = ""

        frame = frame[OUTPUT_COLUMNS].copy()
        frame["price_rm"] = pd.to_numeric(frame["price_rm"], errors="coerce")
        frame = frame.dropna(subset=["price_rm"])
        frame = frame[frame["price_rm"] >= 0].copy()
        frame = frame.drop_duplicates(
            subset=[
                "hospital_code",
                "category",
                "service_name",
                "charge_type",
                "patient_class",
                "ward_class",
                "room_type",
                "price_rm",
                "price_unit",
            ]
        ).reset_index(drop=True)
        return frame

    @staticmethod
    def _table_grid(table: Tag) -> list[list[str]]:
        """Expand HTML rowspan/colspan cells into a rectangular text grid."""

        rows: list[list[str]] = []
        pending: dict[int, tuple[int, str]] = {}

        for html_row in table.find_all("tr"):
            row: list[str] = []
            column_index = 0

            def fill_pending() -> None:
                nonlocal column_index
                while column_index in pending:
                    rows_left, value = pending[column_index]
                    row.append(value)
                    if rows_left <= 1:
                        del pending[column_index]
                    else:
                        pending[column_index] = (rows_left - 1, value)
                    column_index += 1

            fill_pending()
            for cell in html_row.find_all(["th", "td"], recursive=False):
                fill_pending()
                value = " ".join(cell.get_text(" ", strip=True).split())
                colspan = max(int(cell.get("colspan", 1)), 1)
                rowspan = max(int(cell.get("rowspan", 1)), 1)
                for offset in range(colspan):
                    row.append(value)
                    if rowspan > 1:
                        pending[column_index + offset] = (rowspan - 1, value)
                column_index += colspan
            fill_pending()
            rows.append(row)

        width = max((len(row) for row in rows), default=0)
        return [row + [""] * (width - len(row)) for row in rows]

    def _parse_simple_class_table(
        self,
        grid: list[list[str]],
        category: str,
    ) -> list[dict[str, Any]]:
        headers = grid[0]
        records: list[dict[str, Any]] = []

        for row in grid[1:]:
            patient_class = self._patient_class(row[0] if row else "")
            if patient_class is None:
                continue

            for index, header in enumerate(headers[1:], start=1):
                price_text = row[index] if index < len(row) else ""
                price = self._parse_single_price(price_text)
                if price is None:
                    continue

                ward_class = self._ward_class(header)
                room_type = header if category == "Ward Charges" else ""
                charge_type = {
                    "Ward Deposit": "ward admission deposit",
                    "Ward Charges": "ward charge",
                    "Treatment Charges": "treatment charge",
                }[category]

                records.append(
                    {
                        "category": category,
                        "service_name": header,
                        "charge_type": charge_type,
                        "patient_class": patient_class,
                        "ward_class": ward_class,
                        "room_type": room_type,
                        "price_rm": price,
                        "price_unit": (
                            "deposit" if category == "Ward Deposit" else ""
                        ),
                        "notes": self._free_note(price_text),
                        "original_price_text": price_text,
                    }
                )

        return records

    def _parse_specialist_clinic_table(
        self,
        grid: list[list[str]],
    ) -> list[dict[str, Any]]:
        headers = grid[0]
        records: list[dict[str, Any]] = []

        for row in grid[1:]:
            patient_class = self._patient_class(row[0] if row else "")
            if patient_class is None:
                continue

            for index, referral_source in enumerate(headers[1:], start=1):
                price_text = row[index] if index < len(row) else ""
                for price, unit in self._parse_specialist_prices(price_text):
                    records.append(
                        {
                            "category": "Specialist Clinic",
                            "service_name": referral_source,
                            "charge_type": "specialist clinic consultation",
                            "patient_class": patient_class,
                            "ward_class": "",
                            "room_type": "",
                            "price_rm": price,
                            "price_unit": unit,
                            "notes": self._free_note(price_text),
                            "original_price_text": price_text,
                        }
                    )

        return records

    def _parse_clinical_services_table(
        self,
        table: Tag,
        grid: list[list[str]],
    ) -> list[dict[str, Any]]:
        rows = table.find_all("tr")
        if len(rows) < 2:
            return []

        service_cell = rows[1].find(["th", "td"])
        if service_cell is None:
            return []

        services = [
            self._clean_text(item.get_text(" ", strip=True))
            for item in service_cell.find_all("li")
            if self._clean_text(item.get_text(" ", strip=True))
        ]
        if not services:
            services = [self._clean_text(grid[1][0])]

        headers = grid[0]
        price_row = grid[1]
        records: list[dict[str, Any]] = []

        for service_name in services:
            category = CATEGORY_BY_CLINICAL_SERVICE.get(
                service_name.casefold(), service_name
            )
            for index, patient_label in enumerate(headers[1:], start=1):
                patient_class = self._patient_class(patient_label)
                price_text = price_row[index] if index < len(price_row) else ""
                price = self._parse_single_price(price_text)
                if patient_class is None or price is None:
                    continue

                records.append(
                    {
                        "category": category,
                        "service_name": service_name,
                        "charge_type": "clinical service",
                        "patient_class": patient_class,
                        "ward_class": "",
                        "room_type": "",
                        "price_rm": price,
                        "price_unit": (
                            "per session"
                            if "per session" in price_text.casefold()
                            else ""
                        ),
                        "notes": self._free_note(price_text),
                        "original_price_text": price_text,
                    }
                )

        return records

    @staticmethod
    def _parse_specialist_prices(value: str) -> list[tuple[float, str]]:
        normalized = " ".join(value.split())
        lowered = normalized.casefold()
        if not normalized:
            return []

        if "free for first consultation" in lowered:
            prices: list[tuple[float, str]] = [(0.0, "first consultation")]
            rm_values = re.findall(r"RM\s*([\d,]+(?:\.\d+)?)", normalized, re.I)
            if rm_values:
                prices.append((float(rm_values[-1].replace(",", "")), "subsequent visit"))
            return prices

        rm_values = [
            float(match.replace(",", ""))
            for match in re.findall(r"RM\s*([\d,]+(?:\.\d+)?)", normalized, re.I)
        ]
        if len(rm_values) >= 2:
            return [
                (rm_values[0], "first consultation"),
                (rm_values[1], "subsequent visit"),
            ]
        if len(rm_values) == 1:
            unit = (
                "first and subsequent consultation"
                if "subsequent" in lowered
                else "consultation"
            )
            return [(rm_values[0], unit)]
        if re.search(r"\b(?:free|percuma)\b", lowered):
            return [(0.0, "consultation")]
        return []

    @staticmethod
    def _parse_single_price(value: str) -> float | None:
        if re.search(r"\b(?:free|percuma)\b", value, re.I):
            return 0.0
        match = re.search(r"RM\s*([\d,]+(?:\.\d+)?)", value, re.I)
        if match is None:
            return None
        return float(match.group(1).replace(",", ""))

    @staticmethod
    def _patient_class(value: str) -> str | None:
        normalized = re.sub(r"[^a-z]", "", value.casefold())
        if normalized in {"citizen", "citizens"}:
            return "citizen"
        if normalized in {"noncitizen", "noncitizens", "foreigner", "foreigners"}:
            return "foreigner"
        return None

    @staticmethod
    def _ward_class(value: str) -> str:
        match = re.search(r"class\s*(\d+)", value, re.I)
        return f"Class {match.group(1)}" if match else value

    @staticmethod
    def _free_note(value: str) -> str:
        return "Published as free on the official source." if re.search(
            r"\b(?:free|percuma)\b", value, re.I
        ) else ""

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).replace("\xa0", " ").split())

    def _save_outputs(self, html: str, frame: pd.DataFrame) -> None:
        self.raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.combined_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_output_path.write_text(html, encoding="utf-8")
        frame.to_csv(self.output_path, index=False)
        self._save_combined_dataset(frame)

    def _save_combined_dataset(self, hrc_frame: pd.DataFrame) -> None:
        hkl_frame = self._normalize_hkl_dataset()
        combined = pd.concat([hkl_frame, hrc_frame], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=[
                "hospital_code",
                "category",
                "service_name",
                "charge_type",
                "patient_class",
                "ward_class",
                "room_type",
                "price_rm",
                "price_unit",
            ]
        )
        combined.to_csv(self.combined_output_path, index=False)

    def _normalize_hkl_dataset(self) -> pd.DataFrame:
        if not self.hkl_csv_path.exists():
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        legacy = pd.read_csv(self.hkl_csv_path)
        legacy["price_rm"] = pd.to_numeric(legacy["price_rm"], errors="coerce")
        legacy = legacy[
            legacy["patient_class"].fillna("").eq("all")
            & legacy["price_rm"].gt(0)
        ]
        hkl_source = PublicPricingSource(
            source_type="hospital",
            source_code="HKL",
            source_name="Hospital Kuala Lumpur Official Website",
            source_url="https://hkl.moh.gov.my/awam/caj-hospital",
            state="Kuala Lumpur",
            hospital_code="HKL",
            hospital_name="Hospital Kuala Lumpur",
        )
        records = [
            make_record(
                hkl_source,
                category=str(row.category).title(),
                service_name=str(row.service_name),
                original_service_name=str(row.service_name),
                charge_type=str(row.charge_type),
                patient_class="all",
                price_rm=float(row.price_rm),
                original_price_text=f"RM {float(row.price_rm):g}",
                scraped_at="",
            )
            for row in legacy.itertuples(index=False)
        ]
        return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def run_hrc_scraper() -> pd.DataFrame:
    return HRCHospitalChargesScraper().scrape()
