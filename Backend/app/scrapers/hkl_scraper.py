from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup


SOURCE_URL = "https://hkl.moh.gov.my/en/public/hospital-charges"
EXPECTED_COLUMNS = [
    "category",
    "service_name",
    "patient_class",
    "charge_type",
    "price_rm",
    "source_url",
]


class HKLHospitalChargesScraper:
    def __init__(self, source_url: str = SOURCE_URL) -> None:
        self.source_url = source_url

    def scrape(self) -> pd.DataFrame:
        html = self._fetch_page()
        raw_text = self._extract_raw_text(html)
        self._save_raw_text(raw_text)
        charge_rows = self._extract_charge_rows(html)

        if not charge_rows:
            raise ValueError("No hospital charge data could be extracted from the HKL page.")

        frame = pd.DataFrame(charge_rows)
        for column in EXPECTED_COLUMNS:
            if column not in frame.columns:
                frame[column] = ""

        frame = frame[EXPECTED_COLUMNS].copy()
        frame["price_rm"] = pd.to_numeric(frame["price_rm"], errors="coerce")
        frame = frame.dropna(subset=["price_rm"])
        frame = frame.drop_duplicates(subset=["category", "service_name", "patient_class", "charge_type", "price_rm"])

        self._save_processed_csv(frame)
        return frame

    def _fetch_page(self) -> str:
        try:
            response = requests.get(
                self.source_url,
                timeout=30,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                    )
                },
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to fetch HKL hospital charges page: {exc}") from exc

        response.encoding = response.apparent_encoding or response.encoding
        return response.text

    def _extract_raw_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text("\n", strip=True)

    def _save_raw_text(self, raw_text: str) -> None:
        output_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "hkl_hospital_charges_raw.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(raw_text, encoding="utf-8")

    def _save_processed_csv(self, frame: pd.DataFrame) -> None:
        output_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "hkl_public_charges.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)

    def _extract_charge_rows(self, html: str) -> list[dict[str, Any]]:
        lines = self._extract_visible_lines(html)
        charge_rows: list[dict[str, Any]] = []

        ward_start = self._find_line_index(lines, "WARD CHARGES")
        treatment_start = self._find_line_index(lines, "TREATMENT CHARGES")

        if ward_start != -1:
            ward_end = treatment_start if treatment_start != -1 else len(lines)
            charge_rows.extend(self._extract_ward_rows(lines[ward_start + 1 : ward_end]))

        if treatment_start != -1:
            charge_rows.extend(self._extract_treatment_rows(lines[treatment_start + 1 :]))

        return charge_rows

    def _extract_visible_lines(self, html: str) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text("\n", strip=True)
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _find_line_index(self, lines: list[str], needle: str) -> int:
        needle_lower = needle.lower()
        for index, line in enumerate(lines):
            if line.lower() == needle_lower:
                return index
        return -1

    def _extract_ward_rows(self, lines: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        current_section = "ward charges"
        current_class = ""
        pending_service_name = ""

        for index, line in enumerate(lines):
            normalized = line.strip()
            upper_line = normalized.upper()
            next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
            next_next_line = lines[index + 2].strip() if index + 2 < len(lines) else ""

            if upper_line in {"NO.", "CLASS", "CHARGE PER DAY (RM)", "CITIZEN", "FOREIGNER"}:
                pending_service_name = ""
                continue

            if upper_line in {"NON AIR CONDITIONED", "AIR CONDITIONED", "SPECIAL WARD"}:
                current_section = f"ward charges - {normalized.lower()}"
                pending_service_name = ""
                continue

            if upper_line in {"FIRST CLASS", "SECOND CLASS", "THIRD CLASS", "FOURTH CLASS"}:
                current_class = normalized.lower()
                if self._is_direct_class_price(next_line, next_next_line):
                    rows.append(
                        {
                            "category": "ward charges",
                            "service_name": normalized,
                            "patient_class": "all",
                            "charge_type": current_section,
                            "price_rm": self._parse_price(next_line),
                            "source_url": self.source_url,
                        }
                    )
                    continue
                pending_service_name = ""
                continue

            if self._is_numeric(normalized):
                if pending_service_name:
                    rows.append(
                        {
                            "category": "ward charges",
                            "service_name": pending_service_name,
                            "patient_class": "all",
                            "charge_type": current_class or current_section,
                            "price_rm": self._parse_price(normalized),
                            "source_url": self.source_url,
                        }
                    )
                    pending_service_name = ""
                continue

            pending_service_name = normalized

        return rows

    def _extract_treatment_rows(self, lines: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        current_section = "common public hospital charges"

        for index, line in enumerate(lines):
            normalized = line.strip()
            upper_line = normalized.upper()
            next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
            next_next_line = lines[index + 2].strip() if index + 2 < len(lines) else ""

            if upper_line in {"PSIKIATRIK", "FISIOTERAPI", "RADIOTERAPI & ONKOLOGI", "NEFROLOGI", "OCCUTERAPI", "OCCUTERAPI2", "LINGUATHERAPY", "DIETETIK", "TRADISI", "SIASATAN RADIO", "UJIAN", "SIASATAN", "PERGIGIAN"}:
                rows.append(
                    {
                        "category": current_section,
                        "service_name": normalized,
                        "patient_class": "all",
                        "charge_type": "listed_service",
                        "price_rm": 0,
                        "source_url": self.source_url,
                    }
                )
                continue

            if upper_line == "PER ADMISSION (RM)":
                rows.extend(self._extract_admission_table_rows(lines[index + 1 :]))
                break

            if self._is_direct_class_price(next_line, next_next_line) and upper_line in {"FIRST CLASS", "SECOND CLASS", "THIRD CLASS"}:
                rows.append(
                    {
                        "category": current_section,
                        "service_name": normalized,
                        "patient_class": normalized.lower(),
                        "charge_type": "admission_rate",
                        "price_rm": self._parse_price(next_line),
                        "source_url": self.source_url,
                    }
                )

        return rows

    def _extract_admission_table_rows(self, lines: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        index = 0

        while index < len(lines):
            line = lines[index].strip()
            next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
            next_next_line = lines[index + 2].strip() if index + 2 < len(lines) else ""

            if line.upper() == "NO.":
                index += 1
                continue

            if line in {"1", "2", "3"} and next_line and not self._is_numeric(next_line):
                if self._is_direct_class_price(next_next_line, lines[index + 3].strip() if index + 3 < len(lines) else ""):
                    rows.append(
                        {
                            "category": "common public hospital charges",
                            "service_name": next_line,
                            "patient_class": next_line.lower(),
                            "charge_type": "per_admission",
                            "price_rm": self._parse_price(next_next_line),
                            "source_url": self.source_url,
                        }
                    )
                    index += 3
                    continue

            if line.upper() in {"FIRST CLASS", "SECOND CLASS", "THIRD CLASS"} and self._is_numeric(next_line):
                rows.append(
                    {
                        "category": "common public hospital charges",
                        "service_name": line,
                        "patient_class": line.lower(),
                        "charge_type": "per_admission",
                        "price_rm": self._parse_price(next_line),
                        "source_url": self.source_url,
                    }
                )

            index += 1

        return rows

    def _is_numeric(self, value: str) -> bool:
        return bool(re.fullmatch(r"\d+(?:\.\d+)?", value.strip()))

    def _is_direct_class_price(self, next_line: str, next_next_line: str) -> bool:
        if not self._is_numeric(next_line):
            return False
        if self._is_numeric(next_next_line):
            return True
        return next_next_line.upper() in {"AIR CONDITIONED", "SPECIAL WARD", "TREATMENT CHARGES"}

    def _parse_price(self, value: str) -> str:
        match = re.search(r"(\d[\d,]*(?:\.\d+)?)", value)
        return match.group(1).replace(",", "") if match else "0"

    def _clean_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        return str(value).replace("\xa0", " ").strip()


def run_hkl_scraper() -> pd.DataFrame:
    scraper = HKLHospitalChargesScraper()
    return scraper.scrape()
