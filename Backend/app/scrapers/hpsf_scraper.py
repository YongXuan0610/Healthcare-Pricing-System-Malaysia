from __future__ import annotations

from pathlib import Path
import re

from bs4 import BeautifulSoup

from app.scrapers.public_pricing_common import (
    PublicPricingSource,
    clean_text,
    fetch_official_html,
    finalize_records,
    make_record,
    parse_price,
    patient_class_from_label,
    table_grid,
    utc_scraped_at,
    ward_class_from_text,
)


SOURCE = PublicPricingSource(
    source_type="hospital",
    source_code="HPSF",
    source_name="Hospital Pakar Sultanah Fatimah Official Website",
    source_url="https://jknjohor.moh.gov.my/hpsf/caj-bayaran/",
    state="Johor",
    hospital_code="HPSF",
    hospital_name="Hospital Pakar Sultanah Fatimah, Muar",
    source_updated_at="2024-04-17",
)


class HPSFHospitalChargesScraper:
    def __init__(self, *, output_path: str | Path | None = None) -> None:
        data_root = Path(__file__).resolve().parents[1] / "data"
        self.output_path = Path(output_path) if output_path else (
            data_root / "processed" / "hpsf_public_charges.csv"
        )
        self.snapshot_path = data_root / "raw" / "hpsf_hospital_charges_snapshot.html"

    def scrape(self, html: str | None = None, *, scraped_at: str | None = None, save: bool = True):
        used_fallback = False
        if html is None:
            html, used_fallback = fetch_official_html(
                SOURCE,
                fallback_path=self.snapshot_path,
            )
        timestamp = scraped_at or utc_scraped_at()
        records, skipped = self.parse_html(html, scraped_at=timestamp)
        frame, summary = finalize_records(
            records,
            source=SOURCE,
            output_path=self.output_path,
            skipped_malformed=skipped,
            used_snapshot_fallback=used_fallback,
        )
        if frame.empty:
            raise ValueError("No structured prices were extracted from the HPSF page.")
        return frame, summary

    @staticmethod
    def _header_metadata(header: str) -> tuple[str, str]:
        lowered = clean_text(header).casefold()
        if "foreigner" in lowered:
            patient_class = "foreigner"
        elif "resident" in lowered:
            patient_class = "resident"
        elif "citizen" in lowered or "eligible" in lowered:
            patient_class = "citizen"
        else:
            patient_class = patient_class_from_label(header)
        match = re.search(r"class\s*([123])", lowered)
        ward_class = f"Class {match.group(1)}" if match else ""
        return patient_class, ward_class

    @classmethod
    def parse_html(cls, html: str, *, scraped_at: str) -> tuple[list[dict], int]:
        soup = BeautifulSoup(html, "html.parser")
        records: list[dict] = []
        skipped = 0

        for table in soup.find_all("table"):
            category = clean_text(table.get("data-category"))
            charge_type = clean_text(table.get("data-charge-type"))
            price_unit = clean_text(table.get("data-unit"))
            if not category or not charge_type:
                continue
            grid = table_grid(table)
            if len(grid) < 2:
                continue
            headers = grid[0]
            for row in grid[1:]:
                if not row or not row[0]:
                    continue
                service = row[0]
                for index, header in enumerate(headers[1:], start=1):
                    price_text = row[index] if index < len(row) else ""
                    price = parse_price(price_text)
                    if price is None:
                        if clean_text(price_text) not in {"", "-", "–", "—"}:
                            skipped += 1
                        continue
                    patient_class, ward_class = cls._header_metadata(header)
                    if not patient_class:
                        skipped += 1
                        continue
                    service_ward_class = ward_class_from_text(service)
                    ward_class = ward_class or service_ward_class
                    records.append(make_record(
                        SOURCE,
                        category=category,
                        service_name=service,
                        original_service_name=service,
                        charge_type=charge_type,
                        patient_class=patient_class,
                        ward_class=ward_class,
                        room_type=service if category == "Ward Charges" else "",
                        price_rm=price,
                        price_unit=price_unit,
                        original_price_text=price_text,
                        notes="",
                        scraped_at=scraped_at,
                    ))

        return records, skipped


def run_hpsf_scraper():
    return HPSFHospitalChargesScraper().scrape()
