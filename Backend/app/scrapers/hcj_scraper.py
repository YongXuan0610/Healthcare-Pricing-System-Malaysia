from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from app.scrapers.public_pricing_common import (
    PublicPricingSource,
    ScrapeSummary,
    extract_source_updated_at,
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
    source_code="HCJ",
    source_name="Hospital Cyberjaya Official Website",
    # The official www hostname currently presents a mismatched TLS certificate;
    # the same JKN Selangor page is available on its canonical non-www hostname.
    source_url="https://jknselangor.moh.gov.my/hcj/index.php/awam/caj-hospital",
    state="Selangor",
    hospital_code="HCJ",
    hospital_name="Hospital Cyberjaya",
)


class HCJHospitalChargesScraper:
    def __init__(
        self,
        *,
        output_path: str | Path | None = None,
        raw_output_path: str | Path | None = None,
    ) -> None:
        data_root = Path(__file__).resolve().parents[1] / "data"
        self.output_path = Path(output_path) if output_path else (
            data_root / "processed" / "hcj_public_charges.csv"
        )
        self.raw_output_path = Path(raw_output_path) if raw_output_path else (
            data_root / "raw" / "hcj_hospital_charges_raw.html"
        )

    def scrape(
        self,
        html: str | None = None,
        *,
        scraped_at: str | None = None,
        save: bool = True,
    ) -> tuple[object, ScrapeSummary]:
        used_fallback = False
        if html is None:
            html, used_fallback = fetch_official_html(SOURCE)
        timestamp = scraped_at or utc_scraped_at()
        records, skipped = self.parse_html(html, scraped_at=timestamp)
        if save:
            self.raw_output_path.parent.mkdir(parents=True, exist_ok=True)
            self.raw_output_path.write_text(html, encoding="utf-8")
        frame, summary = finalize_records(
            records,
            source=SOURCE,
            output_path=self.output_path,
            skipped_malformed=skipped,
            used_snapshot_fallback=used_fallback,
        )
        if frame.empty:
            raise ValueError("No structured prices were extracted from the HCJ page.")
        return frame, summary

    @staticmethod
    def parse_html(html: str, *, scraped_at: str) -> tuple[list[dict], int]:
        tables = BeautifulSoup(html, "html.parser").find_all("table")
        if len(tables) < 6:
            raise ValueError(f"Expected at least 6 HCJ pricing tables, found {len(tables)}.")

        source_updated_at = extract_source_updated_at(html)
        records: list[dict] = []
        skipped = 0

        # Ward charges: room/class rows x patient classes.
        for row in table_grid(tables[1])[1:]:
            if len(row) < 4:
                skipped += 1
                continue
            ward_class = ward_class_from_text(row[0])
            room_type = row[1]
            service = " - ".join(part for part in (row[0], row[1]) if part)
            for patient_label, price_text in zip(("Warganegara", "Bukan Warganegara"), row[2:4]):
                price = parse_price(price_text)
                if price is None:
                    skipped += 1
                    continue
                patient_class = patient_class_from_label(patient_label)
                records.append(make_record(
                    SOURCE,
                    category="Ward Charges",
                    service_name=service,
                    original_service_name=service,
                    charge_type="ward charge",
                    patient_class=patient_class,
                    ward_class=ward_class,
                    room_type=room_type,
                    price_rm=price,
                    price_unit="per day",
                    original_price_text=price_text,
                    scraped_at=scraped_at,
                    source_updated_at=source_updated_at,
                ))

        # Daily inpatient treatment: patient rows x ward classes.
        treatment = table_grid(tables[2])
        for row in treatment[1:]:
            if not row:
                continue
            patient_class = patient_class_from_label(row[0])
            if not patient_class:
                skipped += 1
                continue
            for class_index, price_text in enumerate(row[1:4], start=1):
                price = parse_price(price_text)
                if price is None:
                    skipped += 1
                    continue
                records.append(make_record(
                    SOURCE,
                    category="Inpatient Treatment",
                    service_name=f"Daily inpatient treatment - Class {class_index}",
                    original_service_name=row[0],
                    charge_type="inpatient treatment charge",
                    patient_class=patient_class,
                    ward_class=f"Class {class_index}",
                    price_rm=price,
                    price_unit="per day",
                    original_price_text=price_text,
                    scraped_at=scraped_at,
                    source_updated_at=source_updated_at,
                ))

        # Malaysian delivery charges, Classes 2 and 3.
        delivery_citizen = table_grid(tables[3])
        for row in delivery_citizen[1:]:
            if len(row) < 3 or not row[0]:
                continue
            for ward_class, price_text in zip(("Class 2", "Class 3"), row[1:3]):
                price = parse_price(price_text)
                if price is None:
                    skipped += 1
                    continue
                records.append(make_record(
                    SOURCE,
                    category="Delivery / Maternity",
                    service_name=row[0],
                    charge_type="delivery charge",
                    patient_class="citizen",
                    ward_class=ward_class,
                    price_rm=price,
                    original_price_text=price_text,
                    scraped_at=scraped_at,
                    source_updated_at=source_updated_at,
                ))

        # Non-Malaysian delivery charges.
        delivery_foreigner = table_grid(tables[4])
        for row in delivery_foreigner[1:]:
            if len(row) < 2 or not row[0]:
                continue
            price = parse_price(row[1])
            if price is None:
                skipped += 1
                continue
            records.append(make_record(
                SOURCE,
                category="Delivery / Maternity",
                service_name=row[0],
                charge_type="delivery charge",
                patient_class="foreigner",
                price_rm=price,
                original_price_text=row[1],
                scraped_at=scraped_at,
                source_updated_at=source_updated_at,
            ))

        # Specialist clinic: each visit/referral row has a citizen and foreign rate.
        specialist = table_grid(tables[5])
        for row in specialist[1:]:
            if len(row) < 3 or not row[0]:
                continue
            for patient_label, price_text in zip(("Warganegara", "Bukan Warganegara"), row[1:3]):
                price = parse_price(price_text)
                if price is None:
                    skipped += 1
                    continue
                records.append(make_record(
                    SOURCE,
                    category="Specialist Clinic",
                    service_name=row[0],
                    charge_type="specialist clinic consultation",
                    patient_class=patient_class_from_label(patient_label),
                    patient_type=row[0],
                    price_rm=price,
                    price_unit="per visit",
                    original_price_text=price_text,
                    scraped_at=scraped_at,
                    source_updated_at=source_updated_at,
                ))

        return records, skipped


def run_hcj_scraper():
    return HCJHospitalChargesScraper().scrape()
