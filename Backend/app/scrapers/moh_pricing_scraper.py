from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from app.scrapers.public_pricing_common import (
    PublicPricingSource,
    clean_text,
    fetch_official_html,
    finalize_records,
    infer_price_unit,
    make_record,
    parse_price,
    table_grid,
    utc_scraped_at,
    ward_class_from_text,
)


MOH_WARD_SOURCE = PublicPricingSource(
    source_type="national_reference",
    source_code="MOH_WARD",
    source_name="Ministry of Health Malaysia – Caj Wad",
    source_url="https://www.moh.gov.my/teras/info-kesihatan/kadar-perkhidmatan/caj-wad",
    state="National",
    source_updated_at="2025-09-23",
)
MOH_TREATMENT_SOURCE = PublicPricingSource(
    source_type="national_reference",
    source_code="MOH_TREATMENT",
    source_name="Ministry of Health Malaysia – Caj Rawatan",
    source_url="https://www.moh.gov.my/teras/info-kesihatan/kadar-perkhidmatan/caj-rawatan",
    state="National",
    source_updated_at="2025-09-23",
)


class MOHWardChargesScraper:
    def __init__(self, *, output_path: str | Path | None = None) -> None:
        data_root = Path(__file__).resolve().parents[1] / "data"
        self.snapshot_path = data_root / "raw" / "moh_caj_wad_raw.html"
        self.output_path = Path(output_path) if output_path else (
            data_root / "processed" / "moh_caj_wad.csv"
        )

    def scrape(self, html: str | None = None, *, scraped_at: str | None = None, save: bool = True):
        used_fallback = False
        if html is None:
            html, used_fallback = fetch_official_html(
                MOH_WARD_SOURCE,
                fallback_path=self.snapshot_path,
            )
        records, skipped = self.parse_html(html, scraped_at=scraped_at or utc_scraped_at())
        frame, summary = finalize_records(
            records,
            source=MOH_WARD_SOURCE,
            output_path=self.output_path,
            skipped_malformed=skipped,
            used_snapshot_fallback=used_fallback,
        )
        if frame.empty:
            raise ValueError("No prices were extracted from the MOH Caj Wad source.")
        return frame, summary

    @staticmethod
    def parse_html(html: str, *, scraped_at: str) -> tuple[list[dict], int]:
        tables = BeautifulSoup(html, "html.parser").find_all("table")
        if len(tables) < 4:
            raise ValueError(f"Expected 4 MOH Caj Wad tables, found {len(tables)}.")
        records: list[dict] = []
        skipped = 0

        def add(service: str, price_text: str, *, ward_class: str = "", room_type: str = "", charge_type: str = "ward charge", unit: str = "per day") -> None:
            nonlocal skipped
            price = parse_price(price_text)
            if price is None:
                skipped += 1
                return
            records.append(make_record(
                MOH_WARD_SOURCE,
                category="Ward Charges",
                service_name=service,
                original_service_name=service,
                charge_type=charge_type,
                patient_class="citizen",
                ward_class=ward_class,
                room_type=room_type,
                price_rm=price,
                price_unit=unit,
                original_price_text=price_text,
                scraped_at=scraped_at,
            ))

        general = table_grid(tables[0])
        for row in general[1:]:
            if len(row) < 3 or not row[0] or (not row[1] and not row[2]):
                continue
            ward_class = ward_class_from_text(row[0]) or "Class 1"
            for room_type, price_text in zip(general[0][1:3], row[1:3]):
                add(
                    f"{row[0]} - {room_type}",
                    price_text,
                    ward_class=ward_class,
                    room_type=f"{row[0]} - {room_type}",
                )

        for row in table_grid(tables[1]):
            if len(row) >= 2:
                add(row[0], row[1], ward_class="Class 1", charge_type="facility admission charge", unit="per admission")

        for row in table_grid(tables[2])[1:]:
            if len(row) >= 2:
                add(f"Psychiatric ward - {row[0]}", row[1], ward_class=ward_class_from_text(row[0]))

        for row in table_grid(tables[3])[1:]:
            if len(row) >= 2:
                add(f"Executive ward - {row[0]}", row[1], room_type=row[0])

        return records, skipped


class MOHTreatmentChargesScraper:
    TABLE_CATEGORIES = {
        0: "Inpatient Treatment",
        1: "Delivery / Maternity",
        2: "Physiotherapy",
        3: "Radiotherapy / Oncology",
        4: "Nephrology",
        5: "Inpatient Treatment",
        6: "Occupational Therapy",
        7: "Speech Therapy",
        8: "Speech Therapy",
        9: "Dietetics",
        10: "Dietetics",
        11: "Traditional and Complementary Medicine",
    }

    def __init__(self, *, output_path: str | Path | None = None) -> None:
        data_root = Path(__file__).resolve().parents[1] / "data"
        self.snapshot_path = data_root / "raw" / "moh_caj_rawatan_raw.html"
        self.output_path = Path(output_path) if output_path else (
            data_root / "processed" / "moh_caj_rawatan.csv"
        )

    def scrape(self, html: str | None = None, *, scraped_at: str | None = None, save: bool = True):
        used_fallback = False
        if html is None:
            html, used_fallback = fetch_official_html(
                MOH_TREATMENT_SOURCE,
                fallback_path=self.snapshot_path,
            )
        records, skipped = self.parse_html(html, scraped_at=scraped_at or utc_scraped_at())
        frame, summary = finalize_records(
            records,
            source=MOH_TREATMENT_SOURCE,
            output_path=self.output_path,
            skipped_malformed=skipped,
            used_snapshot_fallback=used_fallback,
        )
        if frame.empty:
            raise ValueError("No prices were extracted from the MOH Caj Rawatan source.")
        return frame, summary

    @staticmethod
    def _column_metadata(header: str) -> tuple[str, str]:
        text = clean_text(header)
        ward_class = ward_class_from_text(text)
        lowered = text.casefold()
        if "pesakit luar" in lowered:
            return "Outpatient", ""
        if "persendirian" in lowered:
            return "Outpatient - private practitioner referral", ""
        if "kerajaan" in lowered:
            return "Outpatient - government medical officer referral", ""
        return "Inpatient" if ward_class else text, ward_class

    @classmethod
    def parse_html(cls, html: str, *, scraped_at: str) -> tuple[list[dict], int]:
        tables = BeautifulSoup(html, "html.parser").find_all("table")
        if len(tables) < 12:
            raise ValueError(f"Expected 12 MOH Caj Rawatan tables, found {len(tables)}.")
        records: list[dict] = []
        skipped = 0

        for table_index, table in enumerate(tables[:12]):
            grid = table_grid(table)
            if not grid:
                continue
            category = cls.TABLE_CATEGORIES[table_index]
            header_index = 0
            if table_index in {2, 3}:
                header_index = 1
            headers = grid[header_index]
            start_index = header_index + 1

            # Tables 0 and 5 put ward class in the first column and one price in the second.
            if table_index in {0, 5}:
                for row in grid[start_index:]:
                    if len(row) < 2:
                        continue
                    price = parse_price(row[1])
                    if price is None:
                        skipped += 1
                        continue
                    records.append(make_record(
                        MOH_TREATMENT_SOURCE,
                        category=category,
                        service_name=("Daily inpatient treatment" if table_index == 0 else "Psychiatric inpatient treatment"),
                        original_service_name=row[0],
                        charge_type="inpatient treatment charge",
                        patient_class="citizen",
                        patient_type="Inpatient",
                        ward_class=ward_class_from_text(row[0]),
                        price_rm=price,
                        price_unit="per day" if table_index == 0 else "per admission",
                        original_price_text=row[1],
                        scraped_at=scraped_at,
                    ))
                continue

            for row in grid[start_index:]:
                if not row or not clean_text(row[0]):
                    continue
                service = clean_text(row[0])
                for column_index, header in enumerate(headers[1:], start=1):
                    price_text = row[column_index] if column_index < len(row) else ""
                    price = parse_price(price_text)
                    if price is None:
                        continue
                    patient_type, ward_class = cls._column_metadata(header)
                    charge_type = {
                        "Delivery / Maternity": "delivery charge",
                        "Physiotherapy": "physiotherapy treatment",
                        "Occupational Therapy": "occupational therapy service",
                        "Speech Therapy": "speech therapy service",
                        "Dietetics": "dietetics service",
                    }.get(category, "treatment charge")
                    records.append(make_record(
                        MOH_TREATMENT_SOURCE,
                        category=category,
                        service_name=service,
                        original_service_name=service,
                        charge_type=charge_type,
                        patient_class="citizen",
                        patient_type=patient_type,
                        ward_class=ward_class,
                        price_rm=price,
                        price_unit=infer_price_unit(service, header),
                        original_price_text=price_text,
                        scraped_at=scraped_at,
                    ))

        return records, skipped


def run_moh_ward_scraper():
    return MOHWardChargesScraper().scrape()


def run_moh_treatment_scraper():
    return MOHTreatmentChargesScraper().scrape()
