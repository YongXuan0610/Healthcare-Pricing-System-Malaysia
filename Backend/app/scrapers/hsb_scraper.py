from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from app.scrapers.public_pricing_common import (
    PublicPricingSource,
    ScrapeSummary,
    clean_text,
    extract_source_updated_at,
    fetch_official_html,
    finalize_records,
    make_record,
    parse_all_prices,
    parse_price,
    patient_class_from_label,
    table_grid,
    utc_scraped_at,
    ward_class_from_text,
)


SOURCE = PublicPricingSource(
    source_type="hospital",
    source_code="HSB",
    source_name="Hospital Sungai Buloh Official Website",
    source_url="https://jknselangor.moh.gov.my/hsgbuloh/index.php/ms/lihat-caj-bayaran",
    state="Selangor",
    hospital_code="HSB",
    hospital_name="Hospital Sungai Buloh",
)

PAGE_URLS = {
    "specialist": "https://jknselangor.moh.gov.my/hsgbuloh/index.php/ms/component/content/article/86?Itemid=134",
    "deposit": "https://jknselangor.moh.gov.my/hsgbuloh/index.php/ms/component/content/article/135?Itemid=177",
    "ward": "https://jknselangor.moh.gov.my/hsgbuloh/index.php/ms/component/content/article/1230?Itemid=136",
    "delivery": "https://jknselangor.moh.gov.my/hsgbuloh/index.php/ms/component/content/article/89?Itemid=137",
    "reports": "https://jknselangor.moh.gov.my/hsgbuloh/index.php/ms/component/content/article/134?Itemid=176",
    "emergency": "https://jknselangor.moh.gov.my/hsgbuloh/index.php/ms/component/content/article/732?Itemid=682",
}


class HSBHospitalChargesScraper:
    def __init__(
        self,
        *,
        output_path: str | Path | None = None,
        raw_output_path: str | Path | None = None,
    ) -> None:
        data_root = Path(__file__).resolve().parents[1] / "data"
        self.output_path = Path(output_path) if output_path else (
            data_root / "processed" / "hsb_public_charges.csv"
        )
        self.raw_output_path = Path(raw_output_path) if raw_output_path else (
            data_root / "raw" / "hsb_hospital_charges_raw.html"
        )

    def scrape(
        self,
        pages: dict[str, str] | None = None,
        *,
        scraped_at: str | None = None,
        save: bool = True,
    ):
        used_fallback = False
        if pages is None:
            pages = {}
            for name, url in PAGE_URLS.items():
                page_source = PublicPricingSource(**{**SOURCE.__dict__, "source_url": url})
                pages[name], page_fallback = fetch_official_html(page_source)
                used_fallback = used_fallback or page_fallback

        timestamp = scraped_at or utc_scraped_at()
        records, skipped = self.parse_pages(pages, scraped_at=timestamp)
        if save:
            self.raw_output_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot = ["<!doctype html><html><head><meta charset='utf-8'></head><body>"]
            for name, html in pages.items():
                snapshot.extend((f"<section data-source-page='{name}'>", html, "</section>"))
            snapshot.append("</body></html>")
            self.raw_output_path.write_text("\n".join(snapshot), encoding="utf-8")

        frame, summary = finalize_records(
            records,
            source=SOURCE,
            output_path=self.output_path,
            skipped_malformed=skipped,
            used_snapshot_fallback=used_fallback,
        )
        if frame.empty:
            raise ValueError("No structured prices were extracted from the HSB pages.")
        return frame, summary

    @staticmethod
    def _table(page: str) -> list[list[str]]:
        table = BeautifulSoup(page, "html.parser").find("table")
        if table is None:
            return []
        return table_grid(table)

    @staticmethod
    def _tables(page: str) -> list[list[list[str]]]:
        soup = BeautifulSoup(page, "html.parser")
        return [table_grid(table) for table in soup.find_all("table")]

    @classmethod
    def parse_pages(cls, pages: dict[str, str], *, scraped_at: str) -> tuple[list[dict], int]:
        records: list[dict] = []
        skipped = 0
        updated_dates = [extract_source_updated_at(html) for html in pages.values()]
        source_updated_at = next((date for date in updated_dates if date), "")

        def add(
            *,
            category: str,
            service: str,
            charge_type: str,
            patient_class: str,
            price_text: str,
            ward_class: str = "",
            patient_type: str = "",
            unit: str = "",
            notes: str = "",
            source_url: str = SOURCE.source_url,
        ) -> None:
            nonlocal skipped
            price = parse_price(price_text)
            if price is None:
                skipped += 1
                return
            page_source = PublicPricingSource(**{**SOURCE.__dict__, "source_url": source_url})
            records.append(make_record(
                page_source,
                category=category,
                service_name=service,
                original_service_name=service,
                charge_type=charge_type,
                patient_class=patient_class,
                patient_type=patient_type,
                ward_class=ward_class,
                price_rm=price,
                price_unit=unit,
                original_price_text=price_text,
                notes=notes,
                scraped_at=scraped_at,
                source_updated_at=source_updated_at,
            ))

        # Specialist-clinic page: referral source, citizen, foreigner and UNHCR.
        specialist = cls._table(pages.get("specialist", ""))
        for row in specialist[2:]:
            if len(row) < 4 or not clean_text(row[0]):
                continue
            service = clean_text(row[0])
            for label, price_text in zip(("Warganegara", "Bukan Warganegara", "UNHCR"), row[1:4]):
                prices = parse_all_prices(price_text)
                if not prices:
                    skipped += 1
                    continue
                for index, price in enumerate(prices):
                    visit_type = ""
                    lowered = price_text.casefold()
                    if len(prices) > 1:
                        visit_type = "first consultation" if index == 0 else "subsequent visit"
                    elif "pertama" in lowered or "first" in lowered:
                        visit_type = "first consultation"
                    elif "ulangan" in lowered or "subsequent" in lowered:
                        visit_type = "subsequent visit"
                    page_source = PublicPricingSource(**{**SOURCE.__dict__, "source_url": PAGE_URLS["specialist"]})
                    records.append(make_record(
                        page_source,
                        category="Specialist Clinic",
                        service_name=f"{service}{' - ' + visit_type if visit_type else ''}",
                        original_service_name=service,
                        charge_type="specialist clinic consultation",
                        patient_class=patient_class_from_label(label),
                        patient_type=visit_type or service,
                        price_rm=price,
                        price_unit="per visit",
                        original_price_text=price_text,
                        scraped_at=scraped_at,
                        source_updated_at=source_updated_at,
                    ))

        # Admission deposits: discipline x classes/citizenship.
        deposit = cls._table(pages.get("deposit", ""))
        for row in deposit[3:]:
            if len(row) < 5 or not row[0]:
                continue
            for patient_class, ward_class, price_text in (
                ("citizen", "Class 2", row[1]),
                ("citizen", "Class 3", row[2]),
                ("foreigner", "Class 2", row[3]),
                ("foreigner", "Class 3", row[4]),
            ):
                add(
                    category="Ward Deposit",
                    service=row[0],
                    charge_type="ward admission deposit",
                    patient_class=patient_class,
                    ward_class=ward_class,
                    price_text=price_text,
                    unit="deposit",
                    source_url=PAGE_URLS["deposit"],
                )

        # Ward page: ward and daily treatment rows plus foreigner-only facilities.
        ward_tables = cls._tables(pages.get("ward", ""))
        for table_index, ward in enumerate(ward_tables[:2]):
            category = "Ward Charges" if table_index == 0 else "Inpatient Treatment"
            charge_type = "ward charge" if table_index == 0 else "inpatient treatment charge"
            for row in ward[1:]:
                if len(row) < 3 or not ward_class_from_text(row[0]):
                    continue
                ward_class = ward_class_from_text(row[0])
                for patient_class, price_text in (("citizen", row[1]), ("foreigner", row[2])):
                    add(
                        category=category,
                        service=row[0],
                        charge_type=charge_type,
                        patient_class=patient_class,
                        ward_class=ward_class,
                        price_text=price_text,
                        unit="per day",
                        source_url=PAGE_URLS["ward"],
                    )
        if len(ward_tables) > 2:
            for row in ward_tables[2][1:]:
                if len(row) < 2 or not row[0]:
                    continue
                add(
                    category="Other Published Charges",
                    service=row[0],
                    charge_type="facility charge",
                    patient_class="foreigner",
                    price_text=row[1],
                    unit="per day",
                    source_url=PAGE_URLS["ward"],
                )

        # Delivery page: Malaysian classes and foreigner schedule.
        delivery_tables = cls._tables(pages.get("delivery", ""))
        if delivery_tables:
            citizen_delivery = delivery_tables[0]
            headers = citizen_delivery[0]
            for row in citizen_delivery[1:]:
                if not row or not row[0]:
                    continue
                ward_class = ward_class_from_text(row[0])
                for index, service in enumerate(headers[1:], start=1):
                    add(category="Delivery / Maternity", service=service, charge_type="delivery charge", patient_class="citizen", ward_class=ward_class, price_text=row[index], source_url=PAGE_URLS["delivery"])
        if len(delivery_tables) > 1:
            foreign_delivery = delivery_tables[1]
            headers = foreign_delivery[0]
            for index, service in enumerate(headers):
                add(category="Delivery / Maternity", service=service, charge_type="delivery charge", patient_class="foreigner", price_text=foreign_delivery[1][index], source_url=PAGE_URLS["delivery"])

        # Medical reports; ranges become explicitly labelled lower/upper published values.
        reports = cls._table(pages.get("reports", ""))
        for row in reports[2:]:
            if len(row) < 3 or not row[0]:
                continue
            for patient_class, price_text in (("citizen", row[1]), ("foreigner", row[3])):
                prices = parse_all_prices(price_text, allow_bare=True)
                if not prices:
                    skipped += 1
                    continue
                for index, price in enumerate(prices):
                    suffix = ""
                    if len(prices) > 1:
                        suffix = " - published lower value" if index == 0 else " - published upper value"
                    page_source = PublicPricingSource(**{**SOURCE.__dict__, "source_url": PAGE_URLS["reports"]})
                    records.append(make_record(
                        page_source,
                        category="Medical Reports",
                        service_name=f"{row[0]}{suffix}",
                        original_service_name=row[0],
                        charge_type="medical report fee",
                        patient_class=patient_class,
                        price_rm=price,
                        original_price_text=price_text,
                        notes="Published range endpoint." if suffix else "",
                        scraped_at=scraped_at,
                        source_updated_at=source_updated_at,
                    ))

        # Emergency registration and observation-bed fees.
        emergency = cls._table(pages.get("emergency", ""))
        for row in emergency[1:]:
            if len(row) < 3 or not row[0]:
                continue
            add(category="Emergency", service=row[0], charge_type="emergency charge", patient_class="citizen", price_text=row[1], source_url=PAGE_URLS["emergency"])
            foreign_prices = parse_all_prices(row[2])
            for index, patient_class in enumerate(("foreigner", "unhcr")):
                if index >= len(foreign_prices):
                    skipped += 1
                    continue
                page_source = PublicPricingSource(**{**SOURCE.__dict__, "source_url": PAGE_URLS["emergency"]})
                records.append(make_record(
                    page_source,
                    category="Emergency",
                    service_name=row[0],
                    original_service_name=row[0],
                    charge_type="emergency charge",
                    patient_class=patient_class,
                    price_rm=foreign_prices[index],
                    original_price_text=row[2],
                    scraped_at=scraped_at,
                    source_updated_at=source_updated_at,
                ))

        return records, skipped


def run_hsb_scraper():
    return HSBHospitalChargesScraper().scrape()
