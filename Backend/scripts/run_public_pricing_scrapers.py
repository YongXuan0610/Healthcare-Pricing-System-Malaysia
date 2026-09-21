from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.scrapers.hcj_scraper import HCJHospitalChargesScraper
from app.scrapers.hpsf_scraper import HPSFHospitalChargesScraper
from app.scrapers.hrc_scraper import HRCHospitalChargesScraper
from app.scrapers.hsb_scraper import HSBHospitalChargesScraper
from app.scrapers.moh_pricing_scraper import (
    MOHTreatmentChargesScraper,
    MOHWardChargesScraper,
)
from app.scrapers.public_pricing_common import (
    PUBLIC_PRICE_COLUMNS,
    PublicPricingSource,
    ScrapeSummary,
    make_record,
    utc_scraped_at,
)


DATA_ROOT = BACKEND_ROOT / "app" / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"
RAW_ROOT = DATA_ROOT / "raw"
COMBINED_OUTPUT = PROCESSED_ROOT / "public_hospital_prices.csv"
MANIFEST_OUTPUT = PROCESSED_ROOT / "public_pricing_manifest.json"


def normalize_hkl() -> tuple[pd.DataFrame, ScrapeSummary]:
    """Migrate the trusted legacy HKL rows without changing their prices."""

    input_path = PROCESSED_ROOT / "hkl_public_charges.csv"
    output_path = PROCESSED_ROOT / "hkl_public_charges_normalized.csv"
    legacy = pd.read_csv(input_path)
    legacy["price_rm"] = pd.to_numeric(legacy["price_rm"], errors="coerce")
    # The legacy scraper emitted zero-valued section labels and used service names
    # as patient classes in malformed rows. Neither represents a published price.
    valid = legacy[
        legacy["patient_class"].fillna("").eq("all")
        & legacy["price_rm"].gt(0)
    ].copy()
    source = PublicPricingSource(
        source_type="hospital",
        source_code="HKL",
        source_name="Hospital Kuala Lumpur Official Website",
        source_url="https://hkl.moh.gov.my/awam/caj-hospital",
        state="Kuala Lumpur",
        hospital_code="HKL",
        hospital_name="Hospital Kuala Lumpur",
    )
    timestamp = utc_scraped_at()
    records = [
        make_record(
            source,
            category=str(row.category).title(),
            service_name=str(row.service_name),
            original_service_name=str(row.service_name),
            charge_type=str(row.charge_type),
            patient_class="all",
            price_rm=float(row.price_rm),
            original_price_text=f"RM {float(row.price_rm):g}",
            scraped_at=timestamp,
        )
        for row in valid.itertuples(index=False)
    ]
    frame = pd.DataFrame(records, columns=PUBLIC_PRICE_COLUMNS)
    frame = frame.drop_duplicates(
        subset=[
            "source_code", "category", "service_name", "charge_type",
            "patient_class", "ward_class", "room_type", "price_rm", "price_unit",
        ]
    ).reset_index(drop=True)
    frame.to_csv(output_path, index=False)
    summary = ScrapeSummary(
        source_code="HKL",
        records=len(frame),
        categories=sorted(frame["category"].unique(), key=str.casefold),
        duplicates_removed=len(valid) - len(frame),
        skipped_malformed=len(legacy) - len(valid),
        output_path=output_path,
    )
    return frame, summary


def run_all(*, offline: bool = False) -> tuple[pd.DataFrame, list[ScrapeSummary]]:
    hkl_frame, hkl_summary = normalize_hkl()

    hrc_scraper = HRCHospitalChargesScraper()
    hrc_html = (
        (RAW_ROOT / "hrc_ward_and_treatment_charges_raw.html").read_text(encoding="utf-8")
        if offline
        else None
    )
    hrc_frame = hrc_scraper.scrape(hrc_html)
    hrc_summary = ScrapeSummary(
        source_code="HRC",
        records=len(hrc_frame),
        categories=sorted(hrc_frame["category"].unique(), key=str.casefold),
        duplicates_removed=0,
        skipped_malformed=0,
        output_path=PROCESSED_ROOT / "hrc_public_charges.csv",
    )

    hcj_scraper = HCJHospitalChargesScraper()
    hcj_html = None
    if offline:
        hcj_html = (RAW_ROOT / "hcj_hospital_charges_raw.html").read_text(encoding="utf-8")
    hcj_frame, hcj_summary = hcj_scraper.scrape(hcj_html)

    hsb_scraper = HSBHospitalChargesScraper()
    hsb_pages = None
    if offline:
        snapshot = (RAW_ROOT / "hsb_hospital_charges_raw.html").read_text(encoding="utf-8")
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(snapshot, "html.parser")
        hsb_pages = {
            section.get("data-source-page", ""): "".join(str(child) for child in section.contents)
            for section in soup.select("section[data-source-page]")
        }
    hsb_frame, hsb_summary = hsb_scraper.scrape(hsb_pages)

    hpsf_frame, hpsf_summary = HPSFHospitalChargesScraper().scrape(
        (RAW_ROOT / "hpsf_hospital_charges_snapshot.html").read_text(encoding="utf-8")
        if offline
        else None
    )
    moh_ward_frame, moh_ward_summary = MOHWardChargesScraper().scrape(
        (RAW_ROOT / "moh_caj_wad_raw.html").read_text(encoding="utf-8")
        if offline
        else None
    )
    moh_treatment_frame, moh_treatment_summary = MOHTreatmentChargesScraper().scrape(
        (RAW_ROOT / "moh_caj_rawatan_raw.html").read_text(encoding="utf-8")
        if offline
        else None
    )
    if offline:
        hpsf_summary.used_snapshot_fallback = True
        moh_ward_summary.used_snapshot_fallback = True
        moh_treatment_summary.used_snapshot_fallback = True

    frames = [
        hkl_frame, hrc_frame, hcj_frame, hsb_frame, hpsf_frame,
        moh_ward_frame, moh_treatment_frame,
    ]
    combined = pd.concat(frames, ignore_index=True)[PUBLIC_PRICE_COLUMNS]
    before = len(combined)
    combined = combined.drop_duplicates(subset="id").reset_index(drop=True)
    if len(combined) != before:
        raise ValueError("Cross-source record IDs collided while building the combined dataset.")
    if set(combined["source_type"]) != {"hospital", "national_reference"}:
        raise ValueError("Combined data is missing a required source type.")
    if combined["price_rm"].isna().any() or combined["price_rm"].lt(0).any():
        raise ValueError("Combined data contains an invalid normalized price.")
    combined.to_csv(COMBINED_OUTPUT, index=False)

    summaries = [
        hkl_summary, hrc_summary, hcj_summary, hsb_summary, hpsf_summary,
        moh_ward_summary, moh_treatment_summary,
    ]
    manifest = {
        "generated_at": utc_scraped_at(),
        "combined_output": str(COMBINED_OUTPUT),
        "records": len(combined),
        "categories": sorted(combined["category"].unique(), key=str.casefold),
        "sources": [
            {
                "source_code": item.source_code,
                "records": item.records,
                "categories": item.categories,
                "duplicates_removed": item.duplicates_removed,
                "skipped_malformed": item.skipped_malformed,
                "output_path": str(item.output_path),
                "used_snapshot_fallback": item.used_snapshot_fallback,
            }
            for item in summaries
        ],
    }
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return combined, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh all official public pricing sources.")
    parser.add_argument("--offline", action="store_true", help="Parse committed raw snapshots without network requests.")
    args = parser.parse_args()
    combined, summaries = run_all(offline=args.offline)

    for summary in summaries:
        print(f"{summary.source_code}: {summary.records} records; {len(summary.categories)} categories")
        print(f"  duplicates removed: {summary.duplicates_removed}; malformed/skipped: {summary.skipped_malformed}")
        print(f"  output: {summary.output_path}")
        if summary.used_snapshot_fallback:
            print("  status: official-source snapshot fallback used")
    print(f"Combined: {len(combined)} records; {combined['category'].nunique()} categories")
    print(f"Output: {COMBINED_OUTPUT}")


if __name__ == "__main__":
    main()
