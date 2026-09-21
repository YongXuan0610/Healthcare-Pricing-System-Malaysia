from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from app.scrapers.hcj_scraper import HCJHospitalChargesScraper
from app.scrapers.hpsf_scraper import HPSFHospitalChargesScraper
from app.scrapers.hsb_scraper import HSBHospitalChargesScraper
from app.scrapers.moh_pricing_scraper import MOHTreatmentChargesScraper, MOHWardChargesScraper
from app.scrapers.public_pricing_common import PUBLIC_PRICE_COLUMNS
from app.services.public_pricing_service import PublicPricingService
from scripts import main


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = BACKEND_ROOT / "app" / "data"
PROCESSED = DATA_ROOT / "processed"
RAW = DATA_ROOT / "raw"
COMBINED = PROCESSED / "public_hospital_prices.csv"


class MultiSourceDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = pd.read_csv(COMBINED)

    def test_combined_dataset_uses_shared_schema_and_all_sources(self) -> None:
        self.assertEqual(list(self.frame.columns), PUBLIC_PRICE_COLUMNS)
        self.assertEqual(
            set(self.frame["source_code"]),
            {"HKL", "HRC", "HCJ", "HSB", "HPSF", "MOH_WARD", "MOH_TREATMENT"},
        )
        self.assertEqual(set(self.frame["source_type"]), {"hospital", "national_reference"})
        self.assertFalse(self.frame["id"].duplicated().any())
        self.assertTrue(self.frame["price_rm"].ge(0).all())

    def test_national_references_are_not_hospitals(self) -> None:
        references = self.frame[self.frame["source_type"].eq("national_reference")]
        self.assertTrue(references["hospital_code"].isna().all())
        self.assertTrue(references["hospital_name"].isna().all())
        self.assertEqual(set(references["state"]), {"National"})

    def test_committed_source_counts_match_validated_refresh(self) -> None:
        self.assertEqual(
            self.frame.groupby("source_code").size().to_dict(),
            {
                "HCJ": 35,
                "HKL": 15,
                "HPSF": 152,
                "HRC": 30,
                "HSB": 61,
                "MOH_TREATMENT": 779,
                "MOH_WARD": 18,
            },
        )

    def test_snapshot_parsers_reproduce_committed_counts(self) -> None:
        timestamp = "2026-08-25T00:00:00+00:00"
        hcj_html = (RAW / "hcj_hospital_charges_raw.html").read_text(encoding="utf-8")
        hcj_records, _ = HCJHospitalChargesScraper.parse_html(hcj_html, scraped_at=timestamp)
        self.assertGreaterEqual(len(hcj_records), 35)

        hsb_snapshot = (RAW / "hsb_hospital_charges_raw.html").read_text(encoding="utf-8")
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(hsb_snapshot, "html.parser")
        pages = {
            section.get("data-source-page", ""): "".join(str(child) for child in section.contents)
            for section in soup.select("section[data-source-page]")
        }
        hsb_records, _ = HSBHospitalChargesScraper.parse_pages(pages, scraped_at=timestamp)
        self.assertEqual(len(hsb_records), 61)

        hpsf_html = (RAW / "hpsf_hospital_charges_snapshot.html").read_text(encoding="utf-8")
        hpsf_records, _ = HPSFHospitalChargesScraper.parse_html(hpsf_html, scraped_at=timestamp)
        self.assertEqual(len(hpsf_records), 152)

        ward_html = (RAW / "moh_caj_wad_raw.html").read_text(encoding="utf-8")
        ward_records, _ = MOHWardChargesScraper.parse_html(ward_html, scraped_at=timestamp)
        self.assertEqual(len(ward_records), 18)

        treatment_html = (RAW / "moh_caj_rawatan_raw.html").read_text(encoding="utf-8")
        treatment_records, _ = MOHTreatmentChargesScraper.parse_html(treatment_html, scraped_at=timestamp)
        self.assertGreater(len(treatment_records), 779)


class MultiSourcePublicPricingApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = PublicPricingService(COMBINED)
        self.service.reload()

    def predict(self, category: str, citizenship: str = "Malaysian", hospital: str = "all") -> dict:
        with patch.object(main, "public_pricing_service", self.service), patch.object(main, "enforce_pricing_prediction_access"):
            return main.predict_public(main.PublicPredictionInput(category=category, citizenship=citizenship, hospital=hospital), None)

    def test_hospital_list_is_dataset_driven_and_excludes_moh(self) -> None:
        with patch.object(main, "public_pricing_service", self.service):
            result = main.get_public_hospitals()
        self.assertEqual([item["hospital_code"] for item in result["hospitals"]], ["HKL", "HRC", "HCJ", "HSB", "HPSF"])
        self.assertNotIn("MOH_WARD", {item["hospital_code"] for item in result["hospitals"]})

    def test_all_hospitals_and_moh_are_grouped_separately(self) -> None:
        result = self.predict("Ward Charges")
        self.assertEqual({item["hospital_code"] for item in result["hospitals"]}, {"HKL", "HRC", "HCJ", "HSB", "HPSF"})
        self.assertEqual({item["source_code"] for item in result["national_references"]}, {"MOH_WARD"})
        self.assertEqual(result["hospital_sources_checked"], 5)

    def test_specific_hospital_filter_keeps_matching_moh_reference(self) -> None:
        result = self.predict("Ward Charges", hospital="HCJ")
        self.assertEqual([item["hospital_code"] for item in result["hospitals"]], ["HCJ"])
        self.assertGreater(len(result["hospitals"][0]["records"]), 0)
        self.assertEqual([item["source_code"] for item in result["national_references"]], ["MOH_WARD"])

    def test_selected_hospital_no_data_returns_explicit_empty_card(self) -> None:
        result = self.predict("Laboratory", "Non-Malaysian", hospital="HSB")
        self.assertEqual(len(result["hospitals"]), 1)
        self.assertEqual(result["hospitals"][0]["hospital_code"], "HSB")
        self.assertEqual(result["hospitals"][0]["records"], [])

    def test_non_malaysian_filter_does_not_substitute_citizen_moh_schedule(self) -> None:
        result = self.predict("Ward Charges", "Non-Malaysian")
        self.assertEqual(result["national_references"], [])
        self.assertTrue(all(record["patient_class"] in {"all", "foreigner", "unhcr", "resident"} for hospital in result["hospitals"] for record in hospital["records"]))

    def test_records_retain_official_provenance(self) -> None:
        result = self.predict("Physiotherapy")
        records = [record for group in [*result["hospitals"], *result["national_references"]] for record in group["records"]]
        self.assertTrue(records)
        self.assertTrue(all(record["source_url"].startswith("https://") for record in records))
        self.assertTrue(all(record["original_price_text"] for record in records))


if __name__ == "__main__":
    unittest.main()
