from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from app.scrapers.hrc_scraper import (
    HOSPITAL_CODE,
    SOURCE_URL,
    HRCHospitalChargesScraper,
)
from app.services.public_pricing_service import PublicPricingService
from app.services.supabase_pricing_service import SupabasePublicPricingService
from scripts import main


BACKEND_ROOT = Path(__file__).resolve().parents[1]
RAW_HRC_HTML = (
    BACKEND_ROOT
    / "app"
    / "data"
    / "raw"
    / "hrc_ward_and_treatment_charges_raw.html"
)
HRC_CSV = (
    BACKEND_ROOT / "app" / "data" / "processed" / "hrc_public_charges.csv"
)
COMBINED_CSV = (
    BACKEND_ROOT
    / "app"
    / "data"
    / "processed"
    / "public_hospital_prices.csv"
)
HKL_CSV = (
    BACKEND_ROOT / "app" / "data" / "processed" / "hkl_public_charges.csv"
)


class HRCScraperDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = RAW_HRC_HTML.read_text(encoding="utf-8")

    def test_scraper_generates_normalized_deduplicated_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            temp_root = Path(temp_directory)
            scraper = HRCHospitalChargesScraper(
                raw_output_path=temp_root / "raw.html",
                output_path=temp_root / "hrc.csv",
                combined_output_path=temp_root / "combined.csv",
                hkl_csv_path=HKL_CSV,
            )
            frame = scraper.scrape(
                self.html,
                scraped_at="2026-08-25T00:00:00+00:00",
            )

            self.assertEqual(len(frame), 30)
            self.assertTrue((temp_root / "raw.html").exists())
            self.assertTrue((temp_root / "hrc.csv").exists())
            self.assertTrue((temp_root / "combined.csv").exists())
            self.assertEqual(set(frame["hospital_code"]), {HOSPITAL_CODE})
            self.assertTrue(pd.api.types.is_numeric_dtype(frame["price_rm"]))
            self.assertTrue(frame["source_url"].eq(SOURCE_URL).all())

            duplicate_columns = [
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
            self.assertFalse(frame.duplicated(duplicate_columns).any())

    def test_committed_dataset_preserves_free_and_merged_cell_prices(self) -> None:
        frame = pd.read_csv(HRC_CSV)

        free_rows = frame[frame["price_rm"].eq(0)]
        self.assertEqual(len(free_rows), 4)
        self.assertTrue(
            free_rows["original_price_text"].str.contains(
                "Free|Percuma", case=False, regex=True
            ).all()
        )

        foreign_ward_rows = frame[
            frame["category"].eq("Ward Charges")
            & frame["patient_class"].eq("foreigner")
        ]
        self.assertEqual(len(foreign_ward_rows), 3)
        self.assertEqual(set(foreign_ward_rows["price_rm"]), {120.0})

    def test_committed_categories_match_official_tables(self) -> None:
        frame = pd.read_csv(HRC_CSV)
        self.assertEqual(
            set(frame["category"]),
            {
                "Ward Deposit",
                "Ward Charges",
                "Treatment Charges",
                "Specialist Clinic",
                "Physiotherapy",
                "Occupational Therapy",
                "Speech Therapy",
                "Audiology",
                "Traditional and Complementary Medicine",
            },
        )


class TwoHospitalPublicPricingApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = PublicPricingService(COMBINED_CSV)
        self.service.reload()

    def _predict(self, category: str, citizenship: str = "Malaysian") -> dict:
        with (
            patch.object(main, "public_pricing_service", self.service),
            patch.object(main, "enforce_pricing_prediction_access"),
        ):
            return main.predict_public(
                main.PublicPredictionInput(
                    category=category,
                    citizenship=citizenship,
                ),
                authorization=None,
            )

    def test_category_union_contains_hkl_and_hrc_categories(self) -> None:
        with patch.object(main, "public_pricing_service", self.service):
            categories = main.get_public_categories()["categories"]
        self.assertIn("Ward Charges", categories)
        self.assertIn("Physiotherapy", categories)
        self.assertIn("Specialist Clinic", categories)

    def test_common_category_returns_hkl_and_hrc_separately(self) -> None:
        result = self._predict("Ward Charges")
        hospitals = {
            hospital["hospital_code"]: hospital
            for hospital in result["hospitals"]
        }

        self.assertGreater(len(hospitals["HKL"]["records"]), 0)
        self.assertGreater(len(hospitals["HRC"]["records"]), 0)
        self.assertEqual(hospitals["HKL"]["statistics"]["records_used"], 15)
        self.assertEqual(hospitals["HRC"]["statistics"]["records_used"], 3)
        self.assertEqual(result["summary_hospital_code"], "HKL")

    def test_hrc_only_category_returns_only_matching_hospital_for_all_filter(self) -> None:
        result = self._predict("Physiotherapy", "Non-Malaysian")
        hospitals = {
            hospital["hospital_code"]: hospital
            for hospital in result["hospitals"]
        }

        self.assertNotIn("HKL", hospitals)
        self.assertEqual(len(hospitals["HRC"]["records"]), 1)
        self.assertEqual(hospitals["HRC"]["records"][0]["price_rm"], 78.0)
        self.assertEqual(result["summary_hospital_code"], "HRC")

    def test_invalid_category_is_safe_and_returns_no_hospital_cards(self) -> None:
        result = self._predict("not an official category")

        self.assertFalse(result["estimate_available"])
        self.assertEqual(result["records_used"], 0)
        self.assertEqual(result["hospitals"], [])

    def test_runtime_supabase_service_merges_legacy_hkl_with_local_hrc(self) -> None:
        class FakePricingApi:
            def select(self, *_args, **_kwargs):
                return [
                    {
                        "category": "ward charges",
                        "service_name": "Legacy HKL room",
                        "patient_class": "all",
                        "charge_type": "ward charge",
                        "price_rm": 25,
                        "source_url": "https://hkl.moh.gov.my/awam/caj-hospital",
                    }
                ]

        service = SupabasePublicPricingService(
            FakePricingApi(),  # type: ignore[arg-type]
            hrc_csv_path=HRC_CSV,
        )
        frame = service.load_dataframe()

        self.assertEqual(set(frame["hospital_code"]), {"HKL", "HRC"})
        self.assertIn("Physiotherapy", service.list_categories())


if __name__ == "__main__":
    unittest.main()
