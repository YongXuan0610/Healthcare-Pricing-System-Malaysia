"""Focused tests for separated NSS, US benchmark, and Malaysia references."""

import unittest
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException
from pydantic import ValidationError

from app.schemas.prediction import NSS80CostPredictionInput, USBenchmarkPredictionInput
from app.services.public_pricing_service import PublicPricingService
from scripts import main


class PredictionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings_patcher = patch.object(
            main,
            "get_system_settings_record",
            return_value={
                "allow_guest_predictions": True,
                "maintenance_mode": False,
            },
        )
        self.settings_patcher.start()
        self.addCleanup(self.settings_patcher.stop)
        self.public_pricing_patcher = patch.object(
            main,
            "public_pricing_service",
            PublicPricingService(),
        )
        self.public_pricing_patcher.start()
        self.addCleanup(self.public_pricing_patcher.stop)
        self.private_package_patcher = patch.object(
            main.private_pricing_service,
            "get_package",
            return_value={"price": 551.0},
        )
        self.private_ward_patcher = patch.object(
            main.private_pricing_service,
            "get_ward",
            return_value={"dailyRate": 1029.0},
        )
        self.private_list_patcher = patch.object(
            main.private_pricing_service,
            "list_packages",
            return_value=[{"name": "Basic Screening", "price": 551.0}],
        )
        self.private_package_patcher.start()
        self.private_ward_patcher.start()
        self.private_list_patcher.start()
        self.addCleanup(self.private_package_patcher.stop)
        self.addCleanup(self.private_ward_patcher.stop)
        self.addCleanup(self.private_list_patcher.stop)

    @staticmethod
    def nss_input() -> NSS80CostPredictionInput:
        return NSS80CostPredictionInput(
            age_years=35,
            gender="Female",
            chronic_ailment="No",
            pregnant="No",
            communicable_disease="Not suffered",
            other_ailment_last_15_days="No",
            number_of_hospitalisations=1,
            length_of_stay_days=3,
            ailment_nature="All other fevers",
            hospitalisation_treatment_nature="Allopathy",
            medical_institution_type="Private hospital",
            ward_type="Paying general ward",
            place_of_hospitalisation="Same district, urban area",
            surgery="Not received",
            medicine="Received",
        )

    def test_post_nss80_route_handler_returns_v4_inr_context(self) -> None:
        payload = main.predict_with_model(
            self.nss_input(), authorization=None
        )
        self.assertEqual(payload["type"], "machine_learning_primary")
        self.assertEqual(payload["research_role"], "primary")
        self.assertEqual(payload["model_version"], "nss80_practical_15")
        self.assertEqual(payload["model_type"], "HistGradientBoostingRegressor")
        self.assertEqual(payload["target_transform"], "identity")
        self.assertEqual(payload["feature_count"], 15)
        self.assertEqual(payload["encoded_feature_count"], 90)
        self.assertAlmostEqual(payload["r2_test"], 0.39769617865686857)
        self.assertFalse(payload["payment_status_detail_used"])
        self.assertTrue(payload["preprocessing_included_in_artifact"])
        self.assertGreaterEqual(payload["predicted_medical_expenditure"], 0)
        self.assertEqual(payload["currency"], "INR")
        self.assertTrue(payload["prediction_interval_available"])
        self.assertLessEqual(payload["predicted_medical_expenditure_range"]["lower"], payload["predicted_medical_expenditure"])
        self.assertIn("person-calibrated", payload["range_interpretation"])
        self.assertIn("not a Malaysian price", payload["limitation"])

    def test_get_nss80_options_route_handler_exposes_v4_contract(self) -> None:
        payload = main.get_nss80_prediction_options()
        self.assertEqual(payload["model"]["model_version"], "nss80_practical_15")
        self.assertEqual(payload["model"]["feature_count"], 15)
        self.assertEqual(payload["model"]["encoded_feature_count"], 90)
        self.assertAlmostEqual(payload["model"]["r2_test"], 0.39769617865686857)
        self.assertEqual(len(payload["field_order"]), 15)
        self.assertEqual(payload["currency"], "INR")
        self.assertTrue(payload["prediction_interval"]["available"])
        self.assertNotIn("treatment_state_code", payload["categories"])
        self.assertNotIn("state", payload["categories"])
        self.assertNotIn("dependent_options", payload)
        self.assertFalse(
            payload["service_receipt_policy"]["payment_status_detail_used"]
        )
        route_methods = {
            route.path: route.methods
            for route in main.app.routes
            if route.path in {"/predict/nss80", "/predict/nss80/options", "/predict/model"}
        }
        self.assertIn("POST", route_methods["/predict/nss80"])
        self.assertIn("GET", route_methods["/predict/nss80/options"])
        self.assertIn("POST", route_methods["/predict/model"])

    def test_nss_payment_detail_is_rejected_by_practical_contract(self) -> None:
        for field in ["surgery", "medicine"]:
            for value in ["Received free", "Received partly free", "Received on payment"]:
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    NSS80CostPredictionInput(**{**self.nss_input().model_dump(), field: value})

    def test_secondary_us_benchmark_remains_separate(self) -> None:
        payload = main.predict_us_benchmark(
            USBenchmarkPredictionInput(
                age=35,
                sex="male",
                bmi=27.5,
                children=1,
                smoker="no",
                region="southeast",
            ),
            authorization=None,
        )
        self.assertEqual(payload["type"], "machine_learning_benchmark")
        self.assertEqual(payload["research_role"], "secondary_benchmark")
        self.assertEqual(payload["currency"], "USD")
        self.assertNotIn("state", payload["input_received"])
        self.assertIn("Secondary benchmark only", payload["limitation"])

    def test_primary_ml_unavailable_returns_controlled_503(self) -> None:
        request = self.nss_input()
        with patch.object(main, "get_nss80_prediction_service", return_value=None):
            with self.assertRaises(HTTPException) as raised:
                main.predict_with_model(request, authorization=None)
        self.assertEqual(raised.exception.status_code, 503)
        self.assertNotIn("Traceback", str(raised.exception.detail))

    def test_us_benchmark_rejects_unknown_category(self) -> None:
        with self.assertRaises(ValidationError):
            USBenchmarkPredictionInput.model_validate(
                {
                    "age": 35,
                    "sex": "other",
                    "bmi": 27.5,
                    "children": 1,
                    "smoker": "no",
                    "region": "southeast",
                }
            )

    def test_nss_primary_rejects_malaysian_state(self) -> None:
        with self.assertRaises(ValidationError):
            NSS80CostPredictionInput(
                **{
                    **self.nss_input().model_dump(),
                    "state": "Selangor",
                }
            )

    def test_nss_primary_rejects_out_of_range_value(self) -> None:
        with self.assertRaises(ValidationError):
            NSS80CostPredictionInput(
                **{
                    **self.nss_input().model_dump(),
                    "length_of_stay_days": 0,
                }
            )

    def test_nss_primary_rejects_removed_district(self) -> None:
        with self.assertRaises(ValidationError):
            NSS80CostPredictionInput(**{**self.nss_input().model_dump(), "district": "Delhi - district code 99"})

    def test_nss_guest_prediction_setting_is_enforced(self) -> None:
        with patch.object(
            main,
            "get_system_settings_record",
            return_value={"allow_guest_predictions": False, "maintenance_mode": False},
        ):
            with self.assertRaises(HTTPException) as raised:
                main.predict_with_model(self.nss_input(), authorization=None)
        self.assertEqual(raised.exception.status_code, 401)

    def test_nss_maintenance_mode_is_enforced(self) -> None:
        with patch.object(
            main,
            "get_system_settings_record",
            return_value={"allow_guest_predictions": True, "maintenance_mode": True},
        ):
            with self.assertRaises(HTTPException) as raised:
                main.predict_with_model(self.nss_input(), authorization=None)
        self.assertEqual(raised.exception.status_code, 503)

    def test_liam_published_range_is_reference_not_model_output(self) -> None:
        result = main.get_liam_pricing_reference(
            procedure_code="C1",
            care_setting="Inpatient",
            segmentation_type="Overall",
            segment="All",
        )
        self.assertEqual(result["component"], "malaysia_liam_pricing_reference")
        self.assertEqual(result["research_role"], "pricing_reference_only")
        self.assertTrue(result["estimate_available"])
        self.assertEqual(result["currency"], "MYR")
        self.assertEqual(result["lower_reference"], 8800.0)
        self.assertEqual(result["typical_bill_amount"], 11700.0)
        self.assertEqual(result["upper_reference"], 18300.0)

    def test_liam_insufficient_data_is_not_fabricated(self) -> None:
        result = main.get_liam_pricing_reference(
            procedure_code="C1",
            care_setting="Inpatient",
            segmentation_type="Facility State",
            segment="Perlis",
        )
        self.assertFalse(result["estimate_available"])
        self.assertEqual(result["pricing_type"], "insufficient_credible_data")
        self.assertNotIn("typical_bill_amount", result)

    def test_us_benchmark_unavailable_returns_controlled_503(self) -> None:
        request = USBenchmarkPredictionInput(
            age=35,
            sex="male",
            bmi=27.5,
            children=1,
            smoker="no",
            region="southeast",
        )
        with patch.object(main, "get_us_benchmark_service", return_value=None):
            with self.assertRaises(HTTPException) as raised:
                main.predict_us_benchmark(request, authorization=None)
        self.assertEqual(raised.exception.status_code, 503)
        self.assertNotIn("Traceback", str(raised.exception.detail))

    def test_public_and_private_functions_exist_when_ml_is_unavailable(self) -> None:
        with patch.object(main, "get_ml_prediction_service", return_value=None):
            public_result = main.predict_public(
                main.PublicPredictionInput(
                    category="ward charges",
                    state="Selangor",
                    citizenship="Malaysian",
                ),
                authorization=None,
            )
            private_result = main.predict_private(
                main.PrivatePredictionInput(
                    hospital="Gleneagles",
                    package_name="Basic Screening",
                    ward_type="No Stay",
                    nights=0,
                ),
                authorization=None,
            )
        self.assertTrue(public_result["estimate_available"])
        self.assertNotIn("error", private_result)

    def test_citizenship_mapping_is_explicit(self) -> None:
        self.assertEqual(main.map_citizenship_to_patient_class("Malaysian"), "citizen")
        self.assertEqual(
            main.map_citizenship_to_patient_class("Non-Malaysian"), "foreigner"
        )
        with self.assertRaises(HTTPException):
            main.map_citizenship_to_patient_class("unknown")

    def test_public_official_match_has_no_state_adjustment(self) -> None:
        result = main.predict_public(
            main.PublicPredictionInput(
                category="ward charges",
                state="Selangor",
                citizenship="Malaysian",
            ),
            authorization=None,
        )
        self.assertTrue(result["estimate_available"])
        self.assertIsNotNone(result["predicted_cost"])
        self.assertEqual(result["pricing_type"], "range")
        self.assertEqual(result["currency"], "MYR")
        self.assertEqual(result["lower_estimate"], 20.0)
        self.assertEqual(result["typical_estimate"], 60.0)
        self.assertEqual(result["upper_estimate"], 105.0)
        self.assertEqual(result["records_used"], 15)
        self.assertLessEqual(result["lower_estimate"], result["typical_estimate"])
        self.assertLessEqual(result["typical_estimate"], result["upper_estimate"])
        self.assertEqual(result["pricing_source"], "official_public_dataset")
        self.assertFalse(result["state_adjusted"])
        self.assertTrue(
            all(
                charge["patient_class"] in {"all", "citizen", "foreigner"}
                for charge in result["matched_public_charges"]
            )
        )

    def test_public_missing_match_returns_no_fabricated_fallback(self) -> None:
        result = main.predict_public(
            main.PublicPredictionInput(
                category="category not present in published data",
                state="Selangor",
                citizenship="Non-Malaysian",
            ),
            authorization=None,
        )
        self.assertFalse(result["estimate_available"])
        self.assertEqual(result["pricing_type"], "unavailable")
        self.assertIsNone(result["predicted_cost"])
        self.assertEqual(result["records_used"], 0)
        self.assertEqual(result["pricing_source"], "unavailable")
        self.assertEqual(result["patient_class"], "foreigner")

    def test_public_single_record_is_exact_published_charge(self) -> None:
        matches = pd.DataFrame(
            [
                {
                    "category": "ward charges",
                    "service_name": "Example published ward",
                    "patient_class": "all",
                    "charge_type": "published rate",
                    "price_rm": 30.0,
                    "source_url": "https://example.invalid/official-source",
                }
            ]
        )
        reference = PublicPricingService.summarize_matches(matches)
        with patch.object(
            main.public_pricing_service,
            "get_pricing_reference",
            return_value=(reference, matches),
        ):
            result = main.predict_public(
                main.PublicPredictionInput(
                    category="ward charges",
                    state="Selangor",
                    citizenship="Malaysian",
                ),
                authorization=None,
            )
        self.assertTrue(result["estimate_available"])
        self.assertEqual(result["pricing_type"], "exact")
        self.assertEqual(result["published_cost"], 30.0)
        self.assertEqual(result["typical_estimate"], 30.0)
        self.assertEqual(result["records_used"], 1)

    def test_public_two_to_four_records_use_published_min_max_range(self) -> None:
        matches = pd.DataFrame({"price_rm": [10.0, 15.0, 20.0, 30.0]})
        reference = PublicPricingService.summarize_matches(matches)
        self.assertEqual(reference["pricing_type"], "range")
        self.assertEqual(reference["range_method"], "min_max")
        self.assertEqual(reference["lower_estimate"], 10.0)
        self.assertEqual(reference["typical_estimate"], 17.5)
        self.assertEqual(reference["upper_estimate"], 30.0)

    def test_private_total_uses_only_published_package_and_ward_prices(self) -> None:
        result = main.predict_private(
            main.PrivatePredictionInput(
                hospital="Gleneagles",
                package_name="Basic Screening",
                ward_type="Deluxe Suite",
                nights=2,
            ),
            authorization=None,
        )
        self.assertNotIn("error", result)
        self.assertEqual(result["breakdown"]["package_price"], 551.0)
        self.assertEqual(result["breakdown"]["ward_price_per_night"], 1029.0)
        self.assertEqual(result["breakdown"]["ward_cost"], 2058.0)
        self.assertEqual(result["predicted_cost"], 2609.0)
        self.assertIsNone(result["breakdown"]["surgeon_fee"])
        self.assertIsNone(result["breakdown"]["misc_fee"])
        self.assertFalse(result["additional_charges_included"])

    def test_private_missing_package_returns_controlled_404(self) -> None:
        request = main.PrivatePredictionInput(
            hospital="Example Hospital",
            package_name="Missing Package",
            ward_type="No Stay",
            nights=0,
        )
        with patch.object(
            main.private_pricing_service,
            "get_package",
            return_value=None,
        ):
            with self.assertRaises(HTTPException) as raised:
                main.predict_private(request, authorization=None)

        self.assertEqual(raised.exception.status_code, 404)
        self.assertIn("Missing Package", str(raised.exception.detail))
        self.assertNotIn("predicted_cost", str(raised.exception.detail))

    def test_private_missing_ward_returns_controlled_404(self) -> None:
        request = main.PrivatePredictionInput(
            hospital="Example Hospital",
            package_name="Example Package",
            ward_type="Missing Ward",
            nights=1,
        )
        with (
            patch.object(
                main.private_pricing_service,
                "get_package",
                return_value={"price": 100.0},
            ),
            patch.object(
                main.private_pricing_service,
                "get_ward",
                return_value=None,
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                main.predict_private(request, authorization=None)

        self.assertEqual(raised.exception.status_code, 404)
        self.assertIn("Missing Ward", str(raised.exception.detail))
        self.assertNotIn("predicted_cost", str(raised.exception.detail))


class CorsConfigurationTests(unittest.TestCase):
    def test_frontend_origins_are_parsed_and_deduplicated(self) -> None:
        origins = main.parse_frontend_origins(
            "http://localhost:5173, http://127.0.0.1:5173/, "
            "http://localhost:5173"
        )
        self.assertEqual(
            origins,
            ["http://localhost:5173", "http://127.0.0.1:5173"],
        )

    def test_frontend_origins_reject_wildcards(self) -> None:
        with self.assertRaises(ValueError):
            main.parse_frontend_origins("*")

if __name__ == "__main__":
    unittest.main()
