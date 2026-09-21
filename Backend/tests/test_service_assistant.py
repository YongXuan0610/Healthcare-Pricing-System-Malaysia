from __future__ import annotations

import csv
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException
from pydantic import ValidationError

from app.schemas.service_assistant import ServiceAssistantRequest
from app.services.service_assistant import (
    SERVICE_MAPPINGS,
    HealthcareServiceAssistant,
)
from scripts import main


class HealthcareServiceAssistantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assistant = HealthcareServiceAssistant()

    def test_supported_service_examples_return_expected_mapping(self) -> None:
        examples = {
            "Physiotherapy": (
                "I hurt my knee and have difficulty walking",
                "I need rehabilitation after an injury",
                "I have back pain and movement problems",
            ),
            "Dietetics / Nutrition": (
                "I need advice about my diet",
                "I want nutrition advice",
            ),
            "Nephrology": (
                "I have a kidney problem",
                "I need dialysis related care",
            ),
            "Radiotherapy / Oncology": (
                "I need oncology pricing information",
                "I am looking for a cancer related service",
            ),
            "Radiology / Imaging": (
                "I need an X-ray",
                "I need an imaging scan",
            ),
            "Laboratory": (
                "I need a blood test",
                "I am looking for a laboratory test",
            ),
            "Ward / Admission": (
                "How much is a hospital room?",
                "I want to know the ward charges",
            ),
        }

        for expected_service, messages in examples.items():
            for message in messages:
                with self.subTest(expected_service=expected_service, message=message):
                    result = self.assistant.recommend(message)
                    self.assertEqual(result.status, "matched")
                    self.assertEqual(result.service, expected_service)
                    self.assertIsNotNone(result.mapped_service)
                    self.assertIsNotNone(result.category)
                    self.assertIsNotNone(result.confidence)
                    self.assertGreaterEqual(result.confidence or 0, 0.7)
                    self.assertEqual(result.suggestions, [])

    def test_case_punctuation_and_hyphen_differences_are_normalized(self) -> None:
        result = self.assistant.recommend("  I NEED an X-RAY!!!  ")

        self.assertEqual(result.status, "matched")
        self.assertEqual(result.service, "Radiology / Imaging")
        self.assertEqual(result.mapped_service, "Radiology / Imaging")

    def test_two_strong_matches_return_at_most_two_choices(self) -> None:
        result = self.assistant.recommend(
            "I have knee pain and need an X-ray after the injury."
        )

        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.service)
        self.assertEqual(len(result.suggestions), 2)
        self.assertEqual(
            {suggestion.service for suggestion in result.suggestions},
            {"Physiotherapy", "Radiology / Imaging"},
        )

    def test_vague_unrelated_input_returns_clarification_without_guessing(self) -> None:
        result = self.assistant.recommend("I would like some general information")

        self.assertEqual(result.status, "unmatched")
        self.assertIsNone(result.service)
        self.assertIsNone(result.confidence)
        self.assertEqual(result.suggestions, [])
        self.assertIn("not confident", result.message.lower())

    def test_repeated_input_is_deterministic(self) -> None:
        message = "I need advice about nutrition and meal planning"

        first = self.assistant.recommend(message).model_dump()
        second = self.assistant.recommend(message).model_dump()

        self.assertEqual(first, second)

    def test_urgent_descriptions_suppress_normal_recommendations(self) -> None:
        messages = (
            "I have severe chest pain and want to know the price",
            "The person is unconscious and has a knee injury",
            "I am having a seizure now",
            "There is uncontrolled bleeding after an accident",
            "I have severe difficulty breathing",
        )

        for message in messages:
            with self.subTest(message=message):
                result = self.assistant.recommend(message)
                self.assertEqual(result.status, "urgent")
                self.assertIsNone(result.service)
                self.assertIsNone(result.mapped_service)
                self.assertEqual(result.suggestions, [])
                self.assertIn("urgent medical attention", result.message.lower())

    def test_all_mappings_refer_to_current_public_pricing_categories(self) -> None:
        csv_path = (
            Path(__file__).resolve().parents[1]
            / "app"
            / "data"
            / "processed"
            / "public_hospital_prices.csv"
        )
        with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))

        available_categories = {row["category"] for row in rows}

        for mapping_key, definition in SERVICE_MAPPINGS.items():
            with self.subTest(mapping_key=mapping_key):
                self.assertIn(definition.category, available_categories)
                self.assertIsNone(definition.service_name)


class ServiceAssistantSchemaAndApiTests(unittest.TestCase):
    def test_request_trims_message(self) -> None:
        request = ServiceAssistantRequest(message="  I need a blood test  ")
        self.assertEqual(request.message, "I need a blood test")

    def test_request_rejects_empty_whitespace_and_excessively_long_messages(self) -> None:
        invalid_messages = ("", "   ", "x" * 501)

        for message in invalid_messages:
            with self.subTest(length=len(message)):
                with self.assertRaises(ValidationError):
                    ServiceAssistantRequest(message=message)

    def test_request_rejects_unexpected_fields(self) -> None:
        with self.assertRaises(ValidationError):
            ServiceAssistantRequest(message="I need a dental check-up", extra="no")

    def test_endpoint_is_registered_and_returns_typed_result(self) -> None:
        route_methods = {
            (route.path, method)
            for route in main.app.routes
            for method in getattr(route, "methods", set())
        }
        self.assertIn(("/assistant/recommend-service", "POST"), route_methods)

        result = main.recommend_healthcare_service(
            ServiceAssistantRequest(message="I am looking for physiotherapy pricing")
        )
        self.assertEqual(result.status, "matched")
        self.assertEqual(result.service, "Physiotherapy")

    def test_endpoint_hides_unexpected_backend_errors(self) -> None:
        with patch.object(
            main.service_assistant,
            "recommend",
            side_effect=RuntimeError("internal implementation detail"),
        ):
            with self.assertRaises(HTTPException) as raised:
                main.recommend_healthcare_service(
                    ServiceAssistantRequest(message="I need physiotherapy pricing")
                )

        self.assertEqual(raised.exception.status_code, 500)
        self.assertNotIn(
            "internal implementation detail",
            str(raised.exception.detail),
        )
        self.assertIn("select a service manually", str(raised.exception.detail))

    def test_public_pricing_forwards_optional_assistant_service_filter(self) -> None:
        pricing_reference = {
            "estimate_available": True,
            "pricing_type": "exact",
            "published_cost": 10.0,
            "lower_estimate": None,
            "typical_estimate": 10.0,
            "upper_estimate": None,
            "records_used": 1,
            "range_method": "single_published_charge",
        }
        matched_charges = pd.DataFrame(
            [
                {
                    "category": "common public hospital charges",
                    "service_name": "Fisioterapi",
                    "patient_class": "all",
                    "charge_type": "per visit",
                    "price_rm": 10.0,
                    "source_url": "https://example.test/source",
                }
            ]
        )

        with (
            patch.object(main, "enforce_pricing_prediction_access"),
            patch.object(
                main.public_pricing_service,
                "get_pricing_reference",
                return_value=(pricing_reference, matched_charges),
            ) as get_reference,
        ):
            result = main.predict_public(
                main.PublicPredictionInput(
                    category="common public hospital charges",
                    state="Selangor",
                    citizenship="Malaysian",
                    service_name="Fisioterapi",
                ),
                authorization=None,
            )

        get_reference.assert_called_once_with(
            category="common public hospital charges",
            service_name="Fisioterapi",
            patient_class="citizen",
            hospital_code="all",
        )
        self.assertEqual(
            result["input_received"]["service_name"],
            "Fisioterapi",
        )


if __name__ == "__main__":
    unittest.main()
