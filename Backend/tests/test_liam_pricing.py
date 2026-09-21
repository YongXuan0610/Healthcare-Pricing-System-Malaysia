"""Dataset and lookup tests for the published Malaysian LIAM reference."""

import unittest

from app.services.liam_pricing_service import LIAMPricingReferenceService


EXPECTED_FACILITY_STATES = {
    "Johor",
    "Kedah",
    "Kelantan",
    "Kuala Lumpur",
    "Melaka",
    "Negeri Sembilan",
    "Pahang",
    "Perak",
    "Perlis",
    "Pulau Pinang",
    "Sabah",
    "Sarawak",
    "Selangor",
    "Terengganu",
}


class LIAMPricingReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = LIAMPricingReferenceService()

    def test_dataset_counts_and_facility_states(self) -> None:
        frame = self.service.frame

        self.assertEqual(len(frame), 760)
        self.assertEqual(len(frame[frame["segmentation_type"].eq("Overall")]), 40)
        facility_rows = frame[frame["segmentation_type"].eq("Facility State")]
        self.assertEqual(len(facility_rows), 560)
        self.assertEqual(set(facility_rows["segment"]), EXPECTED_FACILITY_STATES)
        self.assertEqual(
            len(
                facility_rows[
                    facility_rows["data_status"].eq("Insufficient credible data")
                ]
            ),
            191,
        )

    def test_dataset_integrity_rules(self) -> None:
        frame = self.service.frame
        numeric_columns = [
            "typical_bill_amount_rm",
            "typical_bill_p25_rm",
            "typical_bill_p75_rm",
        ]
        key_columns = [
            "procedure_code",
            "care_setting",
            "segmentation_type",
            "segment",
        ]

        self.assertFalse(frame.duplicated(key_columns).any())

        published = frame[frame["data_status"].eq("Published")]
        self.assertTrue(
            published["typical_bill_p25_rm"]
            .le(published["typical_bill_amount_rm"])
            .all()
        )
        self.assertTrue(
            published["typical_bill_amount_rm"]
            .le(published["typical_bill_p75_rm"])
            .all()
        )

        insufficient = frame[
            frame["data_status"].eq("Insufficient credible data")
        ]
        self.assertFalse(insufficient[numeric_columns].notna().any(axis=None))

        facility_states = frame[frame["segmentation_type"].eq("Facility State")]
        self.assertTrue(
            facility_states["segment"]
            .astype(str)
            .eq(facility_states["segment"].astype(str).str.strip())
            .all()
        )

    def test_procedure_metadata_includes_available_and_insufficient_states(self) -> None:
        wrist_fracture = next(
            procedure
            for procedure in self.service.list_procedures()
            if procedure["procedure_code"] == "M6"
            and procedure["care_setting"] == "Inpatient"
        )
        states = {
            item["state"]: item["data_status"]
            for item in wrist_fracture["facility_states"]
        }

        self.assertEqual(set(states), EXPECTED_FACILITY_STATES)
        self.assertEqual(states["Johor"], "Published")
        self.assertEqual(states["Perlis"], "Insufficient credible data")

    def test_wrist_fracture_overall_reference_remains_available(self) -> None:
        result = self.service.get_reference(
            procedure_code="M6",
            care_setting="Inpatient",
            segmentation_type="Overall",
            segment="All",
        )

        self.assertTrue(result["estimate_available"])
        self.assertEqual(result["lower_reference"], 3900.0)
        self.assertEqual(result["typical_bill_amount"], 5800.0)
        self.assertEqual(result["upper_reference"], 8900.0)
        self.assertEqual(result["number_of_discharges_band"], "1,000 - 4,999")
        self.assertEqual(result["source_pdf_page"], 103)

    def test_wrist_fracture_johor_and_selangor_references(self) -> None:
        johor = self.service.get_reference(
            procedure_code="M6",
            care_setting="Inpatient",
            segmentation_type="Facility State",
            segment="Johor",
        )
        selangor = self.service.get_reference(
            procedure_code="M6",
            care_setting="Inpatient",
            segmentation_type="Facility State",
            segment="Selangor",
        )

        self.assertEqual(
            (
                johor["lower_reference"],
                johor["typical_bill_amount"],
                johor["upper_reference"],
            ),
            (4200.0, 6700.0, 9400.0),
        )
        self.assertEqual(johor["number_of_discharges_band"], "100 - 499")
        self.assertEqual(johor["source_pdf_page"], 104)

        self.assertEqual(
            (
                selangor["lower_reference"],
                selangor["typical_bill_amount"],
                selangor["upper_reference"],
            ),
            (4000.0, 5900.0, 9300.0),
        )
        self.assertEqual(selangor["number_of_discharges_band"], "500 - 999")
        self.assertEqual(selangor["source_pdf_page"], 104)

    def test_insufficient_state_has_no_numeric_values_or_national_fallback(self) -> None:
        result = self.service.get_reference(
            procedure_code="M6",
            care_setting="Inpatient",
            segmentation_type="Facility State",
            segment="Perlis",
        )

        self.assertFalse(result["estimate_available"])
        self.assertEqual(result["pricing_type"], "insufficient_credible_data")
        self.assertEqual(result["data_status"], "Insufficient credible data")
        self.assertEqual(result["segment"], "Perlis")
        self.assertEqual(result["source_pdf_page"], 104)
        self.assertNotIn("lower_reference", result)
        self.assertNotIn("typical_bill_amount", result)
        self.assertNotIn("upper_reference", result)

    def test_procedure_filter_does_not_leak_another_procedures_state_row(self) -> None:
        result = self.service.get_reference(
            procedure_code="C1",
            care_setting="Inpatient",
            segmentation_type="Facility State",
            segment="Johor",
        )

        self.assertEqual(result["procedure_code"], "C1")
        self.assertNotEqual(result["typical_bill_amount"], 6700.0)


if __name__ == "__main__":
    unittest.main()
