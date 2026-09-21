"""Focused tests for the NSS 80 retraining v3 mapping and ML contract."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from model_training.nss80.retrain_v3 import (
    BINARY_FEATURES,
    CORE_FEATURES,
    KNOWN_LEAKAGE,
    TARGET,
    make_preprocessor,
)
from model_training.nss80.schema_v3 import (
    GENDER,
    HOUSEHOLD_KEY,
    decode_coverage_binary,
    decode_series,
    decode_service_received_binary,
)


class NSS80OfficialMappingTests(unittest.TestCase):
    def test_official_gender_codes_are_not_guessed(self) -> None:
        self.assertEqual(GENDER, {"1": "Male", "2": "Female", "3": "Transgender"})
        decoded = decode_series("b3c4", pd.Series(["1", "2", "3"]))
        self.assertEqual(decoded.tolist(), ["Male", "Female", "Transgender"])

    def test_service_receipt_is_collapsed_only_after_official_decode(self) -> None:
        encoded = decode_service_received_binary(
            pd.Series(["1", "2", "3", "4"]), "b6i13"
        )
        self.assertEqual(encoded.tolist(), [0, 1, 1, 1])

    def test_coverage_is_binary_only_for_modelling(self) -> None:
        encoded = decode_coverage_binary(
            pd.Series(["1", "2", "3", "4", "5", "6", "7", "10", "19"])
        )
        self.assertEqual(encoded.tolist(), [1, 1, 1, 1, 1, 1, 1, 1, 0])

    def test_household_type_decode_is_conditional_on_sector(self) -> None:
        source = pd.Series(["1", "2", "3", "9"])
        rural = decode_series("b5i4", source, sector=pd.Series(["1"] * 4))
        urban = decode_series("b5i4", source, sector=pd.Series(["2"] * 4))
        self.assertEqual(
            rural.tolist(),
            [
                "Self-employed in agriculture",
                "Self-employed in non-agriculture",
                "Regular wage or salary earning in agriculture",
                "Other rural household type",
            ],
        )
        self.assertEqual(
            urban.tolist(),
            [
                "Self-employed",
                "Regular wage or salary earning",
                "Casual labour",
                "Other urban household type",
            ],
        )

    def test_unreliable_suno_is_not_a_cross_level_join_key(self) -> None:
        self.assertNotIn("suno", HOUSEHOLD_KEY)


class NSS80MLContractTests(unittest.TestCase):
    def test_core_feature_set_has_18_features_and_no_known_leakage(self) -> None:
        self.assertEqual(len(CORE_FEATURES), 18)
        self.assertFalse(set(CORE_FEATURES) & KNOWN_LEAKAGE)
        self.assertNotIn(TARGET, CORE_FEATURES)

    def test_gender_is_a_binary_ml_column_and_nominal_fields_are_one_hot(self) -> None:
        frame = pd.DataFrame(
            {
                "household_size": [3, 4, 5, 2],
                "household_usual_consumer_expenditure_rs": [12000, 16000, 9000, 21000],
                "age_years": [24, 51, 36, 68],
                "number_of_hospitalisations": [1, 2, 1, 3],
                "length_of_stay_days": [2, 5, 3, 9],
                "gender": [0, 1, 0, 1],
                "chronic_ailment": [0, 1, 0, 1],
                "health_financing_or_insurance_coverage": [1, 0, 1, 0],
                "surgery": [0, 1, 0, 1],
                "medicine": [1, 1, 1, 0],
                "xray_ecg_eeg_scan": [0, 1, 1, 0],
                "other_diagnostic_tests": [0, 0, 1, 1],
                "sector": ["Rural", "Urban", "Rural", "Urban"],
                "state": ["Kerala", "Kerala", "Delhi", "Delhi"],
                "ailment_nature": ["Malaria", "Diabetes", "Malaria", "Diabetes"],
                "hospitalisation_treatment_nature": ["Allopathy"] * 4,
                "medical_institution_type": [
                    "Government or public hospital",
                    "Private hospital",
                    "Government or public hospital",
                    "Private hospital",
                ],
                "ward_type": [
                    "Free ward",
                    "Paying special ward",
                    "Paying general ward",
                    "Paying special ward",
                ],
            }
        )
        preprocessor = make_preprocessor(CORE_FEATURES)
        transformed = np.asarray(preprocessor.fit_transform(frame[CORE_FEATURES]))
        names = preprocessor.get_feature_names_out().tolist()
        gender_index = names.index("binary__gender")
        self.assertTrue(np.isin(transformed[:, gender_index], [0, 1]).all())
        nominal_indices = [
            index for index, name in enumerate(names) if name.startswith("nominal__")
        ]
        self.assertTrue(np.isin(transformed[:, nominal_indices], [0, 1]).all())
        self.assertIn("gender", BINARY_FEATURES)


if __name__ == "__main__":
    unittest.main()
