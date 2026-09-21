"""Focused scientific-contract tests for the NSS leakage-safe v4 runner."""

from __future__ import annotations

import unittest

import pandas as pd

from model_training.nss80.codebook import AILMENT
from model_training.nss80.safe_improvement_v4 import (
    AILMENT_GROUP_BY_LABEL,
    BASE_FEATURES,
    BASIC_ENGINEERED_FEATURES,
    BINARY_FEATURES,
    L5_ALLOWED_RAW_COLUMNS,
    L5_FEATURES,
    L5_PROHIBITED_PAYMENT_RAW_COLUMNS,
    PROHIBITED_FEATURES,
    CandidateSpec,
    SafeRegressor,
    add_engineered_features,
    candidate_specs,
    feature_types,
)


class NSS80SafeImprovementV4Tests(unittest.TestCase):
    def test_no_baseline_or_candidate_feature_is_known_leakage(self) -> None:
        all_candidate_features = set(BASE_FEATURES) | set(BASIC_ENGINEERED_FEATURES) | set(L5_FEATURES)
        self.assertFalse(all_candidate_features & PROHIBITED_FEATURES)

    def test_l5_aggregation_loads_no_block_9_payment_or_expenditure_fields(self) -> None:
        self.assertFalse(any(column.startswith("b9") for column in L5_ALLOWED_RAW_COLUMNS))
        self.assertNotIn("b8i13", L5_ALLOWED_RAW_COLUMNS)
        self.assertFalse(L5_ALLOWED_RAW_COLUMNS & L5_PROHIBITED_PAYMENT_RAW_COLUMNS)

    def test_every_official_ailment_category_has_one_broad_group(self) -> None:
        self.assertEqual(set(AILMENT.values()), set(AILMENT_GROUP_BY_LABEL))
        self.assertEqual(len(AILMENT_GROUP_BY_LABEL), len(AILMENT))

    def test_controlled_search_covers_all_requested_model_families_and_transforms(self) -> None:
        specs = candidate_specs()
        self.assertEqual(
            {spec.family for spec in specs},
            {
                "hist_gradient_boosting",
                "extra_trees",
                "random_forest",
                "gradient_boosting",
                "catboost",
                "xgboost",
            },
        )
        self.assertTrue({"identity", "log1p", "yeo_johnson"}.issubset({spec.target_transform for spec in specs}))

    def test_safe_service_counts_and_ratios(self) -> None:
        frame = pd.DataFrame(
            {
                "ailment_nature": [AILMENT["13"]],
                "medical_institution_type": ["Private hospital"],
                "age_years": [50.0],
                "length_of_stay_days": [8.0],
                "household_size": [4.0],
                "household_usual_consumer_expenditure_rs": [20000.0],
                "number_of_hospitalisations": [2.0],
                "surgery": [1.0],
                "medicine": [1.0],
                "xray_ecg_eeg_scan": [1.0],
                "other_diagnostic_tests": [0.0],
                "ward_type": ["Paying general ward"],
                "state": ["Kerala"],
            }
        )
        result = add_engineered_features(frame)
        self.assertEqual(float(result.loc[0, "service_use_count"]), 3.0)
        self.assertEqual(float(result.loc[0, "diagnostic_use_count"]), 1.0)
        self.assertEqual(float(result.loc[0, "household_expenditure_per_person"]), 5000.0)
        self.assertEqual(float(result.loc[0, "hospitalisations_per_household_member"]), 0.5)
        self.assertEqual(result.loc[0, "ailment_group"], "Cancer")

    def test_all_dynamic_binary_features_are_typed_as_binary(self) -> None:
        _, binary, _ = feature_types(sorted(set(BASE_FEATURES) | set(L5_FEATURES)))
        self.assertTrue(set(binary).issubset(BINARY_FEATURES))

    def test_serializable_classes_use_stable_import_path(self) -> None:
        expected = "model_training.nss80.safe_improvement_v4"
        self.assertEqual(CandidateSpec.__module__, expected)
        self.assertEqual(SafeRegressor.__module__, expected)


if __name__ == "__main__":
    unittest.main()
