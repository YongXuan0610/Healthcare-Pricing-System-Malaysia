"""Scientific lineage and non-merging checks for the three core components."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import unittest

import pandas as pd
import joblib

from model_training.nss80.data import (
    FEATURES as LEGACY_FEATURES,
    TARGET,
    load_and_prepare_dataset,
)
from model_training.nss80.modeling import leakage_safe_features
from model_training.nss80.production_practical15 import FEATURES as PRACTICAL_FEATURES


BACKEND_ROOT = Path(__file__).resolve().parents[1]


class ComponentSeparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nss_metadata = json.loads(
            (BACKEND_ROOT / "models" / "nss80" / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        cls.us_metadata = json.loads(
            (
                BACKEND_ROOT / "models" / "us_kaggle" / "model_metadata.json"
            ).read_text(encoding="utf-8")
        )

    def test_research_roles_and_currencies_are_explicit(self) -> None:
        self.assertEqual(self.nss_metadata["research_role"], "primary")
        self.assertEqual(self.nss_metadata["target_unit"], "INR per inpatient case")
        self.assertEqual(self.us_metadata["research_role"], "secondary_benchmark")
        self.assertIn("USD", self.us_metadata["output_currency"])

    def test_nss_split_has_no_person_overlap(self) -> None:
        self.assertEqual(self.nss_metadata["split"]["person_overlap_count"], 0)
        self.assertEqual(
            self.nss_metadata["split"]["selection_validation_person_overlap_count"],
            0,
        )
        self.assertFalse(self.nss_metadata["split"]["test_used_for_selection"])
        self.assertEqual(self.nss_metadata["dataset"]["modeling_rows"], 116788)

    def test_nss_practical15_excludes_leakage_and_payment_status(self) -> None:
        self.assertEqual(self.nss_metadata["model_version"], "nss80_practical_15")
        self.assertEqual(self.nss_metadata["input_features"], PRACTICAL_FEATURES)
        self.assertEqual(self.nss_metadata["original_feature_count"], 15)
        self.assertEqual(self.nss_metadata["encoded_feature_count"], 90)
        transform = self.nss_metadata["feature_transform"]
        self.assertFalse(transform["payment_status_detail_used"])
        self.assertFalse(transform["free_medical_service_used"])
        self.assertEqual(
            self.nss_metadata["leakage_guard"]["known_leakage_columns_in_model"],
            [],
        )
        self.assertFalse(self.nss_metadata["leakage_guard"]["target_in_features"])

    def test_payment_detail_changes_do_not_change_model_features(self) -> None:
        base = {
            "age_years": 40,
            "length_of_stay_days": 5,
            "gender": "male",
            "sector": "urban",
            "state": "Delhi",
            "ailment": "All other fever",
            "treatment_system": "allopathy",
            "medical_institution": "private_hospital",
            "ward_type": "paying_general",
            "surgery": "received_free",
            "medicine": "received_on_payment",
            "imaging": "not_received",
            "other_diagnostics": "received_partly_free",
            "free_medical_service": "yes_private_charitable_or_ngo_provider",
        }
        alternative = {
            **base,
            "ward_type": "free",
            "surgery": "received_on_payment",
            "medicine": "received_free",
            "other_diagnostics": "received_on_payment",
            "free_medical_service": "no",
        }
        first = leakage_safe_features(pd.DataFrame([base], columns=LEGACY_FEATURES))
        second = leakage_safe_features(pd.DataFrame([alternative], columns=LEGACY_FEATURES))
        pd.testing.assert_frame_equal(first, second)

    def test_nss_practical15_artifact_loads_with_embedded_preprocessing(self) -> None:
        artifact = joblib.load(
            BACKEND_ROOT / "models" / "nss80" / "primary_model.joblib"
        )
        self.assertEqual(artifact["artifact_type"], "global_safe_regressor")
        self.assertEqual(
            artifact["model_version"],
            "isolated_practical_15_full_data",
        )
        self.assertEqual(artifact["features"], PRACTICAL_FEATURES)
        self.assertEqual(artifact["target_transform"], "identity")
        self.assertEqual(artifact["model_family"], "hist_gradient_boosting")
        self.assertEqual(artifact["encoded_feature_count"], 90)
        self.assertEqual(artifact["known_leakage_columns"], [])
        self.assertIsNotNone(artifact["estimator"].preprocessor_)

    def test_practical15_production_hash_matches_metadata(self) -> None:
        def sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        production = BACKEND_ROOT / "models" / "nss80" / "primary_model.joblib"
        self.assertEqual(
            sha256(production),
            self.nss_metadata["model_sha256"],
        )

    def test_nss_preparation_excludes_target_components(self) -> None:
        raw_dir = BACKEND_ROOT / "datasets" / "nss80" / "raw"
        if not raw_dir.exists():
            self.skipTest("Raw NSS survey files are intentionally not distributed")
        audit = load_and_prepare_dataset(raw_dir)
        self.assertEqual(len(audit.modeling), 121559)
        self.assertIn(TARGET, audit.modeling.columns)
        self.assertTrue(set(LEGACY_FEATURES).issubset(audit.modeling.columns))
        self.assertTrue(
            set(LEGACY_FEATURES).isdisjoint({"b7i6", "b7i7", "b7i12", "b7i15"})
        )
        self.assertEqual(audit.report["duplicate_case_ids"], 0)

    def test_liam_is_a_reference_table_not_an_ml_artifact(self) -> None:
        liam = pd.read_csv(
            BACKEND_ROOT
            / "datasets"
            / "malaysia_liam"
            / "reference"
            / "Medical_Dataset_Malaysia_LIAM_Price_Ranges.csv"
        )
        self.assertEqual(len(liam), 760)
        self.assertEqual(int(liam["data_status"].eq("Published").sum()), 550)
        self.assertEqual(
            int(liam["data_status"].eq("Insufficient credible data").sum()), 210
        )
        self.assertFalse(
            (BACKEND_ROOT / "models" / "malaysia_liam").exists(),
            "LIAM must not have a trained-model artifact directory",
        )


if __name__ == "__main__":
    unittest.main()
