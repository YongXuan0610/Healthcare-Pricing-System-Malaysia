# Project Overview

This update promotes the previously fitted NSS 80 practical 15-input model into the production prediction workflow. The model estimates hospitalisation-related medical expenditure for one Indian inpatient episode in INR. It remains a research estimator and is kept separate from Malaysian public pricing, private pricing, and LIAM reference data.

The fitted point estimator was not retrained. The selected experiment artifact was copied byte-for-byte into the production artifact location, and a new model-specific uncertainty artifact was fitted using non-test, person-disjoint data.

# Previous Production Model

The production workflow immediately before this update used `nss80_compact_25`, artifact version `isolated_full_data_25`.

- Algorithm: `HistGradientBoostingRegressor`
- Original inputs: 25
- Encoded inputs: 161
- Target transform: identity
- Locked-test R²: 0.4191132874
- Locked-test MAE: INR 19,087.17
- Locked-test RMSE: INR 56,605.08
- Locked-test median absolute error: INR 5,241.76
- Locked-test RMSLE: 3.34044
- Production artifact before promotion: `Backend/models/nss80/primary_model.joblib`
- SHA-256: `c7b171a27686ac6d08319a5b20e87f4fc21085df6feace848b39a2e33b938869`
- Interval artifact: `compact_25_intervals.joblib`
- Interval SHA-256: `9125517ed79a91d7df6c0c6b3b72b5241f12d3b0a77594bf59392ffec3124910`
- Rollback archive: `Backend/models/nss80/archive/production_compact25_before_practical15/`

The exact previous feature list was: `age_years`, `gender`, `relation_to_household_head`, `marital_status`, `highest_education_level`, `chronic_ailment`, `health_financing_or_insurance_coverage`, `medical_insurance_premium_rs`, `pregnant`, `communicable_disease`, `other_ailment_last_15_days`, `other_ailment_previous_day`, `number_of_hospitalisations`, `length_of_stay_days`, `ailment_nature`, `hospitalisation_treatment_nature`, `medical_institution_type`, `reason_not_using_government_public_hospital`, `ward_type`, `place_of_hospitalisation`, `treatment_state_code`, `surgery`, `medicine`, `xray_ecg_eeg_scan`, and `other_diagnostic_tests`.

# Final 15-Input Production Model

- Production version: `nss80_practical_15`
- Artifact version: `isolated_practical_15_full_data`
- Algorithm: `HistGradientBoostingRegressor`
- Architecture: global model with embedded preprocessing
- Target: `total_medical_expenditure_rs`
- Target transform: original/identity
- Currency: INR
- Country and survey: India, NSS 80th Round Schedule 25.0
- Training episodes: 93,511
- Locked outer test episodes: 23,277
- Training people: 84,854
- Locked-test people: 21,214
- Original feature count: 15
- Encoded feature count: 90
- Known leakage columns: 0
- Selected source: `Backend/models/nss80/practical_15_input_experiment/practical_15_features.joblib`
- Production artifact: `Backend/models/nss80/primary_model.joblib`
- Source and production SHA-256: `86cfc650a2a8ddfde94d64f8b4f123b34aa6dbc99052d99fe821265f00644695`

# Feature Reduction History

| Model | Inputs | Locked-test R² | Change |
|---|---:|---:|---|
| NSS 80 v4 | 37 | 0.42323 | Baseline production feature set |
| Compact 25 | 25 | 0.41911 | 12 fewer fields than v4 |
| Practical 15 | 15 | 0.39770 | 22 fewer than v4; 10 fewer than compact 25 |

The practical model reduces the 37-input design by 59.46% and the compact 25 design by 40.00%.

# Why 15 Inputs Were Selected

The 15-input model accepts a moderate loss in predictive performance to make the user workflow shorter and easier to complete. Its R² decreases by about 0.02142 from compact 25 and 0.02553 from v4, while removing ten fields from compact 25 and twenty-two fields from v4. This was an explicit product and research decision: the retained inputs describe the patient’s health context and the hospitalisation episode without requiring socioeconomic, insurance, location-code, or detailed diagnostic-service questions.

# Exact Final 15 Features

## Patient and Health Context

1. `age_years`
2. `gender`
3. `chronic_ailment`
4. `pregnant`
5. `communicable_disease`
6. `other_ailment_last_15_days`

## Hospitalisation Episode

7. `number_of_hospitalisations`
8. `length_of_stay_days`
9. `ailment_nature`
10. `hospitalisation_treatment_nature`
11. `medical_institution_type`
12. `ward_type`
13. `place_of_hospitalisation`
14. `surgery`
15. `medicine`

# Removed Features

The practical 15 promotion removes these compact 25 requirements: `relation_to_household_head`, `marital_status`, `highest_education_level`, `health_financing_or_insurance_coverage`, `medical_insurance_premium_rs`, `other_ailment_previous_day`, `reason_not_using_government_public_hospital`, `treatment_state_code`, `xray_ecg_eeg_scan`, and `other_diagnostic_tests`.

Fields removed earlier from the 37-input workflow remain absent: `household_size`, `household_usual_consumer_expenditure_rs`, `sector`, `state`, `nss_region`, `district`, `household_type`, `community_communicable_disease_outbreak`, `treated_on_medical_advice_before_hospitalisation`, `pre_hospitalisation_treatment_nature`, `pre_hospitalisation_level_of_care`, and `pre_hospitalisation_treatment_duration_days`.

None of these fields is defaulted, inferred, sent as a hidden input, or supplied to the estimator.

# Leakage Protection

The production feature list is disjoint from the known expenditure and target-leakage columns. `package_component_rs`, `doctor_surgeon_fee_rs`, `medicines_rs`, `diagnostic_tests_rs`, `bed_charges_rs`, `other_medical_expenses_rs`, `patient_transport_rs`, `other_non_medical_household_expenses_rs`, `total_expenditure_rs`, `insurance_or_employer_reimbursement_rs`, and `household_income_loss_due_to_hospitalisation_rs` remain excluded. Metadata records zero known leakage columns, and the target is not present in the input features.

`surgery` and `medicine` use only the leakage-safe received indicator: `Not received = 0`, `Received = 1`. Payment-status labels such as free, partly free, or paid are rejected.

# Model Training/Evaluation Methodology

The promoted point estimator is the saved result of the isolated practical-feature experiment. Person-grouped outer splitting produced 93,511 training episodes and 23,277 locked test episodes with zero person overlap. Within the outer training data, 74,617 selection-training episodes and 18,894 grouped-validation episodes were used to select 111 boosting iterations. The final estimator was refitted on all 93,511 outer-training episodes. The locked test was used for final evaluation and was not used for model selection, interval calibration, or coverage auditing.

The artifact contains its fitted preprocessing and 90 encoded inputs. Promotion copied the artifact without refitting or changing its serialized contents.

# Final Model Performance

| Metric | Unweighted locked test | Survey-weighted locked test |
|---|---:|---:|
| R² | 0.39770 | 0.39658 |
| MAE | INR 19,614.18 | INR 19,234.66 |
| RMSE | INR 57,639.14 | INR 54,551.48 |
| Median absolute error | INR 5,436.30 | INR 5,441.86 |
| RMSLE | 3.37371 | 3.43848 |

# High-Cost Performance

- Top 10% of locked-test costs: MAE INR 95,759.28
- Top 5%: MAE INR 150,076.50
- Top 1%: MAE INR 388,187.58

These values show the increasing error in the high-cost tail and should be discussed as a key limitation.

# Feature Importance

Permutation importance on a fixed 4,000-episode inner-validation sample ranked the strongest contributors as `length_of_stay_days`, `ward_type`, `surgery`, `ailment_nature`, `medical_institution_type`, and `chronic_ailment`. The mean R² decreases were approximately 0.2603, 0.2361, 0.0608, 0.0573, 0.0543, and 0.0148 respectively. This importance is predictive and does not establish a causal effect on expenditure.

# Backend Changes

- Promoted the existing fitted artifact into `Backend/models/nss80/primary_model.joblib`.
- Replaced production metadata with the practical 15 contract, metrics, split, lineage, leakage audit, and interval details.
- Added `production_practical15.py` for ordered features, codebook mappings, numeric limits, and safe transformations.
- Updated `nss80_prediction_service.py` to load the 15-input artifact and model-specific interval.
- Updated the Pydantic request schema to the exact 15 fields with extra fields forbidden.
- Preserved guest prediction, maintenance mode, authorization, bearer token, and CORS logic.
- Added direct artifact, service, HTTP, interval, serialization, validation, options, and access-control tests.

# Frontend Changes

The NSS form now has exactly two main sections and 15 visible model inputs: six under Patient and Health Context and nine under Hospitalisation Episode. TypeScript state, field definitions, validation, codebook options, and POST payload generation were reduced to the same exact contract. Obsolete headings, controls, state, validation, and payload keys were removed. The result continues to identify the model as an India NSS 80 research estimate in INR and not a Malaysian quotation.

Pregnancy remains conditional. It is required only when gender is Female and age is 15–49. Outside that survey question scope the frontend disables the control and sends `null`; the backend maps that to the codebook-supported missing value. It does not invent a Yes or No response.

# Prediction Interval / Recalibration

The compact 25 interval was model-specific and was not reused. A new `practical_15_intervals.joblib` artifact was created while the frozen practical 15 point estimator remained unchanged.

The method fits separate 5th and 95th percentile HistGradientBoosting quantile models and applies a nonnegative conformalized-quantile score. Calibration is performed at person level by taking the maximum score across each person’s episodes, followed by the finite-sample 90% person-coverage quantile.

- Interval fit: 65,406 episodes, 59,397 people
- Calibration: 14,120 episodes, 12,728 people
- Independent coverage audit: 13,985 episodes, 12,729 people
- Locked test held aside: 23,277 episodes, 21,214 people
- Relevant person overlaps: 0
- Calibration rank: 11,457
- Calibration statistic: INR 0.00
- Independent audit episode coverage: 93.47%
- Independent simultaneous person coverage: 93.22%
- Display-expanded audit episode coverage: 93.49%
- Artifact SHA-256: `caf9fa0418d6f20715bfa0dd9167b76d11122f96bb485eedb91a2670c778e784`

The 90% interval is a marginal, person-calibrated research interval and is not an individual guarantee. Display bounds may widen only to include the point estimate.

# API Changes

`POST /predict/nss80` accepts the exact 15-field schema, rejects missing required retained fields, and rejects extra removed or leakage fields. `GET /predict/nss80/options` returns only the required numerical constraints and readable categories. The options response reports `nss80_practical_15`, 15 original inputs, 90 encoded inputs, the final metrics, and interval availability.

# Production Artifact Verification

The selected source and deployed production artifacts both have SHA-256 `86cfc650a2a8ddfde94d64f8b4f123b34aa6dbc99052d99fe821265f00644695` and are byte-identical. Twelve fixed locked-test records were predicted with the source artifact, the reloaded production artifact, the production service mapping, and `POST /predict/nss80`. Source and reloaded predictions were exactly equal; mapped predictions agreed within `1e-8`; API values agreed within INR 0.00501, the expected two-decimal formatting tolerance. Evidence is stored in `Backend/reports/nss80/practical_15_promotion/api_equivalence.json`.

# Automated Test Results

- Backend compile/import validation: passed.
- Focused practical-15 production suite: 5 tests passed.
- Backend full suite in the restricted environment: 86 tests and 36 subtests passed; four unrelated pricing tests were network-limited because they query live Supabase, and one harmless joblib CPU-detection warning was emitted. The four failures were `test_private_total_uses_only_published_package_and_ward_prices`, `test_public_and_private_functions_exist_when_ml_is_unavailable`, `test_public_missing_match_returns_no_fabricated_fallback`, and `test_public_official_match_has_no_state_adjustment`. No unrelated test was changed to mask these failures.
- Frontend TypeScript (`npx tsc --noEmit`): passed.
- Frontend tests: 18 passed across five files.
- Frontend production build: passed; Vite reported the existing bundle-size advisory for a JavaScript chunk over 500 kB.
- ESLint: passed with zero errors and seven existing Fast Refresh warnings in shared UI component files.
- `git diff --check`: passed; Git reported only Windows LF-to-CRLF notices.

# Regression Results

Passing regression tests cover LIAM reference ranges, the isolated US benchmark, the Healthcare Service Assistant, controlled private-pricing missing-data behavior, guest predictions, maintenance mode, and API routing. The four full-suite failures were confined to tests that call the live Supabase-backed Malaysian public/private pricing database from a network-restricted environment; they were not NSS promotion regressions. Frontend tests cover the prediction form, LIAM reference, service assistant, and hospital pagination. The production build and TypeScript checks compile the authentication, prediction history, dashboard, profile, and admin pages. No access-control implementation was weakened or redesigned.

# Files Added

- `Backend/model_training/nss80/production_practical15.py`
- `Backend/model_training/nss80/practical15_intervals.py`
- `Backend/models/nss80/practical_15_intervals.joblib`
- `Backend/models/nss80/archive/production_compact25_before_practical15/` and its manifest
- `Backend/reports/nss80/practical_15_promotion/` calibration, split, before-state, sample request, and equivalence evidence
- `Backend/tests/test_nss80_practical15_production.py`
- `Practical_15_NSS_Final_Production_Update_Summary.md`

# Files Modified

- `Backend/models/nss80/primary_model.joblib`
- `Backend/models/nss80/metadata.json`
- `Backend/app/services/nss80_prediction_service.py`
- `Backend/app/schemas/prediction.py`
- `Backend/scripts/main.py`
- `Backend/tests/test_prediction.py`
- `Backend/tests/test_component_separation.py`
- `Frontend/src/components/AIPredictionForm.tsx`
- `Frontend/src/test/AIPredictionForm.test.tsx`

The obsolete compact-25 production-specific test file was replaced by the practical-15 production suite. Historical compact, v4, 21-input, comparison, experiment, report, and prediction evidence remains in the project.

# Files Archived

`Backend/models/nss80/archive/production_compact25_before_practical15/` contains the immediately previous `primary_model.joblib`, metadata, compact interval artifact, compact production contract, checksums, version identifiers, feature counts, and metrics. The compact point-model SHA is `c7b171a27686ac6d08319a5b20e87f4fc21085df6feace848b39a2e33b938869`.

# Important FYP2 Report Updates

The FYP2 report, poster, and figures were deliberately left unchanged. A later report update should review:

- Abstract: identify practical 15 as the final deployed NSS model if the abstract states the deployed feature count or final R².
- Chapter 3 ML Methodology: document the 37→25→15 comparison, grouped split, 111 iterations, final refit, and model-specific person calibration.
- Chapter 4 ML Module Design: update the production input contract and two-section form architecture.
- Chapter 5.6 Machine Learning Implementation: replace current-model references to v4/compact 25, list the exact 15 inputs, and explain pregnancy and service encoding.
- Chapter 6.6 Machine Learning Evaluation: report the final unweighted, weighted, high-cost, and interval audit results while retaining prior models as comparisons.
- Chapter 6.8 Limitations and Discussion: discuss R² 0.39770, high-cost tail errors, India/INR scope, marginal interval interpretation, and the usability-performance trade-off.
- Chapter 7: align conclusions and future work with the final practical 15 deployment.
- Figure 5.10: update if it depicts the former production model, feature count, API contract, or metrics.
- Figure 5.11: update if screenshots show obsolete form fields or previous model metrics.

# Final Current System State

The authoritative deployed NSS model is `nss80_practical_15` at `Backend/models/nss80/primary_model.joblib`. It uses exactly 15 leakage-safe inputs, embedded preprocessing with 90 encoded features, an identity target in INR, and the new practical-15 interval artifact. The frontend and API share this contract. The previous deployed model and all research evidence remain preserved for rollback and FYP analysis.
