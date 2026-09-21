# Practical 15 NSS Prediction Range Update Summary

## Why the point-only presentation changed

The practical 15-input NSS 80 model predicts highly variable hospitalisation-related expenditure. A large single predicted value could be interpreted as an exact cost even though locked-test R² is 0.39770 and high-cost errors remain large. The production UI now presents a learned expenditure range as the primary result and retains the central point estimate in secondary text and the API.

## Inputs and point model preserved

The production workflow still uses exactly the same 15 user inputs: `age_years`, `gender`, `chronic_ailment`, `pregnant`, `communicable_disease`, `other_ailment_last_15_days`, `number_of_hospitalisations`, `length_of_stay_days`, `ailment_nature`, `hospitalisation_treatment_nature`, `medical_institution_type`, `ward_type`, `place_of_hospitalisation`, `surgery`, and `medicine`.

The point model was not retrained or overwritten. Its SHA-256 remained `86cfc650a2a8ddfde94d64f8b4f123b34aa6dbc99052d99fe821265f00644695` before and after the experiment. Known leakage columns equal zero.

## Interval methodology

Four auxiliary `HistGradientBoostingRegressor` models use the same preprocessing principles and identity target:

- 5th percentile for the nominal 90% lower bound
- 10th percentile for the nominal 80% lower bound
- 90th percentile for the nominal 80% upper bound
- 95th percentile for the nominal 90% upper bound

A person-disjoint stopping-selection split used 52,307 training episodes/47,517 people and 13,099 validation episodes/11,880 people. Survey-weighted validation pinball loss selected 1, 24, 115, and 100 iterations for the 5th, 10th, 90th, and 95th percentile models.

Final quantile models were fitted on 65,406 episodes from 59,397 people. Calibration used 14,120 episodes from 12,728 separate people and maximum nonconformity per person. Independent audit used 13,985 episodes from 12,729 further people. All relevant person overlaps were zero. The locked 23,277 episodes/21,214 people were excluded from model selection and calibration and used only for final evaluation.

Both calibration statistics were INR 0.00 because the raw intervals already over-covered their nominal levels. Quantile crossing was zero. The documented crossing policy uses a conservative monotone envelope, raising the upper bound to the lower bound if necessary. All lower bounds are clipped to zero.

## Interval comparison

| Interval | Empirical Coverage | Mean Width | Median Width | Top 10% Coverage | Top 5% Coverage | Top 1% Coverage |
|---|---:|---:|---:|---:|---:|---:|
| 80% | 88.47% | INR 54,548.04 | INR 26,496.39 | 56.55% | 45.45% | 14.83% |
| 90% | 94.01% | INR 72,286.37 | INR 37,514.70 | 71.27% | 57.42% | 27.54% |

The selected nominal 80% range reduced mean width by 24.54% and median width by 29.37% compared with the 90% candidate, while retaining 88.47% overall locked-test coverage. Its independent audit coverage was 88.43%, close to the locked-test result. The 90% candidate covers high-cost cases better, and this trade-off must remain part of the limitations discussion.

Width percentiles for 80% were INR 6,489.91 (P25), INR 26,496.39 (P50), INR 70,838.50 (P75), and INR 125,093.19 (P90). The corresponding 90% widths were INR 9,591.84, INR 37,514.70, INR 93,771.77, and INR 164,011.43.

For 80%, coverage was 89.31% in government/public hospitals, 87.51% in private hospitals, and 92.06% in charitable/trust/NGO hospitals. Private-hospital intervals were materially wider.

## Chosen production range

The nominal 80% range was selected for the user-facing result. It gives ordinary users a materially narrower range than the 90% candidate while delivering conservative 88.47% empirical overall coverage. It is presented as an estimate based on similar NSS 80 hospitalisation records, never as a guarantee or quotation.

Production interval artifact:

`Backend/models/nss80/practical_15_prediction_range/practical_15_interval_80.joblib`

SHA-256: `cc2895364b480186ef45c37e7255583861d979e41dd5997ee33be9d040da28b4`

The 90% comparison artifact remains at `Backend/models/nss80/practical_15_prediction_range/practical_15_interval_90.joblib` with SHA-256 `b6bae32bc4579c829d02b56bffddc865f5327b1f630faf9db84dc97db19a50c0`.

## Backend and API changes

Production metadata now identifies the selected 80% artifact, quantiles, calibration partitions, coverage, widths, crossing policy, point-model hash, and selection reason. The NSS service verifies that the interval artifact matches the point hash, exact feature order, and configured nominal level.

The response preserves `predicted_medical_expenditure`, `predicted_cost`, `predicted_medical_expenditure_range`, and `predicted_cost_range`. It additionally exposes:

```json
{
  "central_prediction": 5432.31,
  "lower_bound": 0.0,
  "upper_bound": 11400.21,
  "interval_level": 0.8,
  "currency": "INR",
  "model_version": "nss80_practical_15"
}
```

Authentication, bearer-token handling, Guest Prediction, Maintenance Mode, validation, and NSS access rules were preserved.

## Frontend changes

The main result now reads “Estimated Medical Expenditure Range” and shows the formatted INR lower and upper values as the largest text. It explains the selected 80% level, shows a simple responsive range bar with a central marker, presents the central model estimate secondarily, and explains that actual expenditure varies. The India NSS 80, INR, research-estimate, and non-Malaysian-quotation wording remains.

## Prediction history

History rendering now remains safe for old point-only rows and can render an interval-shaped NSS row with lower bound, upper bound, level, and central estimate. The current repository contains only a public/private pricing-history TypeScript/database contract and no migration proving that `nss80` is accepted by the live table. The NSS form therefore does not force a new history write or risk breaking the existing database. A future database migration can use the already-tested interval rendering contract.

## Tests and regression

- Python compilation: passed.
- Focused practical-15 point/range tests: 10 passed.
- Full backend suite: 91 tests and 36 subtests passed; four unchanged Malaysian public/private pricing tests were network-limited by blocked live Supabase access. One joblib CPU-detection warning was harmless.
- Frontend TypeScript: passed.
- Frontend tests: 19 passed across six files, including range-first rendering and old/new history compatibility.
- ESLint: zero errors and seven existing Fast Refresh warnings.
- Production build: passed with the existing Vite large-chunk advisory.
- `git diff --check`: passed with Windows line-ending notices only.

Malaysian public/private pricing logic, LIAM, and the US benchmark were not modified by this range experiment. Their relevant offline tests passed; four live-data pricing tests remain separately environment-limited.

## Evidence and artifact paths

- `Backend/reports/nss80/practical_15_prediction_range/experiment_summary.md`
- `Backend/reports/nss80/practical_15_prediction_range/interval_comparison.csv`
- `Backend/reports/nss80/practical_15_prediction_range/interval_comparison.json`
- `Backend/reports/nss80/practical_15_prediction_range/test_interval_predictions.csv`
- `Backend/reports/nss80/practical_15_prediction_range/coverage_by_cost_band.csv`
- `Backend/reports/nss80/practical_15_prediction_range/coverage_by_institution.csv`
- `Backend/reports/nss80/practical_15_prediction_range/interval_width_statistics.json`
- `Backend/reports/nss80/practical_15_prediction_range/calibration_summary.md`
- `Backend/reports/nss80/practical_15_prediction_range/coverage_plot.png`
- `Backend/reports/nss80/practical_15_prediction_range/interval_width_plot.png`
- `Backend/reports/nss80/practical_15_prediction_range/api_response_example.json`
- `Backend/models/nss80/practical_15_prediction_range/`

## FYP reporting implications

The FYP report should describe the displayed range as a nominal 80% person-calibrated conditional quantile range with 88.47% empirical locked-test episode coverage. It should report the width distribution and high-cost coverage rather than implying uniform 80% protection for every subgroup. The central prediction remains part of evaluation and API transparency but is no longer the main user-facing amount.
