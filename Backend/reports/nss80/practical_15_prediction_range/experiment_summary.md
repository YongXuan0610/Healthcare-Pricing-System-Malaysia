# Practical 15 Prediction-Range Experiment

## Objective and decision

This isolated experiment compared nominal 80% and 90% conditional prediction ranges for the deployed practical 15-input NSS 80 point model. The **80% range was selected for the user-facing production result** because it retained 88.47% empirical locked-test coverage while reducing mean interval width by 24.54% and median width by 29.37% relative to the 90% candidate. This choice improves ordinary-user readability without presenting the point prediction as exact.

The 90% candidate remains available as experiment evidence. It provides better high-cost coverage, so the reduced coverage in the high-cost tail remains an explicit limitation of the selected 80% range.

## Unchanged point model and inputs

The point artifact remained byte-identical before and after the experiment with SHA-256 `86cfc650a2a8ddfde94d64f8b4f123b34aa6dbc99052d99fe821265f00644695`. It was not retrained.

Both interval candidates use only the same 15 leakage-safe features: `age_years`, `gender`, `chronic_ailment`, `pregnant`, `communicable_disease`, `other_ailment_last_15_days`, `number_of_hospitalisations`, `length_of_stay_days`, `ailment_nature`, `hospitalisation_treatment_nature`, `medical_institution_type`, `ward_type`, `place_of_hospitalisation`, `surgery`, and `medicine`. Known leakage columns equal zero.

## Method

Four auxiliary `HistGradientBoostingRegressor` models were fitted with identity targets and quantile losses at 0.05, 0.10, 0.90, and 0.95. A person-disjoint split inside the interval fitting partition selected stopping iterations by survey-weighted validation pinball loss. The selected iterations were 1, 24, 115, and 100 respectively.

The final quantile models were refitted on 65,406 interval-fit episodes from 59,397 people. Person-level conformalized quantile calibration used 14,120 episodes from 12,728 different people. An independent audit used 13,985 episodes from 12,729 further people. All relevant person overlaps were zero. The locked 23,277 episodes were used only for final evaluation.

Both calibration adjustments were INR 0.00 because the raw conditional ranges already exceeded their nominal coverage on the person-level calibration distribution. Quantile crossing was zero on both selection validation and calibration. The defined crossing policy is a conservative monotone envelope that raises the upper bound to the lower bound if crossing occurs; it does not arbitrarily swap bounds. Lower bounds are clipped at zero.

## Locked-test comparison

| Interval | Empirical coverage | Mean width | Median width | P25 width | P75 width | P90 width | Lower miss | Upper miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 80% | 88.47% | INR 54,548.04 | INR 26,496.39 | INR 6,489.91 | INR 70,838.50 | INR 125,093.19 | 0.009% | 11.52% |
| 90% | 94.01% | INR 72,286.37 | INR 37,514.70 | INR 9,591.84 | INR 93,771.77 | INR 164,011.43 | 0.009% | 5.98% |

Independent audit coverage was 88.43% for the 80% candidate and 93.82% for the 90% candidate. The close audit/test values support stable generalization under the study split.

## Coverage by expenditure band

| Interval | Overall | Top 10% | Top 5% | Top 1% |
|---|---:|---:|---:|---:|
| 80% | 88.47% | 56.55% | 45.45% | 14.83% |
| 90% | 94.01% | 71.27% | 57.42% | 27.54% |

Both candidates cover ordinary cases substantially better than the most expensive cases. Nearly all misses are above the upper bound. The system must therefore retain a clear research disclaimer, especially for unusually expensive admissions.

## Coverage by institution type

| Interval | Government/public | Private | Charitable/trust/NGO |
|---|---:|---:|---:|
| 80% | 89.31% | 87.51% | 92.06% |
| 90% | 94.77% | 93.18% | 96.63% |

Private-hospital ranges are much wider: the selected 80% candidate has mean widths of INR 14,206.59 for government/public hospitals and INR 92,565.07 for private hospitals.

## Production integration

Production metadata now selects `models/nss80/practical_15_prediction_range/practical_15_interval_80.joblib`, SHA-256 `cc2895364b480186ef45c37e7255583861d979e41dd5997ee33be9d040da28b4`. The API retains its point and nested range fields and additionally returns `central_prediction`, `lower_bound`, `upper_bound`, and `interval_level`.

The frontend now presents “Estimated Medical Expenditure Range” as the main result, followed by the nominal 80% explanation, a simple range bar with a central marker, the central estimate in secondary text, the uncertainty explanation, and the existing India/INR research disclaimer.

## Prediction history

The existing `prediction_history` frontend and TypeScript contract supports Malaysian `public` and `private` pricing records only. The NSS research form does not currently write into that table, and no database migration proving support for an NSS record type exists in the repository. The interval update therefore did not force NSS data into that table or change the rendering of existing point-only public/private history records.

## Artifacts and evidence

- `models/nss80/practical_15_prediction_range/quantile_05.joblib`
- `models/nss80/practical_15_prediction_range/quantile_10.joblib`
- `models/nss80/practical_15_prediction_range/quantile_90.joblib`
- `models/nss80/practical_15_prediction_range/quantile_95.joblib`
- `models/nss80/practical_15_prediction_range/practical_15_interval_80.joblib`
- `models/nss80/practical_15_prediction_range/practical_15_interval_90.joblib`
- `reports/nss80/practical_15_prediction_range/interval_comparison.csv`
- `reports/nss80/practical_15_prediction_range/test_interval_predictions.csv`
- `reports/nss80/practical_15_prediction_range/coverage_by_cost_band.csv`
- `reports/nss80/practical_15_prediction_range/coverage_by_institution.csv`
- `reports/nss80/practical_15_prediction_range/coverage_plot.png`
- `reports/nss80/practical_15_prediction_range/interval_width_plot.png`

