# Practical 15-input NSS 80 experiment

**Recommendation: PROMOTE 15-INPUT MODEL.** This experiment did not promote a model.

## Design

The experiment reused the exact corrected outer train (93,511 episodes), locked outer test (23,277), inner selection train (74,617), and inner validation (18,894) IDs from the controlled 25-input comparison. All relevant person overlaps are zero. The HGB algorithm, identity target, preprocessing, normalized survey weights, base parameters, metric code, and high-cost definitions are unchanged.

Grouped validation stopped at 111 iterations. The final experimental estimator was refitted on all 93,511 outer-training episodes with early stopping disabled and that iteration count fixed. The locked test was not used for tuning or stopping.

## Locked-test comparison

| model | inputs | input_reduction_vs_37_percent | mae | rmse | r2 | median_absolute_error | rmsle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Previous v4 production | 37 | 0.00 | 19,105.80 | 56,404.11 | 0.42323 | 5,170.30 | 3.22002 |
| Compact model | 25 | 32.43 | 19,087.17 | 56,605.08 | 0.41911 | 5,241.76 | 3.34044 |
| New practical model | 15 | 59.46 | 19,614.18 | 57,639.14 | 0.39770 | 5,436.30 | 3.37371 |

## Pairwise differences

| comparison | metric | difference | interpretation |
| --- | --- | --- | --- |
| 15 minus 25 | mae | 527.02 | worse |
| 15 minus 25 | rmse | 1,034.06 | worse |
| 15 minus 25 | r2 | -0.02 | worse |
| 15 minus 25 | median_absolute_error | 194.53 | worse |
| 15 minus 25 | rmsle | 0.03 | worse |
| 15 minus previous production | mae | 508.39 | worse |
| 15 minus previous production | rmse | 1,235.03 | worse |
| 15 minus previous production | r2 | -0.03 | worse |
| 15 minus previous production | median_absolute_error | 266.00 | worse |
| 15 minus previous production | rmsle | 0.15 | worse |

Positive error differences are worse; positive R² differences are better.

## Survey-weighted metrics

| model | inputs | mae | rmse | r2 | median_absolute_error | rmsle |
| --- | --- | --- | --- | --- | --- | --- |
| Previous v4 production | 37 | 18,321.44 | 53,118.30 | 0.42787 | 4,832.74 | 3.25122 |
| Compact model | 25 | 18,520.10 | 53,366.19 | 0.42252 | 5,074.68 | 3.40165 |
| New practical model | 15 | 19,234.66 | 54,551.48 | 0.39658 | 5,441.86 | 3.43848 |

## High-cost comparison

| model | band | cutoff_inr | episodes | mae | rmse | r2 | median_absolute_error | rmsle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Compact model | top10 | 70,000.00 | 2412 | 92,848.08 | 163,811.83 | 0.04469 | 50,588.14 | 0.83915 |
| Compact model | top5 | 120,000.00 | 1186 | 145,384.55 | 225,583.79 | -0.21509 | 93,490.61 | 1.00748 |
| Compact model | top1 | 320,000.00 | 236 | 373,445.13 | 454,889.25 | -1.69591 | 317,383.71 | 1.40542 |
| New practical model | top10 | 70,000.00 | 2412 | 95,759.28 | 166,901.53 | 0.00831 | 51,178.01 | 0.87742 |
| New practical model | top5 | 120,000.00 | 1186 | 150,076.50 | 230,346.13 | -0.26694 | 97,692.41 | 1.05405 |
| New practical model | top1 | 320,000.00 | 236 | 388,187.58 | 463,787.40 | -1.80241 | 328,837.89 | 1.46420 |

Legitimate expensive episodes were retained. Tail results are diagnostic and were not used to tune the estimator.

## Predictive importance

| feature | r2_decrease_mean | r2_decrease_std |
| --- | --- | --- |
| length_of_stay_days | 0.26 | 0.02 |
| ward_type | 0.24 | 0.03 |
| surgery | 0.06 | 0.02 |
| ailment_nature | 0.06 | 0.00 |
| medical_institution_type | 0.05 | 0.02 |
| chronic_ailment | 0.01 | 0.01 |
| gender | 0.00 | 0.02 |
| number_of_hospitalisations | 0.00 | 0.01 |
| pregnant | 0.00 | 0.00 |
| age_years | 0.00 | 0.01 |
| communicable_disease | 0.00 | 0.00 |
| hospitalisation_treatment_nature | 0.00 | 0.00 |
| medicine | 0.00 | 0.00 |
| other_ailment_last_15_days | -0.00 | 0.00 |
| place_of_hospitalisation | -0.03 | 0.00 |

Permutation importance measures predictive contribution conditional on the other inputs. It is not causal, and negative values are reported rather than hidden.

Requested individual values:

- `number_of_hospitalisations`: 0.001814 mean validation R² decrease.
- `medicine`: 0.000000 mean validation R² decrease.
- `chronic_ailment`: 0.014781 mean validation R² decrease.
- `communicable_disease`: 0.000155 mean validation R² decrease.
- `pregnant`: 0.000646 mean validation R² decrease.
- `other_ailment_last_15_days`: -0.000399 mean validation R² decrease.

## Input burden

- 37 → 15: 22 inputs removed, a 59.46% reduction.
- 25 → 15: 10 inputs removed, a 40.00% reduction.
- Possible future layout: six Patient and Health Context fields plus nine Hospitalisation Episode fields.

## Supervisor-friendly explanation

The 15 fields were chosen because they describe the patient’s basic health context and the hospitalisation itself in terms ordinary users can answer. Socioeconomic, household, insurance-detail, location-code, and diagnostic-service questions were removed; no removed answer was reconstructed. This reduced the form by 59.46% from 37 inputs and 40% from the current 25-input model. On the same locked test, R² changed from 0.41911 to 0.39770, while MAE changed from INR 19,087.17 to INR 19,614.18. The recommendation is PROMOTE 15-INPUT MODEL.

## Audit conclusion

Known leakage columns: zero. Surgery and medicine use only 0=Not received and 1=Received. Removed compact fields were not supplied, imputed, inferred, or reconstructed. In-memory predictions match exactly before and after reload; independently regenerated full-test predictions match the saved decimal CSV within `5.82e-11` INR. All 23,277 test predictions are finite and nonnegative. Production and protected-file hashes match their pre-experiment values.

Python compilation and the experiment smoke test passed. Independent metric, reload, leakage, overlap, and protected-file verification passed. Relevant NSS backend tests passed 18/18, and the unchanged frontend regression suite passed 17/17.

Runtime: 115.6 seconds.

The current production model, frontend form, API schema, NSS options endpoint, FYP report, Figure 5.10, and Figure 5.11 were not changed.
