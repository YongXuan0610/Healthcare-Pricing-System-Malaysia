# Fixed practical 15-input feature set

```python
FINAL_15_FEATURES = [
  "age_years",
  "gender",
  "chronic_ailment",
  "pregnant",
  "communicable_disease",
  "other_ailment_last_15_days",
  "number_of_hospitalisations",
  "length_of_stay_days",
  "ailment_nature",
  "hospitalisation_treatment_nature",
  "medical_institution_type",
  "ward_type",
  "place_of_hospitalisation",
  "surgery",
  "medicine"
]
```

## Removed from compact 25

- `relation_to_household_head`
- `marital_status`
- `highest_education_level`
- `health_financing_or_insurance_coverage`
- `medical_insurance_premium_rs`
- `other_ailment_previous_day`
- `reason_not_using_government_public_hospital`
- `treatment_state_code`
- `xray_ecg_eeg_scan`
- `other_diagnostic_tests`

The removed fields are not supplied, defaulted, inferred, reconstructed, or imputed as hidden answers.

37 → 15 removes 22 inputs (59.46%). 25 → 15 removes 10 inputs (40.00%). A possible future form would contain six Patient and Health Context fields and nine Hospitalisation Episode fields. No production form was changed.