# Dataset Inspection Report

## Source and schema

- Raw shape: **1338 rows × 8 columns**.
- Source columns: `record_id, age, sex, bmi, children, smoker, region, charges_usd`.
- The supplied `charges_usd` column is explicitly mapped to internal target `charges`; the source CSV is not changed.
- `record_id` is an identifier and is not used as a predictor.
- Modeling view after the documented duplicate policy: **1337 rows**.

## Data types

| Column | Source dtype |
|---|---|
| record_id | int64 |
| age | int64 |
| sex | str |
| bmi | float64 |
| children | int64 |
| smoker | str |
| region | str |
| charges_usd | float64 |

## Data quality

- Missing values: `{"age": 0, "bmi": 0, "charges_usd": 0, "children": 0, "record_id": 0, "region": 0, "sex": 0, "smoker": 0}`.
- Exact duplicates including `record_id`: **0**.
- Duplicate observations excluding `record_id`: **1**.
- Duplicate records: record_id=196 (19, male, BMI 30.59, children 0, smoker no, northwest, charge 1639.5631); record_id=582 (19, male, BMI 30.59, children 0, smoker no, northwest, charge 1639.5631).
- Policy: Source file preserved; later duplicate excluded from modeling view to avoid split leakage.
- Invalid-value checks: `{"age_outside_0_120": 0, "bmi_nonpositive_or_over_100": 0, "charges_negative": 0, "children_negative_or_noninteger": 0, "invalid_region": 0, "invalid_sex": 0, "invalid_smoker": 0, "non_numeric_age": 0, "non_numeric_bmi": 0, "non_numeric_charges": 0, "non_numeric_children": 0}`.

## Ranges and categories

- Age: 18–64 years.
- BMI: 15.96–53.13.
- Children: 0–5.
- Sex: `{"female": 662, "male": 676}`.
- Smoker: `{"no": 1064, "yes": 274}`.
- Region: `{"northeast": 324, "northwest": 325, "southeast": 364, "southwest": 325}`.

## Charges distribution

- Minimum: 1121.87; maximum: 63770.43.
- Mean: 13270.42; median: 9382.03; standard deviation: 12110.01.
- Skewness: 1.516, indicating a pronounced right tail.
- 25th/75th percentiles: 4740.29 / 16639.91.
- 95th/99th percentiles: 41181.83 / 48537.48.

## Basic descriptive statistics

| Variable | Count | Mean | Std. dev. | Minimum | Median | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| age | 1338 | 39.2070 | 14.0500 | 18.0000 | 39.0000 | 64.0000 |
| bmi | 1338 | 30.6634 | 6.0982 | 15.9600 | 30.4000 | 53.1300 |
| children | 1338 | 1.0949 | 1.2055 | 0.0000 | 1.0000 | 5.0000 |
| charges | 1338 | 13270.4223 | 12110.0112 | 1121.8739 | 9382.0330 | 63770.4280 |

## Context limitation

This is the supplied published US medical-charge dataset, not Malaysian or NSS patient-level data. The bundle documentation describes the classic dataset as simulated from US demographic statistics. It is retained only as the secondary benchmark. No rows were fabricated or merged with NSS, LIAM, or Malaysian hospital-pricing data.
