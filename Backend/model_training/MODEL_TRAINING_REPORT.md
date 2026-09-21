# US Kaggle Secondary Benchmark Training Report

## Dataset

The supplied `Insurance_Dataset_US_Medical_Charges_1338.csv` contains 1,338 published observations. Its predictors are age, sex, BMI, children, smoking status, and US region; the source target column is `charges_usd`, mapped explicitly to internal name `charges`. `record_id` is excluded from modeling. The bundle documentation describes the classic dataset as simulated from US demographic statistics. It is not Malaysian or NSS patient-level data and is retained only as the project's secondary benchmark.

One duplicated observation (excluding its distinct record IDs) was documented and removed only from the modeling view, leaving 1337 observations. The original CSV remains unchanged. No LIAM or Malaysian price-reference records were merged.

## Data Preprocessing

The modeling view was first split into 1069 training rows (80%) and 268 testing rows (20%) using `random_state=42`. The training partition was then divided into 855 development rows and 214 untouched calibration rows. All models use a Scikit-learn `Pipeline` containing a `ColumnTransformer`: numeric fields use median imputation and standardization; categorical fields use most-frequent imputation and one-hot encoding with unknown-category protection. Preprocessing is fitted inside each development/CV fold, preventing leakage. The calibration partition is used only to construct the prediction range, and the same held-out test indices are used for every final comparison.

## Exploratory Analysis

Charges are right-skewed (skewness 1.515). Smoking status has the largest visible group separation. Numerical correlations with charges are age 0.298, BMI 0.198, and children 0.067. Smokers average 32,050.23, compared with 8,440.66 for non-smokers. Male/female means are 13,975.00/12,569.58. The Southeast has the highest regional mean (14,735.41); the number-of-children groups are non-monotonic, with small samples for four and five children. These are unadjusted associations, not causal effects.

Useful EDA figures are saved under `model_training/plots/`.

## Models Evaluated

The comparison includes a mean-predicting Dummy Regressor baseline, Linear Regression, an unconstrained Decision Tree, Random Forest, Gradient Boosting, and tuned Random Forest/Gradient Boosting pipelines. XGBoost was not installed and was intentionally omitted because the required Scikit-learn ensembles already provide strong comparison models without an extra compiled dependency. SHAP was also unnecessary; native feature importance is used.

## Evaluation Metrics

- **MAE** is the mean absolute prediction error in the source dataset's charge scale; lower is better.
- **RMSE** penalizes large errors more heavily and is also in the source charge scale; lower is better.
- **R²** is the proportion of test-target variance explained by the model; higher is better. It is not classification accuracy.

## Model Comparison

| Model | Test MAE | Test RMSE | Test R² | CV RMSE (mean ± SD) | CV R² (mean ± SD) |
|---|---:|---:|---:|---:|---:|
| Tuned Gradient Boosting | 2,472.02 | 4,248.99 | 0.9018 | 4,386.56 ± 602.15 | 0.8527 ± 0.0254 |
| Tuned Random Forest | 2,461.78 | 4,302.56 | 0.8993 | 4,417.00 ± 596.74 | 0.8506 ± 0.0259 |
| Gradient Boosting | 2,472.53 | 4,375.20 | 0.8958 | 4,513.65 ± 600.51 | 0.8443 ± 0.0250 |
| Random Forest | 2,560.42 | 4,610.02 | 0.8843 | 4,775.17 ± 512.04 | 0.8258 ± 0.0185 |
| Linear Regression | 4,152.86 | 6,000.56 | 0.8041 | 5,982.06 ± 643.89 | 0.7270 ± 0.0253 |
| Decision Tree | 3,065.84 | 6,457.21 | 0.7731 | 6,212.50 ± 332.33 | 0.6987 ± 0.0589 |
| Dummy Regressor | 9,799.34 | 13,631.46 | -0.0112 | 11,507.50 ± 834.67 | -0.0124 ± 0.0076 |

Five-fold cross-validation was performed only on the training partition. The held-out test set was evaluated after the untuned and tuned candidates were frozen.

## Hyperparameter Tuning

Both ensemble candidates used 24-iteration `RandomizedSearchCV` with five shuffled folds and training-fold RMSE as the selection metric. Random Forest best parameters: `max_depth=5`, `max_features=0.7`, `min_samples_leaf=1`, `min_samples_split=5`, `n_estimators=350`. Gradient Boosting best parameters: `learning_rate=0.02`, `max_depth=3`, `min_samples_leaf=4`, `min_samples_split=15`, `n_estimators=200`, `subsample=0.7`.

## Final Model

**Tuned Gradient Boosting** was selected. Its held-out results are MAE **2,472.02**, RMSE **4,248.99**, and R² **0.9018**. Its training RMSE is 4,085.24, compared with 4,248.99 on test data. Its five-fold training CV RMSE is 4,386.56 ± 602.15. The selection prioritizes training-only cross-validation stability among the tuned ensemble candidates and checks that the held-out result is consistent; it does not rely on training R² alone.

## Estimated Prediction Range

Within this secondary benchmark, a **90% prediction range** is presented with the point estimate. Lower and upper Gradient Boosting quantile models learn feature-dependent bounds, which are then conformally adjusted using 214 records that were not used for model fitting or hyperparameter selection. Because error behaviour differs substantially by smoking status, conformal adjustments are calibrated separately for smokers and non-smokers: `no=-0.5453584966244307`, `yes=1470.0748738886468`. A small negative adjustment is valid and indicates that the corresponding raw quantile bounds were slightly conservative on calibration data. On the untouched test set, the interval contained the observed charge for **92.2%** of records, with mean width **21,170.72** in the source charge scale. This is an uncertainty interval under the dataset/exchangeability assumptions, not a guarantee that an individual bill will fall inside it. See `plots/prediction_intervals.svg` for held-out examples.

## Overfitting Analysis

The unconstrained Decision Tree shows clear overfitting: training R² is 1.0000 but test R² is 0.7731, with test RMSE 6,457.21. The untuned Random Forest also has a train/test gap (R² 0.9758 vs 0.8843), although averaging reduces the tree's instability. The selected tuned model constrains complexity through its searched parameters; its train/test R² values are 0.8741/0.9018, and its CV variation provides a more realistic stability check than training score.

## Feature Importance

One-hot encoded names are recovered from the fitted `ColumnTransformer`. Importance reflects predictive contribution within this model, not causality.

| Feature | Importance |
|---|---:|
| smoker_yes | 0.6752 |
| bmi | 0.1768 |
| age | 0.1295 |
| children | 0.0144 |
| region_northwest | 0.0011 |
| region_northeast | 0.0009 |
| sex_male | 0.0007 |
| region_southwest | 0.0007 |

See `plots/feature_importance.svg` for the visualization.

## Prediction Error Analysis

The top-cost quartile begins at 17,781.10. Its MAE is 4,696.70, versus 1,730.47 for the remaining observations. Therefore, high-cost observations were more difficult on this test split. The actual-vs-predicted and residual figures are in `model_training/plots/`, and `largest_prediction_errors.csv` records the worst held-out cases for discussion.

## Limitations

- This dataset is not localized to Malaysia and should not be treated as Malaysian hospital billing evidence.
- The model output remains in the scale and context of the source `charges_usd` target; it must not be relabeled as Malaysian Ringgit or arbitrarily converted.
- LIAM and Malaysian hospital pricing data are kept separate and are used only as local reference information.
- The small dataset, limited features, right-skewed target, and high-cost residuals limit generalization.
- Predictions are estimates, not guaranteed charges, quotations, medical advice, or financial advice.
