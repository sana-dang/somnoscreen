## CV results by approach (5 folds x 5 repeats, training set, mean ± SD)

| Model | Imbalance | ROC-AUC A | ROC-AUC B | ROC-AUC C | PR-AUC A | PR-AUC B | PR-AUC C |
|---|---|---|---|---|---|---|---|
| Logistic Regression | class_weight | 0.757 ± 0.015 | 0.757 ± 0.015 | 0.729 ± 0.017 | 0.547 ± 0.018 | 0.547 ± 0.018 | 0.497 ± 0.030 |
| Random Forest | class_weight | 0.750 ± 0.019 | 0.749 ± 0.017 | 0.733 ± 0.019 | 0.541 ± 0.025 | 0.538 ± 0.022 | 0.517 ± 0.030 |
| XGBoost | class_weight | 0.752 ± 0.015 | 0.752 ± 0.016 | 0.720 ± 0.020 | 0.543 ± 0.020 | 0.544 ± 0.019 | 0.494 ± 0.022 |
| SVM | class_weight | 0.751 ± 0.014 | 0.750 ± 0.014 | 0.741 ± 0.014 | 0.523 ± 0.027 | 0.524 ± 0.026 | 0.522 ± 0.028 |
| LightGBM | class_weight | 0.750 ± 0.017 | 0.749 ± 0.017 | 0.711 ± 0.018 | 0.540 ± 0.023 | 0.538 ± 0.024 | 0.484 ± 0.023 |
| Gradient Boosting | class_weight | 0.752 ± 0.016 | 0.751 ± 0.016 | 0.715 ± 0.022 | 0.544 ± 0.022 | 0.543 ± 0.022 | 0.489 ± 0.024 |
| KNN | smote | 0.703 ± 0.015 | 0.702 ± 0.016 | 0.661 ± 0.019 | 0.461 ± 0.024 | 0.460 ± 0.025 | 0.400 ± 0.027 |
| Majority-class baseline | - | 0.500 |  |  | 0.268 |  |  |


## Imbalance strategies, Approach A (CV PR-AUC, mean ± SD)

| Model | SMOTE | SMOTENC | class_weight='balanced' |
|---|---|---|---|
| Logistic Regression | 0.545 ± 0.019 | 0.527 ± 0.027 | 0.547 ± 0.018 |
| Random Forest | 0.524 ± 0.027 | 0.512 ± 0.028 | 0.541 ± 0.025 |
| XGBoost | 0.541 ± 0.024 | 0.524 ± 0.025 | 0.543 ± 0.020 |
| SVM | 0.499 ± 0.031 | 0.488 ± 0.035 | 0.523 ± 0.027 |
| LightGBM | 0.543 ± 0.023 | 0.524 ± 0.025 | 0.540 ± 0.023 |
| Gradient Boosting | 0.541 ± 0.023 | 0.522 ± 0.025 | 0.544 ± 0.022 |
| KNN | 0.461 ± 0.024 | 0.446 ± 0.026 | n/a |
| **Mean of 6 models (no KNN)** | **0.532** | **0.516** | **0.540** |


## Is clustering better? (number of the 7 models)

| Comparison | CV corrected t-test: better / worse / no difference | Test bootstrap: better / worse / no difference |
|---|---|---|
| B vs A, ROC-AUC | 0 / 0 / 7 | 0 / 0 / 7 |
| B vs A, PR-AUC | 0 / 0 / 7 | 0 / 0 / 7 |
| C vs A, ROC-AUC | 0 / 7 / 0 | 0 / 6 / 1 |
| C vs A, PR-AUC | 0 / 6 / 1 | 0 / 5 / 2 |


## Calibration (final model, 5x5 CV)

| Calibration | Brier (lower is better) | ROC-AUC | PR-AUC |
|---|---|---|---|
| none | 0.199 ± 0.005 | 0.757 ± 0.015 | 0.547 ± 0.018 |
| sigmoid | 0.163 ± 0.003 | 0.757 ± 0.015 | 0.547 ± 0.018 |
| isotonic | 0.163 ± 0.004 | 0.756 ± 0.014 | 0.527 ± 0.017 |


## Final model on the held-out test set (n=1165), threshold 0.246

| Metric | Test set (at screening threshold) | CV 5x5 (training) | Majority baseline (test) |
|---|---|---|---|
| ROC-AUC | 0.758 (95% CI 0.727-0.790) | 0.757 ± 0.015 | 0.500 |
| PR-AUC | 0.530 (95% CI 0.470-0.593) | 0.547 ± 0.018 | 0.268 |
| Brier score | 0.163 | 0.163 ± 0.003 | 0.196 |
| Recall (sensitivity) | 0.689 |  | 0.000 |
| Precision | 0.427 | 0.453 ± 0.026 (at 70% recall) | 0.000 |
| F1 | 0.528 |  | 0.000 |
| Balanced accuracy | 0.676 |  | 0.500 |
| Accuracy | 0.670 |  | 0.732 |


## Fairness on the test set at the screening threshold

| Group | N | Share with sleep trouble | Share flagged | Recall | Precision |
|---|---|---|---|---|---|
| Male | 541 | 0.240 | 0.388 | 0.662 | 0.410 |
| Female | 624 | 0.292 | 0.470 | 0.709 | 0.440 |
| 40 and above | 724 | 0.336 | 0.546 | 0.733 | 0.451 |
| Under 40 | 441 | 0.156 | 0.245 | 0.536 | 0.343 |
