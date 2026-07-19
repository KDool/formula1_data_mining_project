# EDA Explanation for `prediction_2014.csv`

This document explains the exploratory analysis in `f1_eda_2014.ipynb` for
`dataset/outputs/prediction_2014.csv`.

## Dataset Overview

The 2014 dataset has 5,281 rows and 38 columns, covering seasons 2014-2026.
It is much larger than the previous dataset that started from 2021, so it gives
the models more historical examples but also introduces more cold-start and
regulation-era variation.

Target distribution:

| class | count | rate |
|---|---:|---:|
| podium | 780 | 14.77% |
| points | 1,820 | 34.46% |
| no_points | 2,681 | 50.77% |

The class imbalance remains clear. `no_points` is the majority class, while
`podium` is the smallest class. Accuracy alone is therefore not enough; macro
F1, balanced accuracy, and per-class F1 are still the most important metrics.

## Generated Figures

The notebook writes figures with the `eda2014_` prefix so they do not overwrite
the original EDA outputs.

- `eda2014_01_target_distribution.png`
- `eda2014_02_target_by_year.png`
- `eda2014_03_feature_distributions.png`
- `eda2014_04_separability_driver_avg_position_last3.png`
- `eda2014_04_separability_driver_avg_position_last5.png`
- `eda2014_04_separability_driver_avg_position_last10.png`
- `eda2014_04_separability_teammate_h2h_avg_position_delta.png`
- `eda2014_05_sentinel_minus_one_rates.png`
- `eda2014_06_correlation_heatmap.png`
- `eda2014_07_feature_importance.png`

## Target by Year

The target mix is fairly stable by season. Most full seasons are close to:

- podium: about 15%
- points: about 35%
- no_points: about 50%

Some shorter or unusual seasons show small deviations. This means the target
definition is consistent across the longer 2014-2026 period, which is useful
for training. However, older seasons may still differ because of regulation,
team strength, and calendar changes.

## Sentinel `-1` Values

There are no ordinary NaN missing values, but several features use `-1` as a
sentinel value.

The largest sentinel rates are:

| feature | `-1` rate |
|---|---:|
| `driver_avg_position_at_circuit` | 20.02% |
| `driver_wins_at_circuit` | 20.02% |
| `teammate_h2h_avg_position_delta` | 4.96% |
| `days_since_last_race` | 1.19% |
| `constructor_change_flag` | 1.19% |

This matters most for SVM and other distance-based models. Treating `-1` as a
normal numeric value can distort scaling and distances. The SVM preprocessing
idea used earlier remains relevant: create missing indicators and impute
sentinel values with train-fold medians.

For tree models, `-1` is less dangerous, but it still encodes a special state.
Random Forest and Gradient Boosting may learn that state directly.

## Feature Distributions

The key numeric features show the same broad structure as the smaller dataset:

- `grid` and `qualifying_position` are strong race-position signals.
- `points_gap_to_leader` is highly skewed and should be scaled for SVM.
- rolling form features such as `driver_avg_position_last3` contain clear
  performance information.
- `teammate_h2h_avg_position_delta` is centered near zero but has useful
  directional signal.
- `driver_dnf_rate_historical` includes a sentinel `-1` and otherwise ranges
  between 0 and 1.

One important difference is that the longer dataset includes more historical
edge cases, such as large gaps in `days_since_last_race` and more circuit
cold-start rows.

## Feature Separability

The separability plots check whether individual features can distinguish:

- `podium`
- `points`
- `no_points`

Rolling average position features remain highly useful. Lower average position
usually corresponds to better target classes. The three target classes still
overlap, but the median ordering is meaningful:

`podium` tends to have the best recent average positions, `points` is in the
middle, and `no_points` has worse recent positions.

`teammate_h2h_avg_position_delta` has weaker standalone separation. It is more
useful when combined with constructor strength, qualifying, grid, and recent
form.

## Correlation

The Spearman heatmap should show strong correlation among related feature
families:

- `grid` and `qualifying_position`
- rolling `last3`, `last5`, and `last10` form features
- points average, podium rate, and average position features
- driver and constructor historical strength features

This is acceptable for Random Forest and Gradient Boosting, but SVM can be more
sensitive to redundant dimensions. For SVM, compact feature selection and
scaling remain important.

## Baseline Random Forest Importance

The top baseline Random Forest features are:

| rank | feature | importance |
|---:|---|---:|
| 1 | `grid` | 0.1125 |
| 2 | `qualifying_position` | 0.1117 |
| 3 | `driver_points_avg_last10` | 0.0756 |
| 4 | `driver_points_avg_last5` | 0.0524 |
| 5 | `constructor_avg_position_last5` | 0.0510 |
| 6 | `driver_avg_position_last10` | 0.0447 |
| 7 | `driver_std_position_prev` | 0.0438 |
| 8 | `constructor_avg_position_last3` | 0.0370 |
| 9 | `driver_dnf_rate_historical` | 0.0355 |
| 10 | `driver_points_avg_last3` | 0.0350 |

The result is consistent with Formula 1 domain knowledge. Starting position,
qualifying, recent driver form, constructor form, reliability, and standings
context are the main signals.

## Modeling Implications

The 2014 dataset is promising because it gives substantially more training
data. It may improve model stability, especially for tree-based models.

Recommended next steps:

- Re-run Random Forest, SVM, Gradient Boosting, and ensemble notebooks using
  `prediction_2014.csv`.
- Keep the holdout split time-based, for example train/dev on 2014-2024 and
  holdout on 2025, excluding 2026 if it is incomplete.
- For SVM, keep sentinel handling and scaling.
- For Random Forest and Gradient Boosting, compare whether the longer history
  improves macro F1 or introduces older-season noise.
- Compare per-class F1 carefully, especially `points`, because it remains the
  hardest middle class.

