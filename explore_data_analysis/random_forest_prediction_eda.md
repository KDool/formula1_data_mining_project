# Random Forest EDA Notes for `prediction.csv`

This note summarizes the EDA and Random Forest modeling implications for
`dataset/outputs/prediction.csv`.

## Dataset and Split

`prediction.csv` covers seasons 2021-2026.

| year | rows | races |
|---:|---:|---:|
| 2021 | 440 | 22 |
| 2022 | 440 | 22 |
| 2023 | 440 | 22 |
| 2024 | 479 | 24 |
| 2025 | 479 | 24 |
| 2026 | 176 | 8 |

Recommended modeling split:

| portion | years | rows | races |
|---|---|---:|---:|
| train/development | 2021-2024 | 1,799 | 90 |
| holdout | 2025 | 479 | 24 |
| exclude | 2026 | 176 | 8 |

2026 should stay excluded because it is incomplete.

## Target Distribution

Overall target distribution:

| class | count | rate |
|---|---:|---:|
| podium | 366 | 14.91% |
| points | 854 | 34.80% |
| no_points | 1,234 | 50.29% |

Train/development target distribution is almost identical to holdout:

| split | podium | points | no_points |
|---|---:|---:|---:|
| train 2021-2024 | 15.01% | 35.02% | 49.97% |
| holdout 2025 | 15.03% | 35.07% | 49.90% |

The holdout problem is therefore not class-distribution drift. The main
challenge is separating the middle `points` class from nearby `podium` and
`no_points` cases.

## Sentinel `-1` Values

Largest sentinel rates:

| feature | `-1` rate |
|---|---:|
| `driver_wins_at_circuit` | 17.20% |
| `driver_avg_position_at_circuit` | 17.20% |
| `teammate_h2h_avg_position_delta` | 4.16% |
| `days_since_last_race` | 1.47% |
| `constructor_change_flag` | 1.47% |

For Random Forest, keeping these sentinel values is acceptable because trees can
split on them directly. Earlier experiments also showed that removing or
transforming some of these signals did not improve RF.

## Class Separability

Class medians for important features:

| feature | podium | points | no_points |
|---|---:|---:|---:|
| `grid` | 2.0 | 8.0 | 14.0 |
| `qualifying_position` | 3.0 | 8.0 | 15.0 |
| `driver_avg_position_last3` | 4.33 | 9.67 | 13.33 |
| `driver_podium_rate_last10` | 0.40 | 0.10 | 0.00 |
| `driver_points_avg_last3` | 13.00 | 4.33 | 0.33 |
| `constructor_avg_position_last3` | 5.67 | 9.50 | 13.17 |
| `points_gap_to_leader` | 48.50 | 138.75 | 192.00 |
| `driver_dnf_rate_historical` | 0.264 | 0.411 | 0.495 |

The strongest direct separation is from:

- `qualifying_position`
- `grid`
- recent driver form
- constructor recent form
- championship gap

The `points` class sits between `podium` and `no_points`, which explains why it
has the lowest F1 score.

## Correlated Feature Groups

Highly correlated pairs:

| pair | Spearman |
|---|---:|
| `driver_age` / `driver_seasons_in_f1` | 0.935 |
| `driver_std_points_prev` / `constructor_std_points_prev` | 0.932 |
| `driver_podium_rate_last5` / `driver_podium_rate_last10` | 0.908 |
| `driver_podium_rate_last3` / `driver_podium_rate_last5` | 0.907 |
| `driver_avg_position_last3` / `driver_points_avg_last3` | 0.899 |
| `grid` / `qualifying_position` | 0.858 |

Random Forest handles correlation better than SVM, but importance is split
across correlated feature families. Do not interpret Gini importance as a
perfect single-feature ranking.

## Random Forest Importance

Gini importance from the best RF trained on 2021-2024:

| rank | feature | importance |
|---:|---|---:|
| 1 | `qualifying_position` | 0.2203 |
| 2 | `grid` | 0.1466 |
| 3 | `driver_std_position_prev` | 0.0950 |
| 4 | `driver_points_avg_last3` | 0.0663 |
| 5 | `constructor_std_position_prev` | 0.0582 |
| 6 | `driver_podium_rate_last10` | 0.0482 |
| 7 | `constructor_avg_position_last3` | 0.0466 |
| 8 | `driver_dnf_rate_historical` | 0.0369 |
| 9 | `points_gap_to_leader` | 0.0334 |
| 10 | `constructor_std_points_prev` | 0.0320 |

Permutation importance on the 2025 holdout confirms the most robust features:

| rank | feature | holdout macro-F1 drop |
|---:|---|---:|
| 1 | `qualifying_position` | 0.1149 |
| 2 | `grid` | 0.0408 |
| 3 | `driver_std_position_prev` | 0.0168 |
| 4 | `constructor_std_position_prev` | 0.0156 |
| 5 | `driver_points_avg_last3` | 0.0151 |
| 6 | `points_this_season_prev` | 0.0140 |
| 7 | `teammate_h2h_avg_position_delta` | 0.0135 |
| 8 | `driver_podium_rate_last10` | 0.0120 |

Low or zero holdout permutation importance:

- `constructor_change_flag`
- `is_home_race`
- `is_home_constructor_race`
- `driver_wins_at_circuit`
- `driver_podium_rate_last3`
- `driver_podium_rate_last5`

These are candidates for controlled ablation, but previous experiments showed
that removing some low-importance flags can still hurt, so remove them only in
small groups and compare against the RF baseline.

## Best Current Random Forest

Best confirmed RF:

```json
{
  "n_estimators": 300,
  "min_samples_split": 20,
  "min_samples_leaf": 1,
  "max_samples": 0.6,
  "max_features": 0.3,
  "max_depth": null,
  "criterion": "gini",
  "bootstrap": true,
  "class_weight": "balanced"
}
```

Holdout 2025 result:

| metric | value |
|---|---:|
| accuracy | 0.7077 |
| balanced accuracy | 0.7370 |
| macro F1 | 0.7080 |
| weighted F1 | 0.7064 |

Per-class F1:

| class | F1 |
|---|---:|
| podium | 0.7619 |
| points | 0.5915 |
| no_points | 0.7706 |

## Experiments That Did Not Improve RF

The following were tested and did not beat the RF baseline:

- using longer data windows from 2014 or 2019
- resampling / oversampling
- probability threshold adjustment
- broader hyperparameter tuning with `entropy` / `log_loss`
- removing `is_home_race` and `is_home_constructor_race`

These results suggest that the current RF is already close to the best
single-model RF performance for this feature set.

## Recommendation for Random Forest

Use `prediction.csv` with:

```text
train/development: 2021-2024
holdout: 2025
exclude: 2026
```

Keep:

- `constructorId` and `circuitId` excluded as raw IDs
- `class_weight = "balanced"`
- best RF params above
- macro F1 as the main tuning metric

Do not keep resampling or threshold adjustment as the RF default. They improve
some intermediate CV scores or specific recall values, but reduce holdout macro
F1.

The next practical improvement should come from ensemble modeling rather than
more RF-only tuning.

## Constructor Recent Form Features

`prediction.csv` was regenerated from 2021 onward and now has `2454` rows and
`57` columns. The usable evaluation setup remains:

```text
train/development: 2021-2024
holdout: 2025
exclude: 2026
```

The new constructor-form columns include rolling constructor points, points
rate, podium rate, double-points rate, qualifying position, and short-vs-long
form deltas.

Quick RF check using the old best hyperparameters plus all new constructor
features:

```text
holdout macro F1: 0.6762
old feature-set baseline: 0.7080
```

So these features are informative, but adding all of them without retuning is
not automatically better. They should be tested as a separate RF optimization
experiment.

Top feature importances from the quick check:

```text
qualifying_position                               0.2111
grid                                              0.0978
constructor_points_avg_last10                     0.0702
constructor_points_avg_last5                      0.0687
constructor_avg_position_last10                   0.0414
constructor_avg_qualifying_position_last10        0.0387
constructor_points_avg_last3                      0.0364
constructor_avg_qualifying_position_last5         0.0351
driver_dnf_rate_historical                        0.0243
driver_std_position_prev                          0.0232
teammate_h2h_avg_position_delta                   0.0224
constructor_avg_position_last5                    0.0207
points_gap_to_leader                              0.0203
driver_points_avg_last3                           0.0188
constructor_avg_qualifying_position_last3         0.0179
driver_avg_position_last3                         0.0177
constructor_qualifying_delta_last3_vs_last10      0.0168
constructor_avg_position_delta_last3_vs_last10    0.0157
driver_age                                        0.0155
constructor_points_avg_delta_last3_vs_last10      0.0147
```

Selected class medians:

```text
target     constructor_points_avg_last3  constructor_points_avg_last5  constructor_points_avg_last10  constructor_points_rate_last3  constructor_podium_rate_last3  constructor_avg_qualifying_position_last3
no_points                         1.667                           2.0                            2.0                          0.333                            0.0                                     13.333
podium                           23.333                          24.1                           23.0                          0.833                            0.5                                      5.083
points                           10.000                           9.3                            8.8                          0.667                            0.0                                      9.167
```

Interpretation:

- Constructor long-window form has clear signal. The strongest new features are
  `constructor_points_avg_last10`, `constructor_points_avg_last5`,
  `constructor_avg_position_last10`, and constructor qualifying form.
- The main risk is redundancy with `grid`, `qualifying_position`, and existing
  constructor strength features.
- The RF config now writes this experiment to
  `json-parameters/random-forest/random_forest_best_constructor_features_params.json`
  so it can be compared against the old best without overwriting it.

After the full constructor-feature run underperformed, the next RF config was
reduced to the old baseline plus only the strongest constructor-form features:

```text
constructor_points_avg_last10
constructor_points_avg_last5
constructor_avg_position_last10
constructor_avg_qualifying_position_last10
constructor_avg_qualifying_position_last5
```

This selected-feature experiment writes to:

```text
json-parameters/random-forest/random_forest_best_selected_constructor_features_params.json
```
