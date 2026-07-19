# EDA and Optimization Notes for 2019-2025

This document summarizes the EDA for `dataset/outputs/prediction_2014.csv`
filtered to seasons 2019-2025, and interprets the latest Random Forest
optimization result.

## Dataset Scope

The filtered dataset has 3,038 rows and 38 columns.

| year | rows | races |
|---:|---:|---:|
| 2019 | 420 | 21 |
| 2020 | 340 | 17 |
| 2021 | 440 | 22 |
| 2022 | 440 | 22 |
| 2023 | 440 | 22 |
| 2024 | 479 | 24 |
| 2025 | 479 | 24 |

Current modeling split:

| portion | years | rows | races |
|---|---|---:|---:|
| train/development | 2019-2024 | 2,559 | 128 |
| holdout | 2025 | 479 | 24 |

This is a reasonable time-based split: about 84.2% train/development and 15.8%
holdout.

## Target Distribution

Target distribution for 2019-2025:

| class | count | rate |
|---|---:|---:|
| podium | 456 | 15.01% |
| points | 1,064 | 35.02% |
| no_points | 1,518 | 49.97% |

The class balance is very stable across 2019-2025:

- `podium`: about 15%
- `points`: about 35%
- `no_points`: about 50%

This means the performance drop is probably not caused by a class distribution
shift between train and holdout. The harder issue is feature relationship shift:
the same feature values may map to outcomes differently across recent seasons.

## Sentinel `-1` Values

The largest sentinel rates are:

| feature | `-1` rate |
|---|---:|
| `driver_avg_position_at_circuit` | 19.68% |
| `driver_wins_at_circuit` | 19.68% |
| `teammate_h2h_avg_position_delta` | 4.64% |
| `driver_std_position_prev` | 0.89% |
| `days_since_last_race` | 0.69% |
| `constructor_change_flag` | 0.69% |

For Random Forest, these sentinel values can be learned as a special branch.
For SVM, they should be handled with missing indicators and median imputation.

## Baseline Feature Importance

A baseline Random Forest trained on 2019-2024 gives these top features:

| rank | feature | importance |
|---:|---|---:|
| 1 | `qualifying_position` | 0.1346 |
| 2 | `grid` | 0.1180 |
| 3 | `driver_points_avg_last3` | 0.0594 |
| 4 | `driver_std_position_prev` | 0.0587 |
| 5 | `constructor_avg_position_last3` | 0.0536 |
| 6 | `constructor_std_position_prev` | 0.0534 |
| 7 | `driver_dnf_rate_historical` | 0.0512 |
| 8 | `driver_podium_rate_last10` | 0.0498 |
| 9 | `constructor_std_points_prev` | 0.0460 |
| 10 | `points_gap_to_leader` | 0.0445 |

The model relies most on starting/qualifying position, recent driver form,
constructor strength, reliability, and championship context. This is consistent
with the earlier EDA.

## Latest Random Forest Result

Latest RF setup:

- dataset: `prediction_2014.csv`
- filter: `start_year = 2019`
- train/development: 2019-2024
- holdout: 2025
- excluded: 2026

Nested CV:

| metric | value |
|---|---:|
| mean macro F1 | 0.6917 |
| std macro F1 | 0.0155 |
| mean balanced accuracy | 0.7110 |

Holdout:

| metric | value |
|---|---:|
| accuracy | 0.6764 |
| balanced accuracy | 0.7120 |
| macro F1 | 0.6767 |
| weighted F1 | 0.6751 |

Per-class holdout result:

| class | precision | recall | F1 |
|---|---:|---:|---:|
| podium | 0.6275 | 0.8889 | 0.7356 |
| points | 0.5625 | 0.5357 | 0.5488 |
| no_points | 0.7834 | 0.7113 | 0.7456 |

Confusion matrix:

| true / pred | podium | points | no_points |
|---|---:|---:|---:|
| podium | 64 | 7 | 1 |
| points | 32 | 90 | 46 |
| no_points | 6 | 63 | 170 |

## Interpretation

The 2019 model is not better than the previous 2021-start RF model.

| setup | holdout | macro F1 | points F1 |
|---|---|---:|---:|
| train 2021-2024 | 2025 | 0.7080 | 0.5915 |
| train 2019-2024 | 2025 | 0.6767 | 0.5488 |
| train 2014-2023 | 2024-2025 | 0.6747 | 0.5622 |

The main problem is not recall for `podium`; it is too high. The model predicts
`podium` too easily:

- `points -> podium`: 32 rows
- `no_points -> points`: 63 rows
- `points -> no_points`: 46 rows

This lowers precision for `podium` and hurts the middle `points` class.

## What to Optimize Next

### 1. Return to a More Recent Training Window

The best current RF result still comes from 2021-2024 training. The 2019 and
2014 windows add more rows, but they also add older-season noise.

Recommended next window:

```text
train/development: 2020-2024
holdout: 2025
exclude: 2026
```

This tests whether 2020 adds useful extra data without pulling in too much
older regime noise.

### 2. Tune Class Weight, Not Only Hyperparameters

The current RF uses `class_weight = "balanced"`. This may be pushing the model
to over-detect `podium`, because `podium` is the smallest class.

Try custom class weights:

```json
[
  {"podium": 1.0, "points": 1.0, "no_points": 1.0},
  {"podium": 0.8, "points": 1.1, "no_points": 1.0},
  {"podium": 0.7, "points": 1.2, "no_points": 1.0},
  {"podium": 0.9, "points": 1.15, "no_points": 1.0}
]
```

Goal: reduce false `podium` predictions and improve `points` F1.

### 3. Fine-Tune Around the Old Best RF

The 2019 run selected:

```json
{
  "n_estimators": 800,
  "max_depth": 8,
  "min_samples_leaf": 4,
  "max_features": "log2",
  "criterion": "log_loss",
  "max_samples": 0.9
}
```

But the best 2021-start RF used:

```json
{
  "n_estimators": 300,
  "max_depth": null,
  "min_samples_split": 20,
  "min_samples_leaf": 1,
  "max_features": 0.3,
  "criterion": "gini",
  "max_samples": 0.6
}
```

The older best is more bagged and more regularized through `max_samples=0.6`
and `min_samples_split=20`. For current data, a focused search around that
region is more promising than the broad search.

### 4. Consider Removing Older-Sensitive Features

Features such as `driver_age`, `driver_seasons_in_f1`, and some historical
circuit fields may behave differently across longer time windows. If the 2020
window still underperforms, test excluding:

```text
driver_age
driver_seasons_in_f1
driver_wins_at_circuit
driver_avg_position_at_circuit
```

Keep this as a controlled ablation, not all at once.

## Recommendation

Do not keep the 2019-start RF as the best model. Use it as evidence that adding
older data is not automatically beneficial.

The most useful next experiment is:

```text
dataset: prediction_2014.csv
start_year: 2020
train/development: 2020-2024
holdout: 2025
class_weight search: include custom weights that reduce podium pressure
```

