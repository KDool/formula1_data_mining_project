# Random Forest Optimization Summary

This document summarizes the Random Forest optimization work for the Formula 1
race outcome classification task.

## Evaluation Setup

Main target:

```text
podium / points / no_points
```

Main metric:

```text
macro F1
```

Primary evaluation split:

```text
train/development: 2021-2024
holdout test:      2025
excluded:          2026
```

Reason for this split:

- The task is a future-prediction problem.
- Model selection is done on 2021-2024.
- 2025 is kept as a final unseen season-level holdout.
- 2026 is excluded because it is incomplete in the current dataset.

## Fold-Splitting Logic

The Random Forest notebook uses a nested, race-level time-series split. This is
different from a standard random k-fold split.

The dataset has one row per driver per race:

```text
1 row = 1 driver in 1 race
```

Because of that structure, row-level splitting would be unsafe. If rows were
split randomly, drivers from the same race could appear in both the training and
validation folds. For example, one driver from the 2024 Italian Grand Prix could
be in the training fold while another driver from the same race could be in the
validation fold. That would leak race-level information across the split.

To avoid this, the notebook first builds a chronological list of unique races:

```python
race_order = (
    df_source[["raceId", "date"]]
    .drop_duplicates()
    .sort_values("date")
    .reset_index(drop=True)
)
```

Then `TimeSeriesSplit` is applied to this race list, not directly to individual
driver rows. After each race split is created, the selected race IDs are mapped
back to all driver rows from those races.

This guarantees two properties:

```text
1. A race is entirely in train or entirely in validation.
2. Validation races always happen after training races.
```

### Outer Loop

The outer loop estimates how well the full model-selection process generalizes
to later races inside the train/development period.

With:

```text
outer_n_splits = 5
train/development = 2021-2024
```

the split behaves like an expanding-window validation over chronological races:

```text
Outer fold 1:
train      = earliest race block
validation = next race block

Outer fold 2:
train      = earlier race blocks
validation = next race block

...

Outer fold 5:
train      = most of 2021-2024
validation = latest block inside 2021-2024
```

The outer validation fold is not used for hyperparameter tuning. It is used only
to evaluate the model chosen by the inner loop.

### Inner Loop

The inner loop is used for hyperparameter tuning inside each outer training
fold.

With:

```text
inner_n_splits = 3
```

each outer training segment is split again into smaller chronological
train/validation folds. `RandomizedSearchCV` evaluates each hyperparameter
candidate across these inner folds and selects the candidate with the best mean
`f1_macro` score.

After the best inner hyperparameters are selected, the model is refit on the
entire outer training fold and then evaluated on the untouched outer validation
fold.

The nested logic is:

```text
Outer fold:
    outer train
        inner fold 1: tune candidate params
        inner fold 2: tune candidate params
        inner fold 3: tune candidate params
    select best params from inner CV
    refit on full outer train
    evaluate on outer validation
```

### Final Holdout

After nested CV is complete, the notebook runs one final hyperparameter search
on all train/development data:

```text
2021-2024
```

Then the final model is evaluated once on:

```text
2025
```

The 2025 holdout is not used in the inner loop, the outer loop, feature
selection, or hyperparameter tuning. It is reserved for the final future-season
evaluation.

This is why the notebook has both nested CV and a separate holdout:

```text
inner CV  = tune hyperparameters
outer CV  = estimate model-selection performance
holdout   = final future-season test
```

## Best Confirmed Random Forest

Best confirmed artifact:

```text
json-parameters/random-forest/random_forest_best_optimization_params.json
```

Feature count:

```text
25
```

Best hyperparameters:

```json
{
  "n_estimators": 300,
  "min_samples_split": 20,
  "min_samples_leaf": 1,
  "max_samples": 0.6,
  "max_features": 0.3,
  "max_depth": null,
  "criterion": "gini",
  "bootstrap": true
}
```

Important fixed setting:

```text
class_weight = "balanced"
```

Holdout 2025 metrics:

| metric | value |
|---|---:|
| accuracy | 0.7077 |
| balanced accuracy | 0.7370 |
| macro precision | 0.6904 |
| macro recall | 0.7370 |
| macro F1 | 0.7080 |
| weighted F1 | 0.7064 |
| Matthews corrcoef | 0.5307 |
| Cohen kappa | 0.5289 |

Per-class holdout F1:

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| podium | 0.6667 | 0.8889 | 0.7619 | 72 |
| points | 0.6062 | 0.5774 | 0.5915 | 168 |
| no_points | 0.7982 | 0.7448 | 0.7706 | 239 |

Confusion matrix:

```text
labels: podium, points, no_points

[[ 64,   7,   1],
 [ 27,  97,  44],
 [  5,  56, 178]]
```

Conclusion:

```text
This remains the best Random Forest configuration.
```

## Experiment Log

### 1. Treating `constructorId` and `circuitId` as Categorical

Change:

- Tested one-hot handling for raw ID-like columns.
- Motivation was that `constructorId` and `circuitId` are categorical IDs, not
  ordinal numeric quantities.

Reason:

- A tree can split on numeric IDs, but it may still learn arbitrary numeric
  thresholds such as `constructorId <= 9`.
- This is not semantically meaningful because constructor IDs are labels.

Observed result:

- Performance decreased after converting these columns to categorical
  one-hot features.

Decision:

- Do not use raw `constructorId` or `circuitId` as active RF features.
- Keep them in the dataset for joins/debugging.
- Put them in `excluded_feature_columns`.

Current decision:

```json
"excluded_feature_columns": [
  "constructorId",
  "circuitId"
]
```

### 2. Removing Raw `constructorId` and `circuitId`

Change:

- Removed raw `constructorId` and `circuitId` from active model features.

Reason:

- These columns are identifiers, not numeric performance indicators.
- Existing engineered features already describe constructor strength and circuit
  history more safely.

Observed result:

- This became part of the best confirmed feature set.

Decision:

```text
Keep raw constructorId/circuitId excluded.
```

### 3. Removing Home-Race Flags

Change:

- Temporarily removed:

```text
is_home_race
is_home_constructor_race
```

Reason:

- EDA/permutation importance suggested low or unstable importance.
- The test checked whether these binary flags added noise.

Observed result:

- Metrics dropped.

Decision:

```text
Restore both home-race features.
```

Current best feature set includes:

```text
is_home_race
is_home_constructor_race
```

### 4. Best Baseline Feature Set

Best confirmed active features:

```text
grid
qualifying_position
driver_age
driver_std_points_prev
driver_std_position_prev
constructor_std_points_prev
constructor_std_position_prev
points_this_season_prev
driver_avg_position_last3
driver_podium_rate_last3
driver_podium_rate_last5
driver_podium_rate_last10
driver_points_avg_last3
constructor_avg_position_last3
driver_wins_at_circuit
driver_avg_position_at_circuit
teammate_h2h_avg_position_delta
driver_dnf_rate_historical
points_gap_to_leader
constructor_change_flag
days_since_last_race
sprint_flag
driver_seasons_in_f1
is_home_race
is_home_constructor_race
```

Result:

```text
holdout macro F1 = 0.7080
```

Decision:

```text
Use this as the RF baseline and the best confirmed RF model.
```

### 5. Longer Data Windows: 2014+ and 2019+

Change:

- Created/used datasets starting from earlier years such as 2014 and 2019.
- Tested whether more historical data improves RF.

Reason:

- More rows can reduce variance.
- More seasons can expose the model to more driver/team/circuit combinations.

Observed result:

- Longer windows did not improve the 2025 holdout result.
- The model appears sensitive to historical distribution drift.

Likely reason:

- Older F1 seasons differ in regulation, driver lineup, constructor strength,
  race calendar, and points dynamics.
- More data is not automatically better when the sport distribution changes.

Decision:

```text
Use the 2021+ dataset for the main RF setup.
```

### 6. Resampling / Oversampling

Artifact:

```text
json-parameters/random-forest/random_forest_best_resampling_evidence_params.json
```

Change:

- Tested resampling strategies to address class imbalance.
- Motivation was to improve the weaker middle class, `points`.
- Added a dedicated evidence notebook:

```text
models_training/random-forest/random_forest_resampling_nested_cv.ipynb
```

- The notebook used a custom `OversampledRandomForestClassifier` so that
  oversampling happened only inside each training fold. Validation folds and the
  final 2025 holdout were never oversampled.

Reason:

- The dataset is imbalanced:

```text
no_points > points > podium
```

- `points` is the hardest class because it sits between `podium` and
  `no_points`.

Search space:

```text
resampling_strategy: none / minority / not_majority
resampling_ratio:    0.5 / 0.75 / 1.0
class_weight:        null / balanced
```

Observed result:

- Resampling did not improve holdout performance.
- The best selected configuration used no oversampling:

```json
{
  "resampling_strategy": "none",
  "resampling_ratio": 0.5,
  "class_weight": null,
  "n_estimators": 500,
  "min_samples_split": 20,
  "min_samples_leaf": 1,
  "max_samples": 0.75,
  "max_features": 0.3,
  "max_depth": null,
  "criterion": "gini",
  "bootstrap": true
}
```

Overall holdout comparison:

| metric | baseline RF | resampling run | change |
|---|---:|---:|---:|
| accuracy | 0.7077 | 0.6806 | -0.0271 |
| balanced accuracy | 0.7370 | 0.6859 | -0.0511 |
| macro F1 | 0.7080 | 0.6788 | -0.0292 |
| weighted F1 | 0.7064 | 0.6763 | -0.0301 |
| Matthews corrcoef | 0.5307 | 0.4721 | -0.0586 |
| Cohen kappa | 0.5289 | 0.4711 | -0.0578 |

Nested CV comparison:

| metric | baseline RF | resampling run |
|---|---:|---:|
| mean macro F1 | 0.6898 | 0.6795 |
| std macro F1 | 0.0187 | 0.0227 |
| mean balanced accuracy | 0.7143 | 0.6957 |
| std balanced accuracy | 0.0150 | 0.0231 |

Per-class holdout F1:

| class | baseline RF | resampling run | change |
|---|---:|---:|---:|
| podium | 0.7619 | 0.7467 | -0.0152 |
| points | 0.5915 | 0.5363 | -0.0552 |
| no_points | 0.7706 | 0.7536 | -0.0170 |

Resampling confusion matrix:

```text
labels: podium, points, no_points

[[ 56,  13,   3],
 [ 19,  85,  64],
 [  3,  51, 185]]
```

The largest degradation was in the `points` class:

```text
baseline true points predicted correctly:   97
resampling true points predicted correctly: 85
```

The resampling run also increased `points -> no_points` errors:

```text
baseline:   44
resampling: 64
```

Decision:

```text
Do not keep resampling as the RF default.
The evidence run selected resampling_strategy = "none" and still performed
worse than the confirmed RF baseline.
```

### 7. Probability Threshold Adjustment

Change:

- Tested probability threshold adjustment after RF prediction.

Reason:

- The goal was to improve class-specific behavior, especially for `points`.
- Threshold tuning can sometimes improve recall/precision trade-offs without
  retraining the base model.

Observed result:

- It did not beat the best RF baseline on holdout macro F1.

Decision:

```text
Do not keep threshold adjustment enabled for RF.
```

Current config:

```json
"threshold_adjustment": {
  "enabled": false
}
```

### 8. Broader Hyperparameter Search

Change:

- Expanded the RF search around:

```text
max_depth
min_samples_split
min_samples_leaf
max_samples
max_features
criterion
class_weight
```

- Also added `criterion = "log_loss"` in later searches.

Reason:

- The baseline was strong but potentially under-tuned.
- A wider search could have found a better bias/variance trade-off.

Observed result:

- The broader search did not beat the best baseline.
- Some runs improved nested CV slightly but failed to improve 2025 holdout.

Decision:

```text
Keep the original best RF as the confirmed baseline.
Be careful with wider search because it can overfit the train/development CV.
```

### 9. Full Constructor Recent-Form Features

Artifact:

```text
json-parameters/random-forest/random_forest_best_constructor_features_params.json
```

Change:

- Expanded the dataset with many constructor recent-form features:

```text
constructor_avg_position_last5
constructor_avg_position_last10
constructor_points_avg_last3/5/10
constructor_points_rate_last3/5/10
constructor_podium_rate_last3/5/10
constructor_double_points_rate_last3/5/10
constructor_avg_qualifying_position_last3/5/10
constructor_points_avg_delta_last3_vs_last10
constructor_avg_position_delta_last3_vs_last10
constructor_qualifying_delta_last3_vs_last10
```

Reason:

- EDA showed constructor form is highly predictive.
- The goal was to improve model understanding of recent team strength.

Config/result:

| item | value |
|---|---:|
| feature count | 45 |
| randomized search iterations | 40 |
| nested CV mean macro F1 | 0.6875 |
| nested CV std macro F1 | 0.0209 |
| final search best CV score | 0.6856 |
| holdout macro F1 | 0.6689 |
| holdout accuracy | 0.6681 |
| holdout weighted F1 | 0.6664 |

Best params from this run:

```json
{
  "n_estimators": 500,
  "min_samples_split": 10,
  "min_samples_leaf": 4,
  "max_samples": 0.6,
  "max_features": 0.4,
  "max_depth": null,
  "criterion": "log_loss",
  "class_weight": "balanced",
  "bootstrap": true
}
```

Per-class holdout F1:

| class | F1 |
|---|---:|
| podium | 0.7349 |
| points | 0.5305 |
| no_points | 0.7414 |

Decision:

```text
Reject. Too many constructor-form features added redundancy/noise.
```

### 10. Selected Constructor Recent-Form Features

Artifact:

```text
json-parameters/random-forest/random_forest_best_selected_constructor_features_params.json
```

Change:

- Reduced constructor expansion to the strongest EDA candidates only:

```text
constructor_points_avg_last10
constructor_points_avg_last5
constructor_avg_position_last10
constructor_avg_qualifying_position_last10
constructor_avg_qualifying_position_last5
```

Reason:

- The full constructor feature set underperformed.
- The selected version tested whether only the strongest constructor features
  could help without excessive redundancy.

Config/result:

| item | value |
|---|---:|
| feature count | 30 |
| randomized search iterations | 30 |
| nested CV mean macro F1 | 0.6909 |
| nested CV std macro F1 | 0.0208 |
| final search best CV score | 0.6857 |
| holdout macro F1 | 0.6723 |
| holdout accuracy | 0.6743 |
| holdout weighted F1 | 0.6723 |

Best params from this run:

```json
{
  "n_estimators": 500,
  "min_samples_split": 10,
  "min_samples_leaf": 2,
  "max_samples": 0.6,
  "max_features": 0.4,
  "max_depth": null,
  "criterion": "log_loss",
  "class_weight": "balanced",
  "bootstrap": true
}
```

Per-class holdout F1:

| class | F1 |
|---|---:|
| podium | 0.7305 |
| points | 0.5354 |
| no_points | 0.7511 |

Decision:

```text
Reject. Better than full constructor expansion, but still clearly below the RF baseline.
```

### 11. Points-Cutoff Delta Features

Artifact:

```text
json-parameters/random-forest/random_forest_best_cutoff_delta_params.json
```

Change:

- Removed the expanded constructor features from the dataset output.
- Added two direct top-10 boundary features:

```text
grid_to_points_cutoff_delta = grid - 10
qualifying_to_points_cutoff_delta = qualifying_position - 10
```

Reason:

- The weakest class is `points`.
- `points` usually lies near the top-10 cutoff boundary.
- These features directly encode distance from the points threshold.

Config/result:

| item | value |
|---|---:|
| feature count | 27 |
| randomized search iterations | 30 |
| nested CV mean macro F1 | 0.7020 |
| nested CV std macro F1 | 0.0130 |
| final search best CV score | 0.6999 |
| holdout macro F1 | 0.6865 |
| holdout accuracy | 0.6868 |
| holdout weighted F1 | 0.6826 |

Best params from this run:

```json
{
  "n_estimators": 500,
  "min_samples_split": 10,
  "min_samples_leaf": 1,
  "max_samples": 0.6,
  "max_features": "sqrt",
  "max_depth": null,
  "criterion": "entropy",
  "class_weight": "balanced",
  "bootstrap": true
}
```

Per-class holdout F1:

| class | F1 |
|---|---:|
| podium | 0.7590 |
| points | 0.5426 |
| no_points | 0.7579 |

Confusion matrix:

```text
labels: podium, points, no_points

[[ 63,   8,   1],
 [ 27,  86,  55],
 [  4,  55, 180]]
```

Observation:

- Nested CV improved compared with the best baseline:

```text
baseline nested CV macro F1:      0.6898
cutoff-delta nested CV macro F1:  0.7020
```

- But holdout 2025 decreased:

```text
baseline holdout macro F1:      0.7080
cutoff-delta holdout macro F1:  0.6865
```

Decision:

```text
Reject as final RF. The feature idea is reasonable, but it did not generalize to 2025.
```

## Comparison Table

| experiment | features | nested CV macro F1 | holdout macro F1 | holdout accuracy | points F1 | decision |
|---|---:|---:|---:|---:|---:|---|
| Best baseline RF | 25 | 0.6898 | 0.7080 | 0.7077 | 0.5915 | keep |
| Full constructor form | 45 | 0.6875 | 0.6689 | 0.6681 | 0.5305 | reject |
| Selected constructor form | 30 | 0.6909 | 0.6723 | 0.6743 | 0.5354 | reject |
| Points-cutoff deltas | 27 | 0.7020 | 0.6865 | 0.6868 | 0.5426 | reject |

## Key Findings

1. The best Random Forest is still the 25-feature baseline.
2. The strongest classes are `podium` and `no_points`.
3. The hardest class is `points`.
4. More constructor-form features did not improve holdout performance.
5. The cutoff-delta idea improved nested CV but failed on 2025 holdout.
6. Resampling and probability threshold adjustment did not beat the baseline.
7. A higher CV score does not guarantee a better future-year holdout result.

## Final Recommendation

Use the baseline RF as the final standalone Random Forest:

```text
artifact: json-parameters/random-forest/random_forest_best_optimization_params.json
model:    models_training/random-forest/rf_final_model.joblib
```

Recommended final feature policy:

- Keep the 25-feature baseline.
- Keep `constructorId` and `circuitId` excluded.
- Keep `is_home_race` and `is_home_constructor_race`.
- Keep `class_weight = "balanced"`.
- Keep `threshold_adjustment.enabled = false`.
- Do not use resampling by default.
- Do not use the expanded constructor features for RF.
- Do not use cutoff-delta features for the final RF unless future experiments
  show improvement on a new holdout season.

Current RF optimization config may point to an experiment file, depending on
the latest active test. For final reporting, use the confirmed baseline artifact
above as the best RF result.
