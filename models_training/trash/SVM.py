"""
F1 Race Outcome Classification -- Model training (DT, RF, SVM)
================================================================
Implements decisions from README.md:
  - train 2021-2024, test 2025 (single held-out year, NOT nested CV --
    we already have a true temporal holdout, so nested CV would be
    redundant; walk-forward validation is used only for hyperparameter
    tuning INSIDE the training window)
  - class_weight='balanced' (not SMOTE, per earlier decision)
  - metric: macro-F1 primary (accuracy is misleading with imbalanced
    classes), accuracy reported as secondary reference only
  - cold-start sentinel (-1) handling: for SVM in particular, a raw -1
    is a real problem (distance-based model, -1 can sit unnaturally
    close to legitimate low values after scaling, or get crushed near
    0 for high-range features). Replaced here with an explicit
    is_missing_<col> binary flag + median imputation, fit only on the
    training fold via a Pipeline (never on the full dataset).
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              confusion_matrix)

SENTINEL = -1

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
df = pd.read_csv("/dataset/outputs/prediction.csv")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"Year range: {df.year.min()}-{df.year.max()}")

# ------------------------------------------------------------------
# 2. Identify columns that can legitimately hold the cold-start
#    sentinel (-1), per README / build_prediction_csv.R documentation.
#    NOTE: this list is deliberately explicit, not inferred by scanning
#    for -1 values in the data -- scanning would incorrectly flag any
#    column where -1 happens to be a real, meaningful value (none here,
#    but being explicit avoids that class of bug on principle).
# ------------------------------------------------------------------
COLD_START_COLS = [
    "grid", "qualifying_position",
    "driver_std_points_prev", "driver_std_position_prev",
    "constructor_std_points_prev", "constructor_std_position_prev",
    "driver_avg_position_last3", "driver_avg_position_last5", "driver_avg_position_last10",
    "driver_podium_rate_last3", "driver_podium_rate_last5", "driver_podium_rate_last10",
    "driver_points_avg_last3", "driver_points_avg_last5", "driver_points_avg_last10",
    "constructor_avg_position_last3", "constructor_avg_position_last5",
    "driver_wins_at_circuit", "driver_avg_position_at_circuit",
    "teammate_h2h_avg_position_delta",
    "driver_dnf_rate_historical",
    "points_gap_to_leader",
    "constructor_change_flag",
    "days_since_last_race",
]

for col in COLD_START_COLS:
    flag_col = f"is_missing_{col}"
    df[flag_col] = (df[col] == SENTINEL).astype(int)
    df.loc[df[col] == SENTINEL, col] = np.nan

n_flagged = sum(df[f"is_missing_{c}"].sum() for c in COLD_START_COLS)
print(f"\nTotal sentinel values converted to NaN + flagged: {n_flagged}")

# ------------------------------------------------------------------
# 3. Feature groups for the ColumnTransformer
# ------------------------------------------------------------------
categorical_cols = ["constructorId", "circuitId"]
binary_passthrough_cols = ["is_home_race", "is_home_constructor_race", "sprint_flag"] + \
                           [f"is_missing_{c}" for c in COLD_START_COLS]
# numeric = everything else that's a real feature (not id/date/target/year/round)
exclude_cols = {"raceId", "driverId", "year", "round", "date", "target"}
numeric_cols = [c for c in df.columns
                if c not in exclude_cols
                and c not in categorical_cols
                and c not in binary_passthrough_cols]

print(f"\nNumeric (scaled+imputed): {len(numeric_cols)} cols")
print(f"Categorical (one-hot): {categorical_cols}")
print(f"Binary passthrough: {len(binary_passthrough_cols)} cols")

# ------------------------------------------------------------------
# 4. Train (2021-2024) / test (2025 only -- 2026 excluded, per README
#    Section 2: partial season, not a valid held-out test set)
# ------------------------------------------------------------------
train_mask = (df.year >= 2021) & (df.year <= 2024)
test_mask = (df.year == 2025)

X_train = df.loc[train_mask, numeric_cols + categorical_cols + binary_passthrough_cols]
y_train = df.loc[train_mask, "target"]
X_test = df.loc[test_mask, numeric_cols + categorical_cols + binary_passthrough_cols]
y_test = df.loc[test_mask, "target"]

print(f"\nTrain: {len(X_train)} rows | Test (2025): {len(X_test)} rows")
print("Train class distribution:\n", y_train.value_counts())
print("Test class distribution:\n", y_test.value_counts())

# ------------------------------------------------------------------
# 5. Walk-forward folds for hyperparameter tuning -- INSIDE the
#    training window only. Not K-fold random (would break temporal
#    order), not nested CV (redundant -- we already have the true
#    2025 holdout for the final unbiased estimate).
#      Fold 1: train 2021        -> validate 2022
#      Fold 2: train 2021-2022   -> validate 2023
#      Fold 3: train 2021-2023   -> validate 2024
# ------------------------------------------------------------------
train_years = df.loc[train_mask, "year"].reset_index(drop=True)
X_train_idx = X_train.reset_index(drop=True)

walk_forward_folds = []
for val_year in [2022, 2023, 2024]:
    tr_idx = np.where(train_years < val_year)[0]
    va_idx = np.where(train_years == val_year)[0]
    walk_forward_folds.append((tr_idx, va_idx))
    print(f"Fold (validate {val_year}): train={len(tr_idx)} rows, val={len(va_idx)} rows")

# ------------------------------------------------------------------
# 6. Preprocessing pipeline (shared across all 3 models)
#    Imputer + scaler fit ONLY on each training fold via Pipeline --
#    never on the full dataset, per the leakage discipline established
#    throughout this project.
# ------------------------------------------------------------------
preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("bin", "passthrough", binary_passthrough_cols),
])

# ------------------------------------------------------------------
# 7. Model definitions + hyperparameter grids
# ------------------------------------------------------------------
models = {
    "DecisionTree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        {"clf__max_depth": [2, 3, 5, 8, 12, None],
         "clf__min_samples_leaf": [1, 5, 10, 20, 40],
         "clf__criterion": ["gini", "entropy"]}
    ),
    "RandomForest": (
        RandomForestClassifier(class_weight="balanced", random_state=42, n_estimators=300),
        {"clf__max_depth": [3, 4, 6, 8, 12, None],
         "clf__min_samples_leaf": [1, 5, 10, 20],
         "clf__max_features": ["sqrt", "log2", None]}
    ),
    "SVM": (
        SVC(class_weight="balanced", random_state=42, max_iter=10000),
        # Widened after the previous run: best C=0.1 landed on the LOWER edge
        # of [0.1, 1, 10, 100] -- can't tell if it's a true optimum or the
        # search just hit the boundary. Extending down to 0.0001.
        # max_iter capped: some (C, gamma) combinations do not converge in
        # reasonable time on this sample size; capping iterations is a
        # standard, declarable practice (sklearn will emit a
        # ConvergenceWarning for combinations that hit the cap, which is
        # informative, not silently wrong).
        {"clf__C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
         "clf__kernel": ["rbf", "linear"],
         "clf__gamma": ["scale", "auto"]}
    ),
}

results = {}

for name, (estimator, param_grid) in models.items():
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", estimator),
    ])
    gs = GridSearchCV(pipe, param_grid, cv=walk_forward_folds,
                       scoring="f1_macro", n_jobs=-1)
    gs.fit(X_train_idx, y_train.reset_index(drop=True))

    print(f"Best params: {gs.best_params_}")
    print(f"Best walk-forward macro-F1 (validation): {gs.best_score_:.3f}")

    # final refit on FULL 2021-2024 train set already done by GridSearchCV
    # (refit=True by default), evaluate ONCE on the true 2025 holdout
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    results[name] = {"accuracy": acc, "macro_f1": f1m, "model": best_model}

    print(f"\n--- Test 2025 performance ---")
    print(f"Accuracy: {acc:.3f}")
    print(f"Macro-F1: {f1m:.3f}")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix (rows=true, cols=pred), classes:", sorted(y_test.unique()))
    print(confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique())))

# ------------------------------------------------------------------
# 8. Summary comparison
# ------------------------------------------------------------------
print(f"\n{'='*60}\nSUMMARY (test = 2025 only)\n{'='*60}")
summary = pd.DataFrame({
    name: {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"]}
    for name, r in results.items()
}).T
print(summary.sort_values("macro_f1", ascending=False))