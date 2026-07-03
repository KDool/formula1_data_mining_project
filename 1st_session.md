
### Table triage

| Table | Verdict | Notes |
|---|---|---|
| `races.csv` | Essential | Backbone for temporal ordering. **Use `date`, not `raceId` or `round`, for any chronological logic** (see Section 4.3). |
| `results.csv` | Essential | Source of target and of `grid` (pre-race, safe). |
| `driver_standings.csv` | Usable, only if shifted | Includes points from the current race (verified, Section 4.1). Must reference the immediately preceding race by date. |
| `constructor_standings.csv` | Usable, only if shifted | Same issue as above. |
| `drivers.csv` | Useful | `dob` for age; static, no leakage risk. |
| `constructors.csv` | Useful | Static reference (nationality). |
| `circuits.csv` | Useful, with a caveat | Same `circuitId` may correspond to different physical layouts across decades (safety-driven track changes). Not corrected in this version — declare as limitation. |
| `qualifying.csv` | Usable | Pre-race event, no leakage. Raw `q1/q2/q3` times dropped (Section 6). |
| `lap_times.csv` | Usable only for historical aggregates | Data generated *during* the race — never usable for the race being predicted, only for past races. |
| `pit_stops.csv` | Same as `lap_times.csv` | |
| `sprint_results.csv` | No separate handling needed | All sprint `raceId`s already overlap with `results.csv` (verified). Sprint points are already absorbed into the cumulative `driver_standings` for that race weekend. |
| `constructor_results.csv` | Likely redundant | Appears derivable from `results.csv` grouped by constructor/race. Not used. |
| `status.csv` | Needs remapping | 140+ granular categories, mostly single-occurrence. Must be grouped into macro-categories before use (still open, Section 5). |
| `seasons.csv` | Unused | Only `year` and a Wikipedia URL. No modeling value. |

**Not available in the dataset**: weather data. No column in any table encodes wet/dry conditions. The "wet-track trend" feature discussed early in the project is **not implementable** with this dataset as-is. Decision pending: drop the idea, or bring in an external weather source (adds another leakage-sensitive join by date/circuit — not free).

---

## 2. Time window: training and test split

### Decision: train on 2021–2024, test on 2025.



**What was tested**: a minimal baseline model (RandomForest, `grid`, `age`, shifted standings, `constructorId`) was trained on two windows — starting 2014 and starting 2021 — and evaluated on the *same* held-out period, across 10 random seeds.

| Training window | Rows | Accuracy (mean ± std) | Macro-F1 (mean ± std) |
|---|---|---|---|
| 2014–2024 | 4626 | 0.654 ± 0.003 | 0.662 ± 0.004 |
| 2021–2024 | 1799 | 0.669 ± 0.007 | 0.674 ± 0.005 |

The shorter, more recent window performs consistently better despite having less than half the data — consistent with the hypothesis that team/car turnover between 2014 and 2021 introduces more noise than signal for predicting 2025 races.

**Caveats, stated explicitly, not swept under the rug**:
- This was a single pilot experiment with minimal features and untuned hyperparameters. It is directionally informative, not a final proof. Re-run it once the full feature set is built, with proper nested CV, before citing this comparison as evidence in the report.
- Fewer training examples per class (2021–2024) means noisier cross-validation estimates during model selection, even if the final held-out performance is better. These are two different things — performance on one fixed test set vs. stability of the model-selection procedure — and should not be conflated in the report.

### Test set: 2025 only, not "2025/2026"

Verified: 2025 is a complete season in the dataset (24/24 races have results). 2026 has only 3 of 22 scheduled races completed (current date: dataset snapshot mid-2026). Mixing a complete season with 3 early-season races of an in-progress season produces an inhomogeneous, unbalanced test set (early-season rows systematically differ — e.g., near-zero "current season points so far" for everyone). **2025 alone is the primary test set.** 2026 may be cited only as an informal, non-statistical sanity check ("the model correctly predicted X of the 3 races run so far in 2026"), never blended into reported metrics.

---

## 3. Critical data-quality findings (verified empirically, not assumed)

### 3.1 `driver_standings` / `constructor_standings` leak the current race by construction

Verified on Hamilton, 2009 season: `results.csv` shows 0, 1, 3 points in races 1–3. `driver_standings.csv` shows cumulative points of 1 and 4 at `raceId` 2 and 3 respectively — i.e., the standings row for race X already includes race X's own points. **Any standings feature must reference the row for the race immediately preceding the target race (by date), never the same `raceId`.**

### 3.2 `results.points` already encodes the historically correct scoring system

Verified across 1960, 1990, 2003, 2010, 2020, 2024: the distinct point values per year match the scoring system actually in force that season (e.g., max 8 points in 1960, max 25 in 2010, max 26 in 2020 with the fastest-lap bonus point). **No manual reconstruction of historical points rules is needed** — this had been an open concern earlier in the project and is now resolved.

### 3.3 `raceId` is not a reliable proxy for chronological order

Verified globally and specifically within the 2021–2025 window: 2 pairs of races in 2021 have `raceId` increasing while `date` does not follow the same order (likely due to COVID-era calendar reshuffling). Confirmed isolated to 2021; 2022–2025 are clean (0 anomalies each).

**Consequence**: any `WHERE raceId < X` pattern used as a proxy for "past races" is unsafe for these rows and would silently pull in a future race for a small number of cases. **All temporal joins must sort and filter on `date`, never on `raceId` or `round`.**

**Decision**: this does *not* justify dropping the 2021 season. The underlying data (dates, results, points) is correct; only the `raceId` ordering is unreliable, and it is fully corrected by using `date`-based joins. Discarding a full season of data over an issue that is already neutralized by the join method would be a disproportionate, unjustified loss of data.

### 3.4 Driver standings reset every season

Verified on Hamilton: 387.5 cumulative points at the end of 2021, reset to 15 points after round 1 of 2022. A naive "points at previous race" shift will, at the first race of each season, return the *previous season's final total* rather than 0. This is not leakage, but a semantic error if left undocumented. **Decision: keep two separate features** — `driver_std_points_prev` (raw, may span a season boundary, usable as a proxy for general driver reputation/level) and a season-reset version representing "points scored so far this season" (explicitly 0 at round 1 of each year).

---

## 4. Final `prediction.csv` feature list

### Target
| Column | Source | Note |
|---|---|---|
| `target` | `results.positionOrder`, `results.points` (current race) | `podium` if `positionOrder ≤ 3`; `points` if `points > 0` and `positionOrder > 3`; `no_points` otherwise. |

### Raw features — pre-race, safe without temporal shift
| Column | Source |
|---|---|
| `grid` | results.csv |
| `constructorId` | results.csv |
| `circuitId` | races.csv |
| `qualifying_position` | qualifying.csv |
| `driver_age` | drivers.csv (`dob`) |
| `year`, `round`, `date` | races.csv (join keys / context, not necessarily direct predictors) |

**Dropped from this category**: `driver_nationality`, `constructor_nationality` standalone (no plausible causal link to performance on their own — see `is_home_race` below for the version that does make sense); `number` (identifier, no rationale); `q1`, `q2`, `q3` (redundant with `grid`/`qualifying_position`, and not comparable across circuits without normalization — dropped rather than half-used).

### Raw features — "as-of" (require a date-based shift to the previous race)
| Column | Source |
|---|---|
| `driver_std_points_prev`, `driver_std_position_prev` | driver_standings.csv |
| `constructor_std_points_prev`, `constructor_std_position_prev` | constructor_standings.csv |

### Derived features
| # | Feature | Window | Leakage check |
|---|---|---|---|
| 1 | `driver_avg_position_last3/5/10` | 3 separate windows (feature, not a tuned hyperparameter — see rationale below) | As-of, OK |
| 2 | `driver_podium_rate_last3/5/10` | idem | As-of, OK |
| 3 | `driver_points_avg_last3/5/10` | idem | As-of, OK |
| 4 | `constructor_avg_position_last3/5` | short windows only (car development moves faster than driver form) | As-of, OK |
| 5 | `driver_wins_at_circuit`, `driver_avg_position_at_circuit` | full pre-2021 history | As-of, OK — limitation: circuit layout changes not accounted for (Section 1) |
| 6 | `constructor_avg_position_at_circuit` | idem | As-of, OK |
| 7 | `teammate_h2h_avg_position_delta` | history vs. *current* teammate only | As-of, OK — must filter to races with the same current teammate, not any past teammate |
| 8 | `driver_dnf_rate_historical`, `constructor_dnf_rate_historical` | full history | As-of, OK — depends on the `status.csv` regrouping (open, Section 5) |
| 9 | `points_gap_to_leader` | previous race | As-of, OK |
| 10 | `constructor_change_flag` | current vs. previous race | No risk, `constructorId` for the current race is pre-race information |
| 11 | `days_since_last_race` | current/previous date | No risk |
| 12 | `is_sprint_weekend` | `races.sprint_date` | No risk, calendar fact |
| 13 | `driver_seasons_in_f1` | current year − year of driver's first race | No risk |
| 14 | `is_home_race` | `driver.nationality == circuit.country` | No risk, static |
| 15 | `is_home_constructor_race` | `constructor.nationality == circuit.country` | No risk, static |

**On features 7 and 10 (teammate head-to-head, constructor change)**: these were nearly dropped based on an unverified domain intuition ("doesn't seem that informative"). This is corrected: head-to-head teammate comparison is a standard analytical tool in real F1 analysis precisely because it isolates driver skill from car performance — the intuition to drop it was not supported. Decision: **both features are kept in the first modeling pass.** If, after building an initial model, SHAP/feature importance shows they contribute marginally, they can be dropped with an empirical justification in the report. Dropping them beforehand on intuition alone would be a weak answer if challenged during the oral exam.

**On rolling windows as fixed features vs. a tunable hyperparameter**: fixed features (3/5/10) were chosen over treating the window size as a hyperparameter to search over. Both are equally safe with respect to leakage (this is a deterministic per-row transform based on absolute history, not something fitted on the training fold). The reasons for fixed features: (a) tree-based models can combine multiple windows non-linearly, capturing both "current form" and "general consistency" simultaneously, which a single searched value cannot; (b) implementing window size as a proper sklearn hyperparameter requires a custom transformer with access to full historical data beyond the CV fold's X matrix — an easy place to introduce a subtle bug that looks exactly like a data problem without being classic leakage.

### Excluded — outcome of the race being predicted (direct leakage if used as input)
`position`, `positionOrder`, `positionText`, `points` (as input, not as target source), `laps`, `time`, `milliseconds`, `fastestLap`, `rank`, `fastestLapTime`, `fastestLapSpeed`, `statusId` (raw), any `lap_times`/`pit_stops` row sharing the same `raceId` as the target row.

### Not available
Weather data — no column exists in this dataset (Section 1).

---

## 5. Open decisions — not yet resolved, must be closed before writing code

1. **`status.csv` regrouping**: needs an explicit mapping from `statusId` to macro-categories (e.g., Finished / Mechanical / Accident-Collision / Disqualified / Other) before feature #8 is computable. Not yet defined. Note: lumping all mechanical failures and accidents into one bucket loses the distinction between "own car fault" and "someone else's fault" — decide if that distinction matters enough to keep separate.
2. **Cold-start sentinel value**: for drivers/circuits with no prior history (features 1–3, 5–8), a value distinct from 0 must be used (e.g., `-1`), since 0 is ambiguous with "zero wins/zero gap from an actual sample." Not yet fixed.
3. **Weather data**: drop the idea entirely, or source it externally. If external, it is another leakage-sensitive join (by date + circuit) that needs the same as-of discipline as everything else here — not a free addition.
4. **Feature table scope for as-of computation**: source tables used to compute historical aggregates (standings, lap times, circuit history, etc.) must remain **unrestricted by year** — i.e., use full history back as far as the data allows, even though the *rows being classified* are restricted to 2021+. Restricting the source tables to 2021+ as well would artificially zero out the history of every driver at the start of the 2021 season (including established drivers with years of prior data), which is not representative of reality. This must be implemented deliberately, not left as an accidental side effect of a single global year filter.

---




*This document reflects the state of decisions as of this stage of the project. Any change to the year window, feature list, or leakage-handling approach should be reflected here before implementation, not discovered afterward.*
