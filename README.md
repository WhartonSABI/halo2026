# halo2026

Forecheck (pressing) sequence analysis and player attribution. Builds sequences from dump-ins, computes hazard features per frame, and credits players via participation, distance-weighted, and model-based counterfactual methods.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick start

Place raw data in `data/raw/`, then run:

```bash
python scripts/01_forechecks.py
python scripts/02_eda.py              # EDA (optional)
python scripts/03_features.py
python scripts/04_simple-attribution.py
python scripts/06_tuning.py           # GBM/XGBoost tuning (optional; run before modeling)
python scripts/07_modeling.py
python scripts/08_ranking.py
```

Outputs: `data/results/ranking.csv` (composite player ranks), `participation.csv`, `distance.csv`, `modeling.csv`, `player_press.parquet`.

---

## Data flow

```
raw/
  events.parquet
  tracking.parquet
  stints.parquet
  players.parquet
  games.parquet
       │
       ▼
01_forechecks ──► forechecks.parquet, forecheck_events.parquet, forecheck_tracking.parquet
       │
       ▼
03_features ──► hazard_features.parquet  (one row per frame: fc_sequence_id × sl_event_id)
       │
       ├──────────────────────────────────────────────┐
       ▼                                              ▼
04_simple-attribution                           06_tuning (optional)
       │                                              │
       ├── participation (stints only)               ▼
       │   total = seq_success - avg           tuning_results.csv
       │   equal split among on-ice skaters          │
       │                                              ▼
       ├── distance (terminal frame only)      07_modeling
       │   total = seq_success - avg                 │
       │   weight by 1/r to carrier                  ├── trains hazard model (tuning winner or fallback)
       │   F1..F5 tracked; unseen get fallback      ├── fits slot RF predictors
       │                                             ├── credits on FULL data (stint-based)
       ▼                                             ▼
participation.csv, distance.csv              modeling.csv, player_press.parquet
       │                                             │
       └───────────────────┬─────────────────────────┘
                           ▼
                    08_ranking ──► ranking.csv (composite_rank = mean of method ranks)
```

---

## Pipeline

| Step | Script | Inputs | Outputs |
|------|--------|--------|---------|
| 1 | `01_forechecks.py` | `events.parquet`, `tracking.parquet` | `forechecks.parquet`, `forecheck_events.parquet`, `forecheck_tracking.parquet` |
| 2 | `02_eda.py` | processed, raw, results | `plots/*.png`, `plots/*.gif` *(optional)* |
| 3 | `03_features.py` | processed forecheck data, raw | `hazard_features.parquet` |
| 4 | `04_simple-attribution.py` | forechecks, hazard features, raw | `terminal_recovery_value.parquet`, `participation.csv`, `distance.csv` |
| — | `05_preprocess.py` | *(library imported by 06, 07)* | — |
| 6 | `06_tuning.py` | `hazard_features.parquet` | `tuning_results.csv` *(optional)* |
| 7 | `07_modeling.py` | `hazard_features.parquet` | `modeling.csv`, `model_summary.csv`, `player_press.parquet` |
| 8 | `08_ranking.py` | participation, distance, modeling | `ranking.csv` (composite rank) |

Paths are relative to `scripts/` and `data/` (`raw/`, `processed/`, `results/`). See `data_dictionary.md` for raw schema.

**EDA** (`02_eda.py`) supports `--all` (default) or individual flags: `--possession`, `--gifs`, `--slot-audit`, `--spreads`, `--distributions`, `--rankings`, `--player-press`, `--scatter`. Produces slot-change audit, attribution spreads, contribution histograms, player ranking charts, and modeling-vs-other scatter plots.

### Step summary

| Step | Purpose |
|------|---------|
| 01 | Build forecheck sequences (dump-in → LPR → terminal event) |
| 02 | EDA visuals (optional) |
| 03 | Per-frame hazard features: carrier, F1..F5 positions/angles, outlets, controls |
| 04 | Participation (equal split) + distance (1/r weighted); both terminal-moment only |
| 06 | Tune hist_gbm, gbm, xgboost; pick best by test log loss |
| 07 | Train best model, credit on full data via counterfactual + stint attribution |
| 08 | Merge ranks, output composite |

### Attribution methods

All three methods credit on the full dataset. Each player gets `n_press` (number of forechecks they participated in) and a total value.

| Method | Value definition | Allocation |
|--------|------------------|------------|
| **Participation** | total_recovery = success - avg(success) | Equal split among pressing-team skaters on ice at terminal (from stints) |
| **Distance** | Same | Weight by 1/distance to carrier at terminal; F1..F5 use tracking; unseen get fallback |
| **Modeling** | CIF_success - CIF_failure drop under counterfactual | Stint-based: positioning (start row) + execution (split by non-start row share) |

**Consistency:** Participation uses forechecks + stints. Distance and modeling use hazard features (terminal row vs all rows). All cover the same forecheck sequences. Modeling uses stint-based attribution when slot occupants change mid-sequence (~84% of sequences).

---

## Concepts

### Forecheck outcomes

Sequences are built by `01_forechecks.py`: each starts with a dump-in and the defending team's first LPR under pressure, and ends at the first terminal event.

| Outcome | Definition |
|---------|------------|
| **Success** | Pressing team (Team A) gains possession before the puck exits the zone or play stops |
| **Failure** | Puck exits zone or stoppage occurs before Team A gains possession |
| **Dropped** | Period-end whistle (last second of period) — excluded from totals |

**Stoppages:** `whistle`, `goal`, `icing`, `offside`, `penalty`. Penalty on pressing team → failure. Penalty on defending team → success. Other stoppages → failure.

---

## Model tuning

**Tuning** (`scripts/06_tuning.py`) tunes HistGradientBoosting, GradientBoosting, and XGBoost. When `tuning_results.csv` exists, `07_modeling.py` uses that best model directly. Otherwise it trains and compares logit, HistGradientBoosting, and XGBoost.

- **Method:** `RandomizedSearchCV` with group-based cross-validation (`GroupKFold` on `fc_sequence_id`) to avoid sequence leakage
- **Metric:** log loss (3-class: ongoing, success, failure)
- **Search:** 50 random draws per model (10 in `--quick` mode)
- **Models:** HistGradientBoosting, GradientBoosting (sklearn), XGBoost

```bash
python scripts/06_tuning.py          # full: 50 iterations per model
python scripts/06_tuning.py --quick  # quick: 10 iterations per model
```

---

## Project structure

```
.
├── scripts/
│   ├── 01_forechecks.py         # Forecheck sequences from events
│   ├── 02_eda.py                # EDA animations (optional)
│   ├── 03_features.py           # Hazard features
│   ├── 04_simple-attribution.py # Participation & distance attribution
│   ├── 05_preprocess.py         # Hazard preprocessing (library for 06, 07)
│   ├── 06_tuning.py             # GBM/XGBoost hyperparameter tuning (optional)
│   ├── 07_modeling.py           # Hazard models + counterfactual credit
│   └── 08_ranking.py            # Composite ranking
├── data/
│   ├── raw/                     # events, games, players, stints, tracking
│   ├── processed/               # forechecks, hazard_features, terminal_recovery_value
│   └── results/                 # participation, distance, modeling, player_press, ranking
└── plots/                       # EDA outputs
```

---

## Hazard features reference

`data/processed/hazard_features.parquet` — one row per frame (`fc_sequence_id` × `sl_event_id`).

| Category | Features |
|----------|----------|
| **Identifiers** | `fc_sequence_id`, `sl_event_id`, `event_t`, `terminal_failure_t` |
| **Time** | `time_since_start_s`, `time_since_start_bin` |
| **Carrier** | `carrier_id`, `carrier_x`, `carrier_y`, `carrier_speed` |
| **Forecheckers F1–F5** | `F{i}_id`, `F{i}_r`, `F{i}_vr_carrier`, `F{i}_sinθ`, `F{i}_cosθ`, `F{i}_r_nearestOpp`, `F{i}_vr_nearestOpp`, `F{i}_block_severity`, `F{i}_block_center_severity` |
| **Outlets** | `outlet_candidate_count`, `unblocked_outlet_count`, `center_open`, `min_unblocked_outlet_dist` |
| **Controls** | `manpower_state`, `pressing_is_home`, `score_diff_bin`, `puck_start_x`, `puck_start_y` |
