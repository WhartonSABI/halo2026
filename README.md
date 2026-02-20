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

Place raw data in `data/raw/`, then run the full pipeline:

```bash
# Full pipeline (01 → 07 + visuals)
./scripts/run_full_pipeline.sh
```

Or run step-by-step:

```bash
python scripts/01_forechecks.py
python scripts/02_features.py
python scripts/03_simple-attribution.py
python scripts/04_tuning.py           # optional; run before modeling for better hyperparams
python scripts/05_modeling.py
python scripts/06_ranking.py
python scripts/07_evaluation.py       # calibration + benchmark
python scripts/_visuals.py            # all plots
```

**Modeling + ranking + eval + visuals only** (if 01–03 already run):

```bash
./scripts/run_pipeline.sh
```

Quick test with subsample:

```bash
./scripts/run_pipeline.sh --quick     # 05 uses --max-rows 10000
```

**Outputs:**
- `data/results/` — `ranking.csv`, `participation.csv`, `distance.csv`, `modeling.csv`
- `data/processed/` — `player_press.parquet`
- `plots/` — calibration, attribution spreads, rankings, scatter, etc.

---

## Full pipeline scripts

| Script | Description |
|--------|-------------|
| `./scripts/run_full_pipeline.sh` | Run 01 → 07 + visuals. Use `--quick` for subsampled modeling (05 only). |
| `./scripts/run_pipeline.sh` | Run 05 → 07 + visuals (skips 01–04; assumes forechecks, features, attribution exist). |

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
02_features ──► hazard_features.parquet  (one row per frame: fc_sequence_id × sl_event_id)
       │
       ├─────────────────────────────────────────────────────┐
       ▼                                                     ▼
03_simple-attribution                                  04_tuning (optional)
       │                                                     │
       ├── participation (stints only)                       ▼
       │   total = seq_success - avg                  tuning_quick.csv / tuning_full.csv
       │   equal split among on-ice skaters                  │
       │                                                     ▼
       ├── distance (terminal frame only)             05_modeling
       │   total = seq_success - avg                         │
       │   weight by 1/r to carrier                          ├── start model (p₀) + hazard (exec rows)
       │   F1..F5 tracked; unseen get fallback               ├── slot RF predictors (ghosts)
       │                                                     ├── credit: p₀−p̄, outcome−p₀; allocate by ghost shares
       ▼                                                     ▼
participation.csv, distance.csv                    modeling.csv (player_press.parquet in processed/)
       │                                                     │
       └───────────────────────────┬─────────────────────────┘
                                  ▼
                           06_ranking ──► ranking.csv (composite rank)
                                  │
                                  ▼
                           07_evaluation ──► calibration plots, benchmark (start + hazard)
                                  │
                                  ▼
                           _visuals.py ──► plots/ (possession, spreads, rankings, scatter, etc.)
```

---

## Pipeline

| Step | Script | Inputs | Outputs |
|------|--------|--------|---------|
| 1 | `01_forechecks.py` | `events.parquet`, `tracking.parquet` | `forechecks.parquet`, `forecheck_events.parquet`, `forecheck_tracking.parquet` |
| 2 | `02_features.py` | processed forecheck data, raw | `hazard_features.parquet` |
| 3 | `03_simple-attribution.py` | forechecks, hazard features, raw | `terminal_recovery_value.parquet`, `participation.csv`, `distance.csv` |
| 4 | `04_tuning.py` | `hazard_features.parquet` | `tuning_quick.csv` or `tuning_full.csv` *(optional)* |
| 5 | `05_modeling.py` | `hazard_features.parquet` | `modeling.csv`, `model_summary.csv`; `processed/player_press.parquet` |
| 6 | `06_ranking.py` | participation, distance, modeling | `ranking.csv` (composite rank) |
| 7 | `07_evaluation.py` | hazard_features, modeling | Calibration plots (`plots/`), benchmark results |
| — | `_visuals.py` | results CSVs | `plots/` — possession, spreads, rankings, scatter, etc. |

Paths are relative to `scripts/` and `data/` (`raw/`, `processed/`, `results/`). See `data_dictionary.md` for raw schema.

**EDA visuals** (`_visuals.py`): `python scripts/_visuals.py` (default: all) or `--possession`, `--gifs`, `--slot-audit`, `--spreads`, `--distributions`, `--rankings`, `--player-press`, `--scatter`, `--team-check`. Produces slot-change audit, attribution spreads, contribution histograms, player ranking charts, modeling-vs-other scatter, team-level check ability, and press GIFs.

### Step summary

| Step | Purpose |
|------|---------|
| 01 | Build forecheck sequences (dump-in → LPR → terminal event) |
| 02 | Per-frame hazard features: carrier, F1..F5 positions/angles, outlets, controls |
| 03 | Participation (equal split) + distance (1/r weighted); both terminal-moment only |
| 04 | Tune rf, hist_gbm, xgboost; pick best by test log loss *(optional)* |
| 05 | Hybrid: start model (p₀) + hazard (exec). Credit: p₀−p̄ and outcome−p₀ via ghost shares |
| 06 | Merge ranks, output composite |
| 07 | Calibration diagrams (start + hazard), prediction benchmark |
| _visuals | EDA plots in `plots/` |

### Attribution methods

All three methods credit on the full dataset. Each player gets `n_press` (number of forechecks they participated in) and a total value.

| Method | Value definition | Allocation |
|--------|------------------|------------|
| **Participation** | total_recovery = success - avg(success) | Equal split among pressing-team skaters on ice at terminal (from stints) |
| **Distance** | Same | Weight by 1/distance to carrier at terminal; F1..F5 use tracking; unseen get fallback |
| **Modeling** | Hybrid: p₀−p̄ (positioning) + outcome−p₀ (exec) | Ghost shares for allocation; stint attribution |

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

**Tuning** (`scripts/04_tuning.py`) tunes RandomForest, HistGradientBoosting, and XGBoost. When `tuning_quick.csv` or `tuning_full.csv` exists, `05_modeling.py` uses that best model. Otherwise run tuning first.

- **Method:** `RandomizedSearchCV` with group-based cross-validation (`GroupKFold` on `fc_sequence_id`) to avoid sequence leakage
- **Metric:** log loss (3-class: ongoing, success, failure)
- **Search:** 50 random draws per model (10 in `--quick` mode)
- **Models:** HistGradientBoosting, GradientBoosting (sklearn), XGBoost

```bash
python scripts/04_tuning.py      # quick: 10 iterations per model
python scripts/04_tuning.py --full  # full: 50 iterations per model
```

---

## Project structure

```
.
├── scripts/
│   ├── 01_forechecks.py         # Forecheck sequences from events
│   ├── 02_features.py           # Hazard features
│   ├── 03_simple-attribution.py # Participation & distance attribution
│   ├── 04_tuning.py             # RF/HistGBM/XGBoost hyperparameter tuning (optional)
│   ├── 05_modeling.py           # Hazard models + counterfactual credit
│   ├── 06_ranking.py            # Composite ranking
│   ├── 07_evaluation.py         # Calibration + benchmark
│   ├── run_full_pipeline.sh     # Run 01 → 07 + visuals
│   ├── run_pipeline.sh          # Run 05 → 07 + visuals (assumes 01–03 done)
│   ├── _preprocess.py           # Shared preprocessing (tuning, modeling)
│   └── _visuals.py              # EDA visuals
├── data/
│   ├── raw/                     # events, games, players, stints, tracking
│   ├── processed/               # forechecks, hazard_features, terminal_recovery_value, player_press
│   └── results/                 # participation, distance, modeling, ranking
└── plots/                       # Calibration, attribution spreads, rankings, etc.
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
