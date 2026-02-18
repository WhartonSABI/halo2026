# halo2026

Forecheck (pressing) sequence analysis and player attribution. Builds sequences from dump-ins, computes hazard features, and credits players via participation, distance-weighted, and model-based counterfactual methods.

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
python scripts/02_features.py
python scripts/04_simple-attribution.py
python scripts/05_modeling.py
python scripts/06_build-ranking.py
```

Outputs: `data/results/ranking.csv` (composite player ranks), `participation.csv`, `distance.csv`, `modeling.csv`.

---

## Concepts

### Forecheck outcomes

Sequences are built by `01_forechecks.py`: each starts with a dump-in and the defending team’s first LPR under pressure, and ends at the first terminal event.

| Outcome | Definition |
|---------|------------|
| **Success** | Pressing team (Team A) gains possession before the puck exits the zone or play stops |
| **Failure** | Puck exits zone or stoppage occurs before Team A gains possession |
| **Dropped** | Period-end whistle (last second of period) — excluded from totals |

**Stoppages:** `whistle`, `goal`, `icing`, `offside`, `penalty`. Penalty on pressing team → failure. Penalty on defending team → success. Other stoppages → failure.

---

## Pipeline

| Step | Script | Inputs | Outputs |
|------|--------|--------|---------|
| 1 | `01_forechecks.py` | `events.parquet`, `tracking.parquet` | `forechecks.parquet`, `forecheck_events.parquet`, `forecheck_tracking.parquet` |
| 2 | `02_features.py` | processed forecheck data, raw | `hazard_features.parquet` |
| 3 | `03_eda.py` | processed, raw | `plots/game_play_*.gif` *(optional)* |
| 4 | `04_simple-attribution.py` | forechecks, hazard features, raw | `terminal_recovery_value.parquet`, `participation.csv`, `distance.csv` |
| 5 | `05_modeling.py` | `hazard_features.parquet` | `modeling.csv`, `model_summary.csv` |
| 6 | `06_build-ranking.py` | participation, distance, modeling CSVs | `ranking.csv` (composite rank) |

Paths are relative to `scripts/` and `data/` (`raw/`, `processed/`, `results/`). See `data_dictionary.md` for raw schema.

---

## Project structure

```
.
├── scripts/
│   ├── 01_forechecks.py         # Forecheck sequences from events
│   ├── 02_features.py           # Hazard features
│   ├── 03_eda.py                # EDA animations (optional)
│   ├── 04_simple-attribution.py # Participation & distance attribution
│   ├── 05_modeling.py          # Hazard models + counterfactual credit
│   └── 06_build-ranking.py    # Composite ranking
├── data/
│   ├── raw/                     # events, games, players, stints, tracking
│   ├── processed/               # forechecks, hazard_features, terminal_recovery_value
│   └── results/                 # participation, distance, modeling, ranking
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
