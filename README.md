# halo2026

## Forecheck outcomes

Forecheck sequences are built by `scripts/01_forechecks.py`: each sequence starts with a dump-in and the defending team’s first LPR under pressure, and ends at the first terminal event.

**Success:** pressing team (Team A) gains possession before the puck exits the zone or play stops.

**Failure:** puck exits the zone, or a stoppage occurs, before Team A gains possession.

**Dropped (not counted):** sequences that end on a **period-end whistle** (whistle in the last second of a period). These are excluded from success/failure totals.

**Stoppages and penalties:**

- **Penalty on pressing team** → failure.
- **Penalty on defending team** (team in possession) → success.
- Other stoppages (generic **whistle**, **goal**, **icing**, **offside**) → failure.

Stoppage event types: `whistle`, `goal`, `icing`, `offside`, `penalty`.

---

## Directory Structure

```
.
├── scripts/
│   ├── 01_forechecks.py       # forecheck sequences from event data
│   ├── 02_features.py
│   ├── pass_lane_features.py  # distance, angle, lane obstruction from tracking
│   └── PassLanes.py           # velocity-adjusted pass cones, lane_blocked
└── data/
    ├── processed/
    └── raw/
        ├── events.parquet
        ├── games.parquet
        ├── players.parquet
        ├── stints.parquet
        └── tracking.parquet
```