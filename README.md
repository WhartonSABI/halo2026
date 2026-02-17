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

## Hazard features (02_features.py)

Output: `data/processed/hazard_features.parquet`. One row per frame (fc_sequence_id × sl_event_id) within forecheck sequences.

### Identifiers & targets

| Feature | Description |
|---------|-------------|
| `fc_sequence_id` | Forecheck sequence ID |
| `sl_event_id` | Event ID within game |
| `event_t` | 1 if success at this terminal event |
| `terminal_failure_t` | 1 if failure at this terminal event |

### Time

| Feature | Description |
|---------|-------------|
| `time_since_start_s` | Seconds since sequence start |
| `time_since_start_bin` | Integer seconds bin |

### Carrier (primary actor / puck holder)

| Feature | Description |
|---------|-------------|
| `carrier_id` | Player ID of carrier |
| `carrier_x`, `carrier_y` | Carrier position (event x,y) |
| `carrier_speed` | Carrier speed from tracking |

### Forecheckers F1–F5 (closest 5 to carrier)

| Feature | Description |
|---------|-------------|
| `F{i}_id` | Player ID |
| `F{i}_r` | Distance to carrier |
| `F{i}_vr_carrier` | Radial closing speed toward carrier |
| `F{i}_sinθ`, `F{i}_cosθ` | Unit vector toward carrier (dy/r, dx/r) |
| `F{i}_r_nearestOpp`, `F{i}_vr_nearestOpp` | Distance and closing speed to nearest opponent (F2–F5) |
| `F{i}_block_severity` | Severity of any outlet lane blocked |
| `F{i}_block_center_severity` | Severity of center lane blocked |

### Outlets / passing lanes

| Feature | Description |
|---------|-------------|
| `outlet_candidate_count` | Number of outlet candidates |
| `unblocked_outlet_count` | Number of unblocked outlets |
| `center_open` | 1 if any unblocked center lane |
| `min_unblocked_outlet_dist` | Distance to nearest unblocked outlet |

### Sequence-level controls

| Feature | Description |
|---------|-------------|
| `manpower_state` | e.g. 5v5 |
| `pressing_is_home` | 1 if pressing team is home |
| `score_diff_bin` | trailing / tied / leading |
| `puck_start_x`, `puck_start_y` | Puck location at sequence start |

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
    │   ├── forechecks.parquet
    │   ├── forecheck_events.parquet
    │   ├── forecheck_tracking.parquet
    │   └── hazard_features.parquet
    └── raw/
        ├── events.parquet
        ├── games.parquet
        ├── players.parquet
        ├── stints.parquet
        └── tracking.parquet
```