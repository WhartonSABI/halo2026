# halo2026

## Directory Structure

```
.
├── scripts/
│   ├── pass_outcome_model.ipynb    # predict pass success in defensive zone
│   ├── possession_zone_exits.ipynb  # possession tracking, zone exits, turnovers
│   └── pass_lane_features.ipynb    # distance, angle, lane obstruction from tracking
└── data/
    ├── processed/
    └── raw/
        ├── events.parquet
        ├── games.parquet
        ├── players.parquet
        ├── stints.parquet
        └── tracking.parquet
```