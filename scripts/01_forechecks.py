#!/usr/bin/env python3
"""
Build forecheck (pressing) sequences from hockey event data.

A forecheck is when a team dumps the puck into the opponent's zone and pressures
them to try to recover it. This script finds every such sequence and labels
whether the pressing team succeeded (got the puck back) or failed (defense exited).

Each forecheck starts at a dump-in event and ends at the first terminal event
within the same run of play (faceoff to whistle).

Outcomes:
- Success (y=1): pressing team recovered the puck (LPR = loose puck recovery)
  before the defense got it out
- Failure (y=0): defense exited (controlled breakout or dump out), or play
  stopped (whistle, goal, icing, offside)

Output files:
- forechecks.parquet: one row per forecheck, with outcome and labels
- forecheck_events.parquet: events that fall within a forecheck
- forecheck_tracking.parquet: tracking (player positions) for events within forechecks
"""

from pathlib import Path

import pandas as pd

##############
### CONFIG ###
##############

# Paths for input and output
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
# Event types that end a forecheck
TERMINAL_SUCCESS = ("lpr",)
TERMINAL_FAILURE = ("controlledbreakout", "dumpout", "whistle", "goal", "icing", "offside")


#################
### LOAD DATA ###
#################

# Read events and games from data/raw.
def load_events(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "events.parquet")


def load_games(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "games.parquet")


def load_tracking(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "tracking.parquet")


#######################
### FORECHECK LOGIC ###
#######################

# For each dump-in, scan forward to the first terminal event.
# LPR by pressing team = success; controlledbreakout/dumpout/whistle/etc = failure.
# Sequences with no terminal event are skipped.
def build_forecheck_sequences(events: pd.DataFrame) -> pd.DataFrame:
    events = events.sort_values(["game_id", "sl_event_id"]).reset_index(drop=True)
    dumpins = events.loc[events["event_type"] == "dumpin"].copy()
    dumpins = dumpins.rename(columns={
        "sl_event_id": "sl_event_id_start",
        "team_id": "pressing_team_id",
        "opp_team_id": "defending_team_id",
        "detail": "dumpin_detail",
    })

    # First terminal failure after each dump-in (merge + filter + groupby)
    failures = events.loc[
        events["event_type"].isin(TERMINAL_FAILURE),
        ["game_id", "sequence_id", "sl_event_id", "event_type"],
    ].copy()
    failures = failures.rename(columns={"event_type": "terminal_event_type"})
    fail_candidates = failures.merge(
        dumpins[["game_id", "sequence_id", "sl_event_id_start"]],
        on=["game_id", "sequence_id"],
        how="inner",
    )
    fail_candidates = fail_candidates[
        fail_candidates["sl_event_id"] > fail_candidates["sl_event_id_start"]
    ].sort_values(["game_id", "sequence_id", "sl_event_id_start", "sl_event_id"])
    first_failure = (
        fail_candidates.groupby(
            ["game_id", "sequence_id", "sl_event_id_start"],
            as_index=False,
        )
        .agg(
            sl_event_id_end_fail=("sl_event_id", "min"),
            terminal_event_type_fail=("terminal_event_type", "first"),
        )
    )
    first_failure = dumpins.merge(
        first_failure,
        on=["game_id", "sequence_id", "sl_event_id_start"],
        how="left",
    )

    # First LPR by pressing team after each dump-in (merge + filter + groupby)
    lpr_events = events.loc[
        events["event_type"] == "lpr",
        ["game_id", "sequence_id", "sl_event_id", "team_id"],
    ].copy()
    lpr_candidates = lpr_events.merge(
        dumpins[["game_id", "sequence_id", "sl_event_id_start", "pressing_team_id"]],
        on=["game_id", "sequence_id"],
        how="inner",
    )
    lpr_candidates = lpr_candidates[
        (lpr_candidates["team_id"] == lpr_candidates["pressing_team_id"])
        & (lpr_candidates["sl_event_id"] > lpr_candidates["sl_event_id_start"])
    ]
    first_success = (
        lpr_candidates.groupby(
            ["game_id", "sequence_id", "sl_event_id_start", "pressing_team_id"],
            as_index=False,
        )
        .agg(sl_event_id_end_success=("sl_event_id", "min"))
    )

    # Combine: for each dumpin, take whichever terminal comes first
    fc = first_failure.merge(
        first_success,
        on=["game_id", "sequence_id", "sl_event_id_start", "pressing_team_id"],
        how="left",
    )
    # Use success endpoint when it exists and is <= failure (or no failure)
    use_success = (
        fc["sl_event_id_end_success"].notna()
        & (
            fc["sl_event_id_end_fail"].isna()
            | (fc["sl_event_id_end_success"] <= fc["sl_event_id_end_fail"])
        )
    )
    fc["sl_event_id_end"] = fc["sl_event_id_end_success"].where(
        use_success, fc["sl_event_id_end_fail"]
    )
    fc["outcome"] = "failure"
    fc.loc[use_success, "outcome"] = "success"
    fc["y"] = use_success.astype(int)
    fc["terminal_event_type"] = fc["terminal_event_type_fail"]
    fc.loc[use_success, "terminal_event_type"] = "lpr"

    # Drop rows with no terminal event (period end / truncation)
    fc = fc.dropna(subset=["sl_event_id_end"]).copy()

    # Select and order columns
    fc = fc[
        [
            "game_id",
            "period",
            "period_time",
            "sequence_id",
            "sl_event_id_start",
            "sl_event_id_end",
            "pressing_team_id",
            "defending_team_id",
            "dumpin_detail",
            "outcome",
            "y",
            "terminal_event_type",
        ]
    ].reset_index(drop=True)
    fc.insert(0, "fc_sequence_id", fc.index)

    return fc


def get_forecheck_events(
    events: pd.DataFrame, forechecks: pd.DataFrame
) -> pd.DataFrame:
    # Merge events to forecheck windows on (game_id, sequence_id) to avoid
    # Cartesian product; each forecheck lives in exactly one sequence
    events_with_fc = events.merge(
        forechecks[
            ["game_id", "sequence_id", "sl_event_id_start", "sl_event_id_end", "fc_sequence_id"]
        ],
        on=["game_id", "sequence_id"],
        how="inner",
    )
    mask = (
        (events_with_fc["sl_event_id"] >= events_with_fc["sl_event_id_start"])
        & (events_with_fc["sl_event_id"] <= events_with_fc["sl_event_id_end"])
    )
    events_with_fc = events_with_fc.loc[mask, :].drop(
        columns=["sl_event_id_start", "sl_event_id_end"]
    )
    return events_with_fc.reset_index(drop=True)


def get_forecheck_tracking(
    tracking: pd.DataFrame, forecheck_events: pd.DataFrame
) -> pd.DataFrame:
    # Inner join: keep only tracking rows for (game_id, sl_event_id) in forechecks
    fc_event_keys = forecheck_events[["game_id", "sl_event_id", "fc_sequence_id"]].drop_duplicates()
    return tracking.merge(
        fc_event_keys,
        on=["game_id", "sl_event_id"],
        how="inner",
    )


############
### MAIN ###
############
def main() -> None:

    # Load data
    data_dir = DATA_DIR
    events = load_events(data_dir)
    games = load_games(data_dir)
    tracking = load_tracking(data_dir)

    # Build forechecks and extract only forecheck events
    forechecks = build_forecheck_sequences(events)
    events_with_fc = get_forecheck_events(events, forechecks)
    tracking_fc = get_forecheck_tracking(tracking, events_with_fc)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    forechecks.to_parquet(OUT_DIR / "forechecks.parquet", index=False)
    events_with_fc.to_parquet(OUT_DIR / "forecheck_events.parquet", index=False)
    tracking_fc.to_parquet(OUT_DIR / "forecheck_tracking.parquet", index=False)

    # Print summary
    n_fc = len(forechecks)
    n_success = forechecks["y"].sum()
    print(f"Built {n_fc} forecheck sequences ({n_success} success, {n_fc - n_success} failure)")
    print(f"  Written: {OUT_DIR / 'forechecks.parquet'}")
    print(f"  Written: {OUT_DIR / 'forecheck_events.parquet'}")
    print(f"  Written: {OUT_DIR / 'forecheck_tracking.parquet'}")


if __name__ == "__main__":
    main()
