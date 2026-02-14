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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

# Terminal events determine forecheck outcome (first one after dump-in wins).
# Success: pressing team recovers puck before defense exits.
TERMINAL_SUCCESS = ("lpr",)
# Failure: defense exits (breakout/dump-out) or play stops.
TERMINAL_FAILURE = ("controlledbreakout", "dumpout", "whistle", "goal", "icing", "offside")


#################
### LOAD DATA ###
#################

def load_events(data_dir: Path) -> pd.DataFrame:
    """Load event stream from data_dir/events.parquet."""
    return pd.read_parquet(data_dir / "events.parquet")


def load_games(data_dir: Path) -> pd.DataFrame:
    """Load game metadata from data_dir/games.parquet."""
    return pd.read_parquet(data_dir / "games.parquet")


def load_tracking(data_dir: Path) -> pd.DataFrame:
    """Load player tracking from data_dir/tracking.parquet."""
    return pd.read_parquet(data_dir / "tracking.parquet")


#######################
### FORECHECK LOGIC ###
#######################

def build_forecheck_sequences(events: pd.DataFrame) -> pd.DataFrame:
    """
    Identify forecheck sequences: each starts at a dump-in and ends at the first
    terminal event (LPR = success, defense exit/stoppage = failure).
    """
    events = events.sort_values(["game_id", "sl_event_id"]).reset_index(drop=True)

    # --- 1. Find all dump-ins (forecheck start events) ---
    dumpins = events.loc[events["event_type"] == "dumpin"].copy()
    dumpins = dumpins.rename(columns={
        "sl_event_id": "sl_event_id_start",
        "team_id": "pressing_team_id",
        "opp_team_id": "defending_team_id",
        "detail": "dumpin_detail",
    })

    # --- 2. For each dump-in: first failure event in same sequence after dump-in ---
    failures = events.loc[
        events["event_type"].isin(TERMINAL_FAILURE),
        ["game_id", "sequence_id", "sl_event_id", "event_type"],
    ].copy()
    failures = failures.rename(columns={"event_type": "terminal_event_type"})
    # Join failures to dump-ins on (game_id, sequence_id); keep only failures after dump-in.
    fail_candidates = failures.merge(
        dumpins[["game_id", "sequence_id", "sl_event_id_start"]],
        on=["game_id", "sequence_id"],
        how="inner",
    )
    fail_candidates = fail_candidates[
        fail_candidates["sl_event_id"] > fail_candidates["sl_event_id_start"]
    ].sort_values(["game_id", "sequence_id", "sl_event_id_start", "sl_event_id"])
    # Take min(sl_event_id) per dump-in = first failure.
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

    # --- 3. For each dump-in: first LPR by pressing team in same sequence after dump-in ---
    lpr_events = events.loc[
        events["event_type"] == "lpr",
        ["game_id", "sequence_id", "sl_event_id", "team_id"],
    ].copy()
    lpr_candidates = lpr_events.merge(
        dumpins[["game_id", "sequence_id", "sl_event_id_start", "pressing_team_id"]],
        on=["game_id", "sequence_id"],
        how="inner",
    )
    # LPR must be by pressing team and after dump-in.
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

    # --- 4. Decide outcome: whichever terminal event comes first ---
    fc = first_failure.merge(
        first_success,
        on=["game_id", "sequence_id", "sl_event_id_start", "pressing_team_id"],
        how="left",
    )
    # Success wins when LPR exists and occurs before (or at same event as) first failure.
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

    # Drop rows with no terminal event (e.g. period end, data truncation).
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
    """
    Return all events that fall within each forecheck window [sl_event_id_start, sl_event_id_end].
    Join on (game_id, sequence_id); each forecheck belongs to exactly one sequence.
    """
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
    """
    Return tracking (player positions) for events that belong to forechecks.
    Keeps only (game_id, sl_event_id) pairs that appear in forecheck_events.
    """
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
    data_dir = DATA_DIR
    events = load_events(data_dir)
    games = load_games(data_dir)
    tracking = load_tracking(data_dir)

    # Build forecheck sequences, extract their events and tracking, write to parquet.
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
