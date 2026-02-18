#!/usr/bin/env python3
"""
Build forecheck (pressing) sequences from hockey event data.

Logic:
  1. Start: dump-in → LPR+ under pressure (Team B recovers).
  2. Success: Team A gets possession before puck exits zone.
  3. Failure: puck exits or stoppage before Team A gets it.
  4. Penalty on pressing team → failure; penalty on defending team (possession) → success.
  5. Period-end whistle → dropped (not success or failure).
"""

from pathlib import Path

import numpy as np
import pandas as pd

##############
### CONFIG ###
##############

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

LPR_UNDER_PRESSURE = ("opdump", "opdumpcontested", "hipresopdump", "hipresopdumpcontested", "contested")
# Stoppage event_types (other whistles): whistle (generic, e.g. puck out of play; period-end dropped),
# goal, icing, offside, penalty (outcome set by which team took the penalty)
STOPPAGE = ("whistle", "goal", "icing", "offside", "penalty")
POSSESSION_EVENTS = ("lpr", "reception", "carry", "pass", "dumpin", "shot")
BLUE_LINE = 25
# Period length (sec) for detecting period-end whistle: regulation 1200, OT often 300
REGULATION_PERIOD_LENGTH = 1200.0
OT_PERIOD_LENGTH = 300.0


#################
### LOAD DATA ###
#################

def load_events(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "events.parquet")


def load_tracking(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "tracking.parquet")


#######################
### FORECHECK LOGIC ###
#######################

def build_forecheck_sequences(events: pd.DataFrame) -> pd.DataFrame:
    """Identify forecheck sequences. Vectorized scan for first terminal event per start."""
    events = events.sort_values(["game_id", "sl_event_id"]).reset_index(drop=True)

    # 1. Dump-ins (Team A dumps, Team B defends)
    dumpins = (
        events[events["event_type"] == "dumpin"]
        [["game_id", "period", "period_time", "sequence_id", "sl_event_id", "team_id", "opp_team_id", "detail"]]
        .rename(columns={"team_id": "pressing_team_id", "opp_team_id": "defending_team_id", "detail": "dumpin_detail"})
    )

    # 2. First LPR+ by defending team after dump-in (under-pressure detail)
    lpr_def = (
        events[
            (events["event_type"] == "lpr")
            & (events["outcome"] == "successful")
            & (events["detail"].isin(LPR_UNDER_PRESSURE))
        ]
        [["game_id", "sequence_id", "sl_event_id", "team_id", "x", "y", "detail"]]
        .rename(columns={"sl_event_id": "sl_event_id_lpr", "detail": "lpr_detail"})
    )

    starts = (
        dumpins.merge(lpr_def, on=["game_id", "sequence_id"], how="inner")
        .query("team_id == defending_team_id and sl_event_id_lpr > sl_event_id")
        .sort_values("sl_event_id_lpr")
        .groupby(["game_id", "sequence_id", "sl_event_id"], as_index=False)
        .first()
    )
    starts = starts.rename(columns={"sl_event_id": "sl_event_id_dumpin"})
    starts["sl_event_id_start"] = starts["sl_event_id_lpr"]
    starts["puck_x_at_start"] = starts["x"]
    starts["puck_y_at_start"] = starts["y"]
    starts["sign_negative"] = starts["puck_x_at_start"] < 0

    # 3. Vectorized scan: for each start, find first terminal event (stoppage, zone exit, or Team A possession)
    ev_cols = ["game_id", "sequence_id", "sl_event_id", "event_type", "team_id", "x", "period", "period_time"]
    scan = events[ev_cols].merge(
        starts[["game_id", "sequence_id", "sl_event_id_start", "sl_event_id_dumpin", "pressing_team_id", "sign_negative"]],
        on=["game_id", "sequence_id"],
        how="inner",
    )
    scan = scan[scan["sl_event_id"] > scan["sl_event_id_start"]]

    # Terminal conditions. Priority: stoppage > zone_exit > team_a (zone exit before possession).
    x = scan["x"].astype(float)
    sn = scan["sign_negative"]
    is_stoppage = scan["event_type"].isin(STOPPAGE)
    is_zone_exit = x.notna() & ((sn & (x > -BLUE_LINE)) | (~sn & (x < BLUE_LINE)))
    is_team_a = (
        scan["event_type"].isin(POSSESSION_EVENTS)
        & scan["team_id"].notna()
        & (scan["team_id"] == scan["pressing_team_id"])
    )
    scan["is_terminal"] = is_stoppage | is_zone_exit | is_team_a

    terminal = scan[scan["is_terminal"]].copy()
    first_term = terminal.loc[terminal.groupby(["game_id", "sequence_id", "sl_event_id_dumpin"])["sl_event_id"].idxmin()].copy()

    # Compute terminal booleans on first_term (avoids index alignment after merges)
    is_stoppage_ft = first_term["event_type"].isin(STOPPAGE)
    x_ft = first_term["x"].astype(float)
    sn_ft = first_term["sign_negative"]
    is_zone_exit_ft = x_ft.notna() & ((sn_ft & (x_ft > -BLUE_LINE)) | (~sn_ft & (x_ft < BLUE_LINE)))
    first_term["outcome"] = np.where(
        is_stoppage_ft,
        "failure",
        np.where(is_zone_exit_ft, "failure", "success"),
    )
    # Need defending_team_id for penalty outcome
    first_term = first_term.merge(
        starts[["game_id", "sequence_id", "sl_event_id_dumpin", "defending_team_id"]],
        on=["game_id", "sequence_id", "sl_event_id_dumpin"],
        how="left",
    )
    # Penalty: pressing team took penalty -> failure; defending team (possession) took penalty -> success
    is_penalty = first_term["event_type"] == "penalty"
    penalty_on_pressing = is_penalty & (first_term["team_id"] == first_term["pressing_team_id"])
    penalty_on_defending = is_penalty & (first_term["team_id"] == first_term["defending_team_id"])
    first_term.loc[penalty_on_pressing, "outcome"] = "failure"
    first_term.loc[penalty_on_defending, "outcome"] = "success"
    first_term["y"] = (first_term["outcome"] == "success").astype(int)
    # Recompute booleans after merge so they align with current first_term
    is_stoppage_ft = first_term["event_type"].isin(STOPPAGE)
    x_ft = first_term["x"].astype(float)
    is_zone_exit_ft = x_ft.notna() & (
        (first_term["sign_negative"] & (x_ft > -BLUE_LINE))
        | (~first_term["sign_negative"] & (x_ft < BLUE_LINE))
    )
    first_term["terminal_event_type"] = np.where(
        is_stoppage_ft,
        first_term["event_type"],
        np.where(is_zone_exit_ft, "zone_exit", np.where(first_term["event_type"] == "lpr", "lpr", "possession")),
    )

    # Drop period-end whistles (do not count as success or failure)
    max_pt = events.groupby(["game_id", "period"])["period_time"].max().reset_index().rename(columns={"period_time": "max_period_time"})
    first_term = first_term.merge(max_pt, on=["game_id", "period"], how="left")
    period_end_whistle = (
        (first_term["event_type"] == "whistle")
        & (first_term["period_time"] >= first_term["max_period_time"] - 1.0)
    )
    first_term = first_term.loc[~period_end_whistle].drop(columns=["max_period_time"])

    fc = (
        first_term
        .merge(
            starts[["game_id", "sequence_id", "sl_event_id_dumpin", "period", "period_time", "dumpin_detail", "lpr_detail", "puck_x_at_start", "puck_y_at_start"]],
            on=["game_id", "sequence_id", "sl_event_id_dumpin"],
            how="left",
        )
        .rename(columns={"sl_event_id": "sl_event_id_end"})
        [["game_id", "period", "period_time", "sequence_id", "sl_event_id_start", "sl_event_id_end", "pressing_team_id", "defending_team_id", "dumpin_detail", "lpr_detail", "puck_x_at_start", "puck_y_at_start", "sign_negative", "outcome", "y", "terminal_event_type"]]
    )
    fc.insert(0, "fc_sequence_id", range(len(fc)))
    return fc


def get_forecheck_events(events: pd.DataFrame, forechecks: pd.DataFrame) -> pd.DataFrame:
    """Events within [sl_event_id_start, sl_event_id_end] for each forecheck."""
    fc = forechecks[["game_id", "sequence_id", "sl_event_id_start", "sl_event_id_end", "fc_sequence_id"]]
    out = events.merge(fc, on=["game_id", "sequence_id"], how="inner")
    out = out[(out["sl_event_id"] >= out["sl_event_id_start"]) & (out["sl_event_id"] <= out["sl_event_id_end"])]
    return out.drop(columns=["sl_event_id_start", "sl_event_id_end"]).reset_index(drop=True)


def get_forecheck_tracking(tracking: pd.DataFrame, forecheck_events: pd.DataFrame) -> pd.DataFrame:
    """Tracking for (game_id, sl_event_id) pairs in forecheck_events."""
    keys = forecheck_events[["game_id", "sl_event_id", "fc_sequence_id"]].drop_duplicates()
    return tracking.merge(keys, on=["game_id", "sl_event_id"], how="inner")


def flip_xy_for_negative_x(
    forechecks: pd.DataFrame,
    events: pd.DataFrame,
    tracking: pd.DataFrame,
    x_threshold: float = -25,
) -> None:
    """For plays where puck_x_at_start < x_threshold, negate x and y so all forechecks share the same orientation."""
    flip_mask = forechecks["puck_x_at_start"].astype(float) < x_threshold
    flip_fc_ids = forechecks.loc[flip_mask, "fc_sequence_id"].values
    fc_flip = forechecks["fc_sequence_id"].isin(flip_fc_ids)

    # Forechecks: negate puck_x_at_start and puck_y_at_start
    forechecks.loc[fc_flip, "puck_x_at_start"] = -forechecks.loc[fc_flip, "puck_x_at_start"].astype(float)
    forechecks.loc[fc_flip, "puck_y_at_start"] = -forechecks.loc[fc_flip, "puck_y_at_start"].astype(float)
    forechecks.loc[fc_flip, "sign_negative"] = forechecks.loc[fc_flip, "puck_x_at_start"].astype(float) < 0

    # Events: negate x and y
    if "fc_sequence_id" in events.columns and "x" in events.columns and "y" in events.columns:
        ev_flip = events["fc_sequence_id"].isin(flip_fc_ids)
        events.loc[ev_flip, "x"] = -events.loc[ev_flip, "x"].astype(float)
        events.loc[ev_flip, "y"] = -events.loc[ev_flip, "y"].astype(float)

    # Tracking: negate tracking_x, tracking_y, and velocities if present
    if "fc_sequence_id" in tracking.columns:
        tr_flip = tracking["fc_sequence_id"].isin(flip_fc_ids)
        if "tracking_x" in tracking.columns and "tracking_y" in tracking.columns:
            tracking.loc[tr_flip, "tracking_x"] = -tracking.loc[tr_flip, "tracking_x"].astype(float)
            tracking.loc[tr_flip, "tracking_y"] = -tracking.loc[tr_flip, "tracking_y"].astype(float)
        for vx, vy in [("tracking_vel_x", "tracking_vel_y")]:
            if vx in tracking.columns and vy in tracking.columns:
                tracking.loc[tr_flip, vx] = -tracking.loc[tr_flip, vx].astype(float)
                tracking.loc[tr_flip, vy] = -tracking.loc[tr_flip, vy].astype(float)


############
### MAIN ###
############

def main() -> None:
    events = load_events(DATA_DIR)
    tracking = load_tracking(DATA_DIR)

    # Build forecheck sequences, extract their events and tracking, write to parquet.
    forechecks = build_forecheck_sequences(events)
    events_with_fc = get_forecheck_events(events, forechecks)
    tracking_fc = get_forecheck_tracking(tracking, events_with_fc)

    # For plays where x < -25, flip x and y so all forechecks share the same orientation.
    flip_xy_for_negative_x(forechecks, events_with_fc, tracking_fc, x_threshold=-25)

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
