#!/usr/bin/env python3
"""
Build forecheck (pressing) sequences from hockey event data.

Logic:
  1. Start: dump-in → LPR+ under pressure (Team B recovers).
  2. Success: Team A gets possession before puck exits zone.
  3. Loss: puck exits or stoppage before Team A gets it.
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
STOPPAGE = ("whistle", "goal", "icing", "offside")
POSSESSION_EVENTS = ("lpr", "reception", "carry", "pass", "dumpin", "shot")
BLUE_LINE = 25


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
        [["game_id", "sequence_id", "sl_event_id", "team_id", "x", "detail"]]
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
    starts["sign_negative"] = starts["puck_x_at_start"] < 0

    # 3. Vectorized scan: for each start, find first terminal event (stoppage, zone exit, or Team A possession)
    ev_cols = ["game_id", "sequence_id", "sl_event_id", "event_type", "team_id", "x"]
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
    first_term = terminal.loc[terminal.groupby(["game_id", "sequence_id", "sl_event_id_dumpin"])["sl_event_id"].idxmin()]

    # Effective condition: stoppage overrides zone_exit overrides team_a
    first_term["outcome"] = np.where(
        is_stoppage.loc[first_term.index],
        "failure",
        np.where(is_zone_exit.loc[first_term.index], "failure", "success"),
    )
    first_term["y"] = (first_term["outcome"] == "success").astype(int)
    first_term["terminal_event_type"] = np.where(
        is_stoppage.loc[first_term.index],
        first_term["event_type"],
        np.where(is_zone_exit.loc[first_term.index], "zone_exit", np.where(first_term["event_type"] == "lpr", "lpr", "possession")),
    )

    fc = (
        first_term
        .merge(
            starts[["game_id", "sequence_id", "sl_event_id_dumpin", "period", "period_time", "defending_team_id", "dumpin_detail", "lpr_detail", "puck_x_at_start"]],
            on=["game_id", "sequence_id", "sl_event_id_dumpin"],
            how="left",
        )
        .rename(columns={"sl_event_id": "sl_event_id_end"})
        [["game_id", "period", "period_time", "sequence_id", "sl_event_id_start", "sl_event_id_end", "pressing_team_id", "defending_team_id", "dumpin_detail", "lpr_detail", "puck_x_at_start", "sign_negative", "outcome", "y", "terminal_event_type"]]
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
