#!/usr/bin/env python3
"""Naive player press-value allocation from terminal forecheck frames.

Implements two simple allocation schemes on top of a difficulty-adjusted
sequence press value:
1) participation: equal split among present F1..F5 players at terminal frame
2) distance: weighted split among present F1..F5 by inverse distance to carrier

Outputs:
- data/processed/player_press_value_participation.csv
- data/processed/player_press_value_distance.csv
- data/processed/terminal_press_value.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORECHECKS_PATH = PROJECT_ROOT / "data" / "processed" / "forechecks.parquet"
HAZARD_PATH = PROJECT_ROOT / "data" / "processed" / "hazard_features.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "processed"


def build_terminal_table() -> pd.DataFrame:
    forechecks = pd.read_parquet(FORECHECKS_PATH)
    hazard = pd.read_parquet(HAZARD_PATH)

    seq_end = forechecks[["fc_sequence_id", "sl_event_id_end", "y"]].rename(columns={"y": "sequence_success"})
    terminal = hazard.merge(seq_end, on="fc_sequence_id", how="inner")
    terminal = terminal.loc[terminal["sl_event_id"] == terminal["sl_event_id_end"]].copy()

    # Difficulty context from sequence start location.
    terminal["x_bin"] = (terminal["puck_start_x"].astype(float) // 10).astype("Int64")
    terminal["y_bin"] = (terminal["puck_start_y"].astype(float).abs() // 10).astype("Int64")

    grp = ["manpower_state", "score_diff_bin", "x_bin", "y_bin"]
    baseline = terminal.groupby(grp, dropna=False)["sequence_success"].mean().rename("p_success_baseline")
    terminal = terminal.merge(baseline, on=grp, how="left")

    overall_success = float(terminal["sequence_success"].mean())
    baseline_filled = terminal["p_success_baseline"].fillna(overall_success)

    # press_value: only successful presses get positive value, scaled by difficulty.
    terminal["press_value"] = np.where(terminal["sequence_success"].eq(1), 1.0 - baseline_filled, 0.0)
    return terminal


def allocate_participation(terminal: pd.DataFrame) -> pd.DataFrame:
    t = terminal.copy()
    slot_ids = [f"F{i}_id" for i in range(1, 6)]

    present = np.zeros(len(t), dtype=float)
    for sid in slot_ids:
        present += t[sid].notna().astype(float)
    present = np.where(present > 0, present, 1.0)

    parts = []
    for sid in slot_ids:
        tmp = t[[sid, "sequence_success", "press_value"]].copy()
        tmp = tmp.dropna(subset=[sid])
        tmp["player_id"] = tmp[sid]
        tmp["share"] = 1.0 / present[tmp.index]
        tmp["player_press_value"] = np.where(
            tmp["sequence_success"].eq(1),
            tmp["press_value"] * tmp["share"],
            0.0,
        )
        parts.append(tmp[["player_id", "player_press_value"]])

    out = (
        pd.concat(parts, ignore_index=True)
        .groupby("player_id", as_index=False)["player_press_value"]
        .sum()
        .rename(columns={"player_press_value": "press_value_total_participation"})
        .sort_values("press_value_total_participation", ascending=False)
    )
    return out


def allocate_distance(terminal: pd.DataFrame, eps: float = 1e-3) -> pd.DataFrame:
    t = terminal.copy()

    for i in range(1, 6):
        sid = f"F{i}_id"
        rcol = f"F{i}_r"
        wcol = f"w{i}"

        r = t[rcol].astype(float)
        w = 1.0 / (r.clip(lower=eps))
        t[wcol] = np.where(t[sid].notna(), w, 0.0)

    w_cols = [f"w{i}" for i in range(1, 6)]
    t["w_sum"] = t[w_cols].sum(axis=1)
    t["w_sum"] = np.where(t["w_sum"] > 0, t["w_sum"], 1.0)

    parts = []
    for i in range(1, 6):
        sid = f"F{i}_id"
        share = t[f"w{i}"] / t["w_sum"]

        tmp = t[[sid, "sequence_success", "press_value"]].copy()
        tmp = tmp.dropna(subset=[sid])
        tmp["player_id"] = tmp[sid]
        tmp["share"] = share[tmp.index]
        tmp["player_press_value"] = np.where(
            tmp["sequence_success"].eq(1),
            tmp["press_value"] * tmp["share"],
            0.0,
        )
        parts.append(tmp[["player_id", "player_press_value"]])

    out = (
        pd.concat(parts, ignore_index=True)
        .groupby("player_id", as_index=False)["player_press_value"]
        .sum()
        .rename(columns={"player_press_value": "press_value_total_distance"})
        .sort_values("press_value_total_distance", ascending=False)
    )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    terminal = build_terminal_table()
    participation = allocate_participation(terminal)
    distance = allocate_distance(terminal)

    terminal_path = OUT_DIR / "terminal_press_value.parquet"
    part_path = OUT_DIR / "player_press_value_participation.csv"
    dist_path = OUT_DIR / "player_press_value_distance.csv"

    terminal.to_parquet(terminal_path, index=False)
    participation.to_csv(part_path, index=False)
    distance.to_csv(dist_path, index=False)

    print(f"Saved: {terminal_path}")
    print(f"Saved: {part_path}")
    print(f"Saved: {dist_path}")
    print("\nTop participation credits:")
    print(participation.head(10).to_string(index=False))
    print("\nTop distance credits:")
    print(distance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
