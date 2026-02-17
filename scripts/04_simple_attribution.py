#!/usr/bin/env python3
"""Simple player attribution from terminal forecheck frames.

Implements two allocation schemes on top of a decomposed forecheck value:
1) participation: equal split among present F1..F5 players at terminal frame
2) distance: weighted split among present F1..F5 by inverse distance to carrier

Value decomposition per sequence:
- start_positioning_value = P(recovery | start state) - avg(recovery)
- execution_value = observed_recovery - P(recovery | start state)
- total_recovery_value = observed_recovery - avg(recovery)

Outputs:
- data/processed/terminal_recovery_value.parquet
- data/processed/player_recovery_value_participation.csv
- data/processed/player_recovery_value_distance.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORECHECKS_PATH = PROJECT_ROOT / "data" / "processed" / "forechecks.parquet"
HAZARD_PATH = PROJECT_ROOT / "data" / "processed" / "hazard_features.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "processed"


VALUE_COLS = [
    "start_positioning_value",
    "execution_value",
    "total_recovery_value",
]


def build_terminal_table() -> pd.DataFrame:
    forechecks = pd.read_parquet(FORECHECKS_PATH)
    hazard = pd.read_parquet(HAZARD_PATH)

    seq_end = forechecks[["fc_sequence_id", "sl_event_id_end", "y"]].rename(columns={"y": "sequence_success"})
    terminal = hazard.merge(seq_end, on="fc_sequence_id", how="inner")
    terminal = terminal.loc[terminal["sl_event_id"] == terminal["sl_event_id_end"]].copy()

    # Start-state context from sequence start location and game state.
    terminal["x_bin"] = (terminal["puck_start_x"].astype(float) // 10).astype("Int64")
    terminal["y_bin"] = (terminal["puck_start_y"].astype(float).abs() // 10).astype("Int64")

    grp = ["manpower_state", "score_diff_bin", "x_bin", "y_bin"]
    baseline = terminal.groupby(grp, dropna=False)["sequence_success"].mean().rename("p_recovery_given_start")
    terminal = terminal.merge(baseline, on=grp, how="left")

    avg_recovery = float(terminal["sequence_success"].mean())
    p_start = terminal["p_recovery_given_start"].fillna(avg_recovery)

    terminal["start_positioning_value"] = p_start - avg_recovery
    terminal["execution_value"] = terminal["sequence_success"].astype(float) - p_start
    terminal["total_recovery_value"] = terminal["sequence_success"].astype(float) - avg_recovery
    return terminal


def _build_slot_shares_participation(terminal: pd.DataFrame) -> dict[str, pd.Series]:
    slot_ids = [f"F{i}_id" for i in range(1, 6)]

    present = np.zeros(len(terminal), dtype=float)
    for sid in slot_ids:
        present += terminal[sid].notna().astype(float)
    present = np.where(present > 0, present, 1.0)

    return {sid: pd.Series(1.0 / present, index=terminal.index) for sid in slot_ids}


def _build_slot_shares_distance(terminal: pd.DataFrame, eps: float = 1e-3) -> dict[str, pd.Series]:
    t = terminal.copy()
    slot_ids = [f"F{i}_id" for i in range(1, 6)]

    for i in range(1, 6):
        sid = f"F{i}_id"
        rcol = f"F{i}_r"
        wcol = f"w{i}"

        r = t[rcol].astype(float)
        w = 1.0 / r.clip(lower=eps)
        t[wcol] = np.where(t[sid].notna(), w, 0.0)

    w_cols = [f"w{i}" for i in range(1, 6)]
    w_sum = t[w_cols].sum(axis=1)
    w_sum = np.where(w_sum > 0, w_sum, 1.0)

    return {sid: t[f"w{i}"] / w_sum for i, sid in enumerate(slot_ids, start=1)}


def _allocate_values(terminal: pd.DataFrame, slot_shares: dict[str, pd.Series], suffix: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for sid, share in slot_shares.items():
        tmp = terminal[[sid] + VALUE_COLS].copy()
        tmp = tmp.dropna(subset=[sid])
        tmp["player_id"] = tmp[sid]

        for col in VALUE_COLS:
            tmp[f"{col}_alloc"] = tmp[col] * share[tmp.index]

        parts.append(
            tmp[
                [
                    "player_id",
                    "start_positioning_value_alloc",
                    "execution_value_alloc",
                    "total_recovery_value_alloc",
                ]
            ]
        )

    out = (
        pd.concat(parts, ignore_index=True)
        .groupby("player_id", as_index=False)
        .sum()
        .rename(
            columns={
                "start_positioning_value_alloc": f"start_positioning_value_{suffix}",
                "execution_value_alloc": f"execution_value_{suffix}",
                "total_recovery_value_alloc": f"total_recovery_value_{suffix}",
            }
        )
        .sort_values(f"total_recovery_value_{suffix}", ascending=False)
    )
    return out


def allocate_participation(terminal: pd.DataFrame) -> pd.DataFrame:
    shares = _build_slot_shares_participation(terminal)
    return _allocate_values(terminal, shares, suffix="participation")


def allocate_distance(terminal: pd.DataFrame, eps: float = 1e-3) -> pd.DataFrame:
    shares = _build_slot_shares_distance(terminal, eps=eps)
    return _allocate_values(terminal, shares, suffix="distance")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    terminal = build_terminal_table()
    participation = allocate_participation(terminal)
    distance = allocate_distance(terminal)

    terminal_path = OUT_DIR / "terminal_recovery_value.parquet"
    part_path = OUT_DIR / "player_recovery_value_participation.csv"
    dist_path = OUT_DIR / "player_recovery_value_distance.csv"

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
