#!/usr/bin/env python3
"""Simple player attribution from terminal forecheck frames.

Implements two allocation schemes:
1) participation (APM-style): equal split of total value among pressing-team skaters
   on ice at terminal event. No puck-location adjustment, no decomposition.
   total_recovery_value = observed_recovery - avg(recovery).

2) distance: weighted by inverse distance to carrier at terminal moment.
   F1..F5 use tracked distance; unseen participants get fallback.
   No decomposition—just total_recovery_value allocated by 1/r.

Outputs:
- data/processed/terminal_recovery_value.parquet
- data/results/participation.csv
- data/results/distance.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORECHECKS_PATH = PROJECT_ROOT / "data" / "processed" / "forechecks.parquet"
HAZARD_PATH = PROJECT_ROOT / "data" / "processed" / "hazard_features.parquet"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def _participants_from_stints(
    forechecks: pd.DataFrame,
    events: pd.DataFrame,
    stints: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    """Pressing-team skaters on ice at terminal event. Uses stints (game_stint) only—no tracking required."""
    term_events = forechecks[["fc_sequence_id", "game_id", "sl_event_id_end", "pressing_team_id"]].merge(
        events[["game_id", "sl_event_id", "game_stint"]].drop_duplicates(),
        left_on=["game_id", "sl_event_id_end"],
        right_on=["game_id", "sl_event_id"],
        how="inner",
    )[["fc_sequence_id", "game_id", "game_stint", "pressing_team_id"]]

    # Exclude goalies from participants
    pos_col = "primary_position" if "primary_position" in players.columns else "position"
    if pos_col in players.columns:
        skater_ids = set(players.loc[~players[pos_col].isin({"G"}), "player_id"])
    else:
        skater_ids = None

    stint_players = stints[["game_id", "game_stint", "player_id", "team_id"]].dropna(subset=["player_id"])
    merged = term_events.merge(
        stint_players,
        on=["game_id", "game_stint"],
        how="left",
    )
    merged = merged[merged["team_id"] == merged["pressing_team_id"]]
    if skater_ids is not None:
        merged = merged[merged["player_id"].isin(skater_ids)]

    participants = (
        merged.groupby("fc_sequence_id")["player_id"]
        .apply(lambda x: x.dropna().unique().tolist())
        .reset_index()
        .rename(columns={"player_id": "participant_ids"})
    )
    # Include sequences with no participants (empty list)
    all_seq = term_events[["fc_sequence_id"]].drop_duplicates()
    participants = all_seq.merge(participants, on="fc_sequence_id", how="left")
    participants["participant_ids"] = participants["participant_ids"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    return participants


def build_terminal_table_participation() -> pd.DataFrame:
    """APM-style: total value only. Equal split among participants; no puck location, no decomposition."""
    forechecks = pd.read_parquet(FORECHECKS_PATH)
    events = pd.read_parquet(RAW_DIR / "events.parquet")
    stints = pd.read_parquet(RAW_DIR / "stints.parquet")
    players = pd.read_parquet(RAW_DIR / "players.parquet")

    seq_end = forechecks[["fc_sequence_id", "y"]].rename(columns={"y": "sequence_success"})
    participants = _participants_from_stints(forechecks, events, stints, players)

    terminal = seq_end.merge(participants, on="fc_sequence_id", how="left")
    terminal["participant_ids"] = terminal["participant_ids"].apply(
        lambda x: x if isinstance(x, list) and len(x) > 0 else []
    )

    avg_recovery = float(terminal["sequence_success"].mean())
    terminal["total_recovery_value"] = terminal["sequence_success"].astype(float) - avg_recovery
    return terminal


def build_terminal_table_from_hazard() -> pd.DataFrame:
    """Terminal moment only. Total value = observed - avg; no decomposition. Used for distance allocation."""
    forechecks = pd.read_parquet(FORECHECKS_PATH)
    hazard = pd.read_parquet(HAZARD_PATH)

    seq_end = forechecks[["fc_sequence_id", "sl_event_id_end", "y"]].rename(columns={"y": "sequence_success"})
    terminal = hazard.merge(seq_end, on="fc_sequence_id", how="inner")
    # Keep only the terminal-event row per sequence (F1..F5 distances at resolution moment)
    terminal = terminal.loc[terminal["sl_event_id"] == terminal["sl_event_id_end"]].copy()

    avg_recovery = float(terminal["sequence_success"].mean())
    terminal["total_recovery_value"] = terminal["sequence_success"].astype(float) - avg_recovery
    return terminal


def _build_slot_shares(terminal: pd.DataFrame) -> dict[str, pd.Series]:
    """Equal shares among F1..F5 slots present. Used when allocating by slot (e.g. distance companion)."""
    slot_ids = [f"F{i}_id" for i in range(1, 6)]
    present = np.zeros(len(terminal), dtype=float)
    for sid in slot_ids:
        present += terminal[sid].notna().astype(float)
    present = np.where(present > 0, present, 1.0)
    return {sid: pd.Series(1.0 / present, index=terminal.index) for sid in slot_ids}


def _fallback_distance_ft(manpower_state) -> float:
    """Fallback distance when no F1..F5 observed. Scale by manpower (5v5→50ft, 4v4→55ft, etc)."""
    if pd.isna(manpower_state) or not isinstance(manpower_state, str):
        return 50.0
    parts = manpower_state.lower().split("v")
    if len(parts) != 2:
        return 50.0
    try:
        n = max(int(parts[0].strip()), int(parts[1].strip()))
        return {5: 50.0, 4: 55.0, 3: 60.0}.get(n, 45.0 + n * 2)
    except (ValueError, TypeError):
        return 50.0


def _build_distance_weights_with_unseen(
    terminal: pd.DataFrame,
    participants: pd.DataFrame,
    eps: float = 1e-3,
) -> list[tuple[pd.Series, dict]]:
    """Build distance-weighted shares per row. F1..F5 use 1/r; unseen participants use 1/max(puck_to_center, furthest_observed)."""
    merged = terminal.merge(participants, on="fc_sequence_id", how="left")
    merged["participant_ids"] = merged["participant_ids"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    cx = merged["carrier_x"].astype(float).fillna(0)
    cy = merged["carrier_y"].astype(float).fillna(0)
    d_puck_to_center = np.hypot(cx, cy)

    rows_out = []
    for idx, r in merged.iterrows():
        weights: dict = {}
        observed_distances = []

        # F1..F5: weight by inverse distance to carrier
        for i in range(1, 6):
            sid = f"F{i}_id"
            rcol = f"F{i}_r"
            pid = r[sid]
            if pd.notna(pid):
                dist = float(r[rcol]) if pd.notna(r[rcol]) else np.nan
                if np.isfinite(dist):
                    observed_distances.append(dist)
                weights[pid] = 1.0 / max(dist, eps) if np.isfinite(dist) else 0.0

        # Unseen participants (on ice but not in F1..F5): use max(puck→center, furthest observed)
        furthest_observed = max(observed_distances) if observed_distances else 0.0
        d_to_center = float(d_puck_to_center.loc[idx]) if np.isfinite(d_puck_to_center.loc[idx]) else 50.0
        d_unseen = max(d_to_center, furthest_observed)
        if d_unseen <= 0:
            d_unseen = _fallback_distance_ft(r.get("manpower_state"))
        w_unseen = 1.0 / max(d_unseen, eps)

        seen = set(weights.keys())
        for pid in r["participant_ids"]:
            if pd.isna(pid):
                continue
            if pid not in seen:
                weights[pid] = w_unseen

        total = sum(weights.values())
        if total <= 0:
            continue
        shares = {p: w / total for p, w in weights.items()}  # normalize to sum to 1
        rows_out.append((r, shares))

    return rows_out


def allocate_participation_from_participants(terminal: pd.DataFrame) -> pd.DataFrame:
    """Equal split of total value among participants. APM-style—terminal must have participant_ids."""
    rows = []
    for _, r in terminal.iterrows():
        pids = r["participant_ids"]
        if not pids:
            continue
        n = len(pids)
        share = 1.0 / n
        for pid in pids:
            rows.append(
                {
                    "player_id": pid,
                    "fc_sequence_id": r["fc_sequence_id"],
                    "total_recovery_value_alloc": r["total_recovery_value"] * share,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["player_id", "n_press", "total_recovery_value_participation"]
        )

    out = (
        pd.DataFrame(rows)
        .groupby("player_id", as_index=False)
        .agg(
            n_press=("fc_sequence_id", "nunique"),
            total_recovery_value_participation=("total_recovery_value_alloc", "sum"),
        )
        .sort_values("total_recovery_value_participation", ascending=False)
    )
    return out


def allocate_distance(
    terminal: pd.DataFrame,
    participants: pd.DataFrame,
    eps: float = 1e-3,
) -> pd.DataFrame:
    """Distance-weighted allocation at terminal moment. F1..F5 use 1/r; unseen use fallback. No decomposition."""
    weight_rows = _build_distance_weights_with_unseen(terminal, participants, eps=eps)

    rows = []
    for r, shares in weight_rows:
        for pid, share in shares.items():
            rows.append(
                {
                    "player_id": pid,
                    "fc_sequence_id": r["fc_sequence_id"],
                    "total_recovery_value_alloc": r["total_recovery_value"] * share,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["player_id", "n_press", "total_recovery_value_distance"]
        )

    out = (
        pd.DataFrame(rows)
        .groupby("player_id", as_index=False)
        .agg(
            n_press=("fc_sequence_id", "nunique"),
            total_recovery_value_distance=("total_recovery_value_alloc", "sum"),
        )
        .sort_values("total_recovery_value_distance", ascending=False)
    )
    return out


def _write_clean_csv(
    df: pd.DataFrame,
    value_col: str,
    out_path: Path,
    has_n_press: bool = True,
) -> None:
    """Write clean CSV: player_id, player_name, position, n_press?, total."""
    players_df = pd.read_parquet(RAW_DIR / "players.parquet")
    pid_col = "player_id" if "player_id" in players_df.columns else "id"
    name_col = "player_name" if "player_name" in players_df.columns else "name"
    pos_col = "primary_position" if "primary_position" in players_df.columns else "position"
    merge_cols = {pid_col: "player_id", name_col: "player_name"}
    if pos_col in players_df.columns:
        merge_cols[pos_col] = "position"
    cols = ["player_id", value_col]
    if has_n_press and "n_press" in df.columns:
        cols.insert(1, "n_press")
    out = df[cols].copy()
    out = out.rename(columns={value_col: "total"})
    if "n_press" in out.columns:
        out["total_per_press"] = np.where(out["n_press"] > 0, out["total"] / out["n_press"], np.nan)
    out = out.sort_values("total", ascending=False).reset_index(drop=True)
    out = out.merge(
        players_df[[c for c in [pid_col, name_col, pos_col] if c in players_df.columns]].rename(columns=merge_cols),
        on="player_id",
        how="left",
    )
    if "n_presses" in out.columns and "n_press" not in out.columns:
        out = out.rename(columns={"n_presses": "n_press"})
    out_cols = ["player_id", "player_name", "position", "n_press", "total", "total_per_press"]
    out = out[[c for c in out_cols if c in out.columns]]
    out.to_csv(out_path, index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Participation attribution: stints only, no tracking ----
    terminal_participation = build_terminal_table_participation()
    participation = allocate_participation_from_participants(terminal_participation)

    # ---- Distance attribution: requires hazard_features (tracking pipeline) ----
    try:
        forechecks = pd.read_parquet(FORECHECKS_PATH)
        events = pd.read_parquet(RAW_DIR / "events.parquet")
        stints = pd.read_parquet(RAW_DIR / "stints.parquet")
        players = pd.read_parquet(RAW_DIR / "players.parquet")
        participants = _participants_from_stints(forechecks, events, stints, players)

        terminal_hazard = build_terminal_table_from_hazard()
        distance = allocate_distance(terminal_hazard, participants)
        terminal_path = OUT_DIR / "terminal_recovery_value.parquet"
        terminal_hazard.to_parquet(terminal_path, index=False)
        dist_path = RESULTS_DIR / "distance.csv"
        _write_clean_csv(distance, "total_recovery_value_distance", dist_path)
        print(f"Saved: {terminal_path}, {dist_path}")
        print("Top distance credits:")
        print(pd.read_csv(dist_path).head(10).to_string(index=False))
    except FileNotFoundError:
        print("Skipping distance allocation: hazard_features.parquet not found (run 02_features.py first)")

    part_path = RESULTS_DIR / "participation.csv"
    _write_clean_csv(participation, "total_recovery_value_participation", part_path)
    print(f"Saved: {part_path}")
    print("\nTop participation credits:")
    print(pd.read_csv(part_path).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
