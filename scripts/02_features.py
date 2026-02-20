#!/usr/bin/env python3
"""Build per-time-step hazard features for forecheck sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = PROCESSED_DIR / "hazard_features.parquet"

# Lane blocking constants (snapshot-based interceptability proxy)
# Width of the intercept zone around the pass path (ft). Defender within this perpendicular
# distance of the lane can intercept. Chosen as plausible (~stick reach)
LANE_RADIUS_FT = 3.0
# Assumed pass speed (ft/s) for time-to-intercept. ~50 ft/s ≈ 34 mph from literature for controlled passes
PUCK_SPEED_FTPS = 50.0
# Minimum closing speed (ft/s) we assign the defender when projecting to the lane. Conservative floor.
DEFENDER_FLOOR_SPEED_FTPS = 8.0
# Base delay (s) before defender reacts. Scaled down when defender is already closing toward the lane.
DEFENDER_REACTION_S = 0.10
# |y| <= this (ft) defines center lane; boards-side lanes have larger |y|
CENTER_LANE_Y_MAX = 15.0
MAX_OUTLET_DIST_FILL = 200.0


def _period_time_to_seconds(period_time) -> float:
    if pd.isna(period_time):
        return np.nan
    if isinstance(period_time, str) and ":" in period_time:
        mm, ss = period_time.split(":")
        return int(mm) * 60 + float(ss)
    return float(period_time)


def _radial_closing_speed(actor_xy, actor_v, target_xy) -> float:
    """Single actor; see _radial_closing_speed_batch for vectorized version."""
    out = _radial_closing_speed_batch(
        np.asarray(actor_xy).reshape(1, 2),
        np.asarray(actor_v).reshape(1, 2),
        np.asarray(target_xy).reshape(1, 2),
    )
    return float(out[0])


def _radial_closing_speed_batch(actor_xy: np.ndarray, actor_v: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    """Vectorized radial closing speed. Shapes (n, 2); returns (n,)."""
    rel = target_xy - actor_xy
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    np.putmask(dist, dist < 1e-9, 1.0)
    unit = rel / dist
    return (actor_v * unit).sum(axis=1)


def _lane_margin_seconds(passer_xy, recv_xy, def_xy, def_v) -> float:
    """Single (passer, recv, def) lane margin. See _lane_margin_batch for logic."""
    margins = _lane_margin_batch(passer_xy, recv_xy, def_xy, def_v)
    return float(margins[0])


def _lane_margin_batch(passer_xy, recv_xy, def_xy: np.ndarray, def_v: np.ndarray) -> np.ndarray:
    """
    Lane blocking: time margin (positive = lane open, negative = blocked).
    Vectorized over defenders. def_xy, def_v shape (n_def, 2); returns (n_def,).
    """
    a = np.asarray(passer_xy, dtype=float)
    b = np.asarray(recv_xy, dtype=float)
    d = np.asarray(def_xy, dtype=float)
    v = np.asarray(def_v, dtype=float)
    if d.ndim == 1:
        d = d.reshape(1, 2)
        v = v.reshape(1, 2)

    ab = b - a
    lane_len = np.linalg.norm(ab)
    if lane_len < 1e-9:
        return np.full(len(d), np.inf)

    ab_hat = ab / lane_len
    s_along = np.clip((d - a) @ ab_hat, 0.0, lane_len)
    closest = a + s_along[:, np.newaxis] * ab_hat

    perp = np.linalg.norm(d - closest, axis=1)
    d_eff = np.maximum(0.0, perp - LANE_RADIUS_FT)

    to_lane = closest - d
    to_lane_norm = np.linalg.norm(to_lane, axis=1, keepdims=True)
    to_lane_norm = np.where(to_lane_norm > 1e-9, to_lane_norm, 1.0)
    v_proj = (v * (to_lane / to_lane_norm)).sum(axis=1)
    v_eff = np.maximum(DEFENDER_FLOOR_SPEED_FTPS, v_proj)
    reaction_s = DEFENDER_REACTION_S / (1 + np.maximum(0.0, v_proj) / DEFENDER_FLOOR_SPEED_FTPS)

    t_def = d_eff / np.maximum(v_eff, 1e-9) + reaction_s
    t_puck = s_along / PUCK_SPEED_FTPS
    return t_def - t_puck


def _build_sequence_level_controls(forechecks: pd.DataFrame, events: pd.DataFrame, stints: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    starts = forechecks[["fc_sequence_id", "game_id", "sl_event_id_start", "pressing_team_id", "puck_x_at_start", "puck_y_at_start"]].copy()
    start_events = events[["game_id", "sl_event_id", "game_stint"]].drop_duplicates()
    starts = starts.merge(start_events, left_on=["game_id", "sl_event_id_start"], right_on=["game_id", "sl_event_id"], how="left")

    stint_cols = ["game_id", "game_stint", "n_home_skaters", "n_away_skaters", "home_score", "away_score"]
    starts = starts.merge(stints[stint_cols].drop_duplicates(), on=["game_id", "game_stint"], how="left")
    starts = starts.merge(games[["game_id", "home_team_id"]], on="game_id", how="left")

    # Pressing team is not always home; this flag is "pressing team == home team".
    starts["pressing_is_home"] = (starts["pressing_team_id"] == starts["home_team_id"]).astype(int)
    is_pressing_home = starts["pressing_is_home"] == 1
    starts["pressing_score"] = np.where(is_pressing_home, starts["home_score"], starts["away_score"])
    starts["opp_score"] = np.where(is_pressing_home, starts["away_score"], starts["home_score"])
    starts["score_diff"] = starts["pressing_score"] - starts["opp_score"]
    starts["score_diff_bin"] = np.select(
        [starts["score_diff"] < 0, starts["score_diff"] == 0, starts["score_diff"] > 0],
        ["trailing", "tied", "leading"],
        default="unknown",
    )

    starts["manpower_state"] = starts["n_home_skaters"].astype("Int64").astype(str) + "v" + starts["n_away_skaters"].astype("Int64").astype(str)

    starts["puck_start_x"] = starts["puck_x_at_start"]
    starts["puck_start_y"] = starts["puck_y_at_start"]

    return starts[[
        "fc_sequence_id",
        "manpower_state",
        "pressing_is_home",
        "score_diff_bin",
        "puck_start_x",
        "puck_start_y",
    ]]


def _process_one_frame(r, frame: pd.DataFrame, skater_ids: set | None = None) -> dict | None:
    """Process a single (fc_sequence_id, sl_event_id) frame; returns feat dict or None if skip."""
    # Carrier = primary actor; puck is with them. Position from event, velocity from tracking.
    carrier_xy = (float(r.x), float(r.y)) if np.isfinite(r.x) and np.isfinite(r.y) else (0.0, 0.0)
    carrier_frame = frame[frame["player_id"] == r.player_id] if pd.notna(r.player_id) else pd.DataFrame()
    if not carrier_frame.empty:
        c = carrier_frame.iloc[0]
        carrier_v = (float(c["tracking_vel_x"]), float(c["tracking_vel_y"]))
    else:
        carrier_v = (0.0, 0.0)
    carrier_speed = float(np.hypot(*carrier_v))

    forecheckers = frame[frame["team_id"] == r.pressing_team_id].copy()
    if skater_ids is not None:
        forecheckers = forecheckers[forecheckers["player_id"].isin(skater_ids)]
    forecheckers["dist_to_carrier"] = np.hypot(forecheckers["tracking_x"] - carrier_xy[0], forecheckers["tracking_y"] - carrier_xy[1])
    forecheckers = forecheckers.sort_values("dist_to_carrier").head(5).reset_index(drop=True)

    skaters = frame[frame["player_id"].notna()]

    feat = {
        "fc_sequence_id": r.fc_sequence_id,
        "sl_event_id": r.sl_event_id,
        "event_t": int(r.event_t),
        "terminal_failure_t": int(r.terminal_failure_t),
        "time_since_start_s": float(r.time_since_start_s),
        "time_since_start_bin": int(r.time_since_start_bin),
        "carrier_id": r.player_id,
        "carrier_x": carrier_xy[0],
        "carrier_y": carrier_xy[1],
        "carrier_speed": carrier_speed,
    }

    n_fc = len(forecheckers)
    fc_xy = forecheckers[["tracking_x", "tracking_y"]].to_numpy(dtype=float)
    fc_v = forecheckers[["tracking_vel_x", "tracking_vel_y"]].to_numpy(dtype=float)
    carrier_xy_arr = np.array(carrier_xy, dtype=float)
    rel_to_carrier = fc_xy - carrier_xy_arr
    dist_to_carrier = np.linalg.norm(rel_to_carrier, axis=1, keepdims=True)
    np.putmask(dist_to_carrier, dist_to_carrier < 1e-9, 1.0)
    unit_to_carrier = rel_to_carrier / dist_to_carrier
    vr_carrier = _radial_closing_speed_batch(fc_xy, fc_v, np.broadcast_to(carrier_xy_arr, (n_fc, 2)))

    for i in range(1, 6):
        if i <= n_fc:
            feat[f"F{i}_id"] = forecheckers.iloc[i - 1]["player_id"]
            feat[f"F{i}_r"] = float(dist_to_carrier[i - 1, 0])
            feat[f"F{i}_vr_carrier"] = float(vr_carrier[i - 1])
            feat[f"F{i}_sinθ"] = float(unit_to_carrier[i - 1, 1])
            feat[f"F{i}_cosθ"] = float(unit_to_carrier[i - 1, 0])
        else:
            feat[f"F{i}_id"] = pd.NA
            feat[f"F{i}_r"] = np.nan
            feat[f"F{i}_vr_carrier"] = np.nan
            feat[f"F{i}_sinθ"] = np.nan
            feat[f"F{i}_cosθ"] = np.nan

    opp_skaters = skaters[skaters["team_id"] != r.pressing_team_id]
    opp_xy = opp_skaters[["tracking_x", "tracking_y"]].to_numpy(dtype=float)
    for i in range(2, 6):
        if i <= n_fc and len(opp_skaters) > 0:
            p_xy = fc_xy[i - 1]
            p_v = fc_v[i - 1]
            d2 = ((opp_xy[:, 0] - p_xy[0]) ** 2 + (opp_xy[:, 1] - p_xy[1]) ** 2)
            nearest_idx = int(np.argmin(d2))
            nx, ny = opp_xy[nearest_idx, 0], opp_xy[nearest_idx, 1]
            feat[f"F{i}_r_nearestOpp"] = float(np.hypot(p_xy[0] - nx, p_xy[1] - ny))
            feat[f"F{i}_vr_nearestOpp"] = float(_radial_closing_speed_batch(
                p_xy.reshape(1, 2), p_v.reshape(1, 2), np.array([[nx, ny]])
            )[0])
        else:
            feat[f"F{i}_r_nearestOpp"] = np.nan
            feat[f"F{i}_vr_nearestOpp"] = np.nan

    fi_rows = {i: forecheckers.iloc[i - 1] for i in range(1, n_fc + 1)}

    poss_team = r.team_id
    carrier_id = r.player_id
    if pd.notna(poss_team):
        teammates = skaters[skaters["team_id"] == poss_team]
        if pd.notna(carrier_id):
            carrier_rows = teammates[teammates["player_id"] == carrier_id]
        else:
            carrier_rows = pd.DataFrame()
        if carrier_rows.empty:
            passer_xy = carrier_xy
        else:
            passer_xy = (float(carrier_rows.iloc[0]["tracking_x"]), float(carrier_rows.iloc[0]["tracking_y"]))
        candidates = teammates[teammates["player_id"] != carrier_id] if pd.notna(carrier_id) else teammates
    else:
        passer_xy = carrier_xy
        candidates = pd.DataFrame()

    def_xy = forecheckers[["tracking_x", "tracking_y"]].to_numpy(dtype=float)
    def_v = forecheckers[["tracking_vel_x", "tracking_vel_y"]].to_numpy(dtype=float)
    def_player_ids = forecheckers["player_id"].to_numpy()

    lane_rows = []
    for c in candidates.itertuples(index=False):
        recv_xy = (float(c.tracking_x), float(c.tracking_y))
        dist = float(np.hypot(recv_xy[0] - passer_xy[0], recv_xy[1] - passer_xy[1]))
        center = abs(recv_xy[1]) <= CENTER_LANE_Y_MAX
        if dist == 0:
            continue
        if len(forecheckers) == 0:
            best_margin = np.inf
            best_def_player = None
        else:
            margins = _lane_margin_batch(passer_xy, recv_xy, def_xy, def_v)
            best_idx = int(np.argmin(margins))
            best_margin = float(margins[best_idx])
            best_def_player = def_player_ids[best_idx]
        lane_rows.append({
            "receiver_id": c.player_id,
            "distance": dist,
            "center": center,
            "blocked": int(best_margin < 0),
            "severity": max(0.0, -best_margin),
            "blocker_player_id": best_def_player,
        })

    n_lanes = len(lane_rows)
    feat["outlet_candidate_count"] = n_lanes
    if n_lanes == 0:
        feat["unblocked_outlet_count"] = 0
        feat["center_open"] = 0
        feat["min_unblocked_outlet_dist"] = MAX_OUTLET_DIST_FILL
    else:
        unblocked = sorted((lr["distance"] for lr in lane_rows if lr["blocked"] == 0))
        n_unblocked = len(unblocked)
        feat["unblocked_outlet_count"] = n_unblocked
        feat["center_open"] = int(any(lr["center"] and lr["blocked"] == 0 for lr in lane_rows))
        feat["min_unblocked_outlet_dist"] = float(unblocked[0]) if n_unblocked > 0 else MAX_OUTLET_DIST_FILL

    blocker_stats = {i: {"severity": 0.0, "center_severity": 0.0} for i in range(1, 6)}
    player_to_rank = {forecheckers.iloc[i - 1]["player_id"]: i for i in range(1, n_fc + 1)}
    for lr in lane_rows:
        if lr["blocked"] == 0:
            continue
        rank = player_to_rank.get(lr["blocker_player_id"])
        if rank is None:
            continue
        s = blocker_stats[rank]
        s["severity"] = max(s["severity"], lr["severity"])
        if lr["center"]:
            s["center_severity"] = max(s["center_severity"], lr["severity"])

    for i in range(1, 6):
        feat[f"F{i}_block_severity"] = blocker_stats[i]["severity"]
        feat[f"F{i}_block_center_severity"] = blocker_stats[i]["center_severity"]

    return feat


def _process_chunk(rows_chunk: pd.DataFrame, tracking_dict: dict, skater_ids: set | None = None) -> list[dict]:
    """Process a chunk of rows; tracking_dict maps (fc_sequence_id, sl_event_id) -> frame DataFrame."""
    out = []
    for r in rows_chunk.itertuples(index=False):
        key = (r.fc_sequence_id, r.sl_event_id)
        frame = tracking_dict.get(key)
        if frame is None:
            continue
        feat = _process_one_frame(r, frame, skater_ids)
        if feat is not None:
            out.append(feat)
    return out


TRACKING_COLUMNS = ["fc_sequence_id", "sl_event_id", "team_id", "player_id", "tracking_x", "tracking_y", "tracking_vel_x", "tracking_vel_y"]


def build_hazard_rows(max_frames: int | None = None, n_jobs: int = 1) -> pd.DataFrame:
    forechecks = pd.read_parquet(PROCESSED_DIR / "forechecks.parquet")
    fc_events = pd.read_parquet(PROCESSED_DIR / "forecheck_events.parquet")
    tracking = pd.read_parquet(PROCESSED_DIR / "forecheck_tracking.parquet", columns=TRACKING_COLUMNS)
    events = pd.read_parquet(RAW_DIR / "events.parquet")
    stints = pd.read_parquet(RAW_DIR / "stints.parquet")
    games = pd.read_parquet(RAW_DIR / "games.parquet")
    players = pd.read_parquet(RAW_DIR / "players.parquet")
    pos_col = "primary_position" if "primary_position" in players.columns else "position"
    skater_ids = set(players.loc[~players[pos_col].isin({"G"}), "player_id"]) if pos_col in players.columns else None

    seq_controls = _build_sequence_level_controls(forechecks, events, stints, games)

    event_meta = fc_events[[
        "fc_sequence_id",
        "game_id",
        "sl_event_id",
        "period",
        "period_time",
        "team_id",
        "player_id",
        "x",
        "y",
    ]].drop_duplicates(["fc_sequence_id", "sl_event_id"])

    seq_end = forechecks[["fc_sequence_id", "sl_event_id_end", "y", "pressing_team_id", "defending_team_id"]].copy()
    seq_end = seq_end.rename(columns={"y": "sequence_success"})

    frame_keys = tracking[["fc_sequence_id", "sl_event_id"]].drop_duplicates()
    rows = frame_keys.merge(event_meta, on=["fc_sequence_id", "sl_event_id"], how="left")
    rows = rows.merge(seq_end, on="fc_sequence_id", how="left")

    start_clock = event_meta.sort_values(["fc_sequence_id", "sl_event_id"]).groupby("fc_sequence_id").first().reset_index()[["fc_sequence_id", "period", "period_time"]]
    start_clock = start_clock.rename(columns={"period": "period_start", "period_time": "period_time_start"})
    rows = rows.merge(start_clock, on="fc_sequence_id", how="left")

    rows["clock_elapsed"] = (rows["period"] - 1) * 1200 + rows["period_time"].map(_period_time_to_seconds)
    rows["clock_start"] = (rows["period_start"] - 1) * 1200 + rows["period_time_start"].map(_period_time_to_seconds)
    rows["time_since_start_s"] = (rows["clock_elapsed"] - rows["clock_start"]).clip(lower=0)
    rows["time_since_start_bin"] = (rows["time_since_start_s"] // 1.0).astype(int)

    rows["event_t"] = ((rows["sequence_success"] == 1) & (rows["sl_event_id"] == rows["sl_event_id_end"])).astype(int)
    rows["terminal_failure_t"] = ((rows["sequence_success"] == 0) & (rows["sl_event_id"] == rows["sl_event_id_end"])).astype(int)
    rows = rows.sort_values(["fc_sequence_id", "sl_event_id"]).reset_index(drop=True)
    if max_frames is not None:
        rows = rows.head(max_frames).copy()

    tracking_dict = {k: v for k, v in tracking.groupby(["fc_sequence_id", "sl_event_id"])}
    del tracking

    n_rows = len(rows)
    if n_rows == 0:
        out = pd.DataFrame()
    elif n_jobs == 1:
        out_records = _process_chunk(rows, tracking_dict, skater_ids)
        out = pd.DataFrame(out_records)
    else:
        n_chunks = max(1, min(abs(n_jobs), n_rows))
        chunk_size = (n_rows + n_chunks - 1) // n_chunks
        chunks = [rows.iloc[i : i + chunk_size] for i in range(0, n_rows, chunk_size)]
        chunk_keys = [set(zip(c["fc_sequence_id"], c["sl_event_id"])) for c in chunks]
        tracking_subdicts = [{k: tracking_dict[k] for k in keys if k in tracking_dict} for keys in chunk_keys]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_chunk)(chunk, td, skater_ids) for chunk, td in zip(chunks, tracking_subdicts)
        )
        out_records = [r for sub in results for r in sub]
        out = pd.DataFrame(out_records)

    if out.empty:
        out = pd.DataFrame()
    out = out.merge(seq_controls, on="fc_sequence_id", how="left")
    out = out.sort_values(["fc_sequence_id", "sl_event_id"]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build forecheck hazard features.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on number of time-step rows for quick validation runs.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs (-1 = all cores, 1 = sequential).")
    parser.add_argument("--output", type=Path, default=OUT_PATH, help="Output parquet path.")
    args = parser.parse_args()

    features = build_hazard_rows(max_frames=args.max_frames, n_jobs=args.n_jobs)
    features.to_parquet(args.output, index=False)
    print(f"Wrote {len(features):,} rows to {args.output}")


if __name__ == "__main__":
    main()
