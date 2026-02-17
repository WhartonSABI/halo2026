#!/usr/bin/env python3
"""Build per-time-step hazard features for forecheck sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = PROCESSED_DIR / "hazard_features.parquet"

# Lane blocking constants (snapshot-based interceptability proxy)
LANE_RADIUS_FT = 3.0
PUCK_SPEED_FTPS = 45.0
DEFENDER_FLOOR_SPEED_FTPS = 8.0
DEFENDER_REACTION_S = 0.10
CENTER_LANE_Y_MAX = 15.0
MAX_OUTLET_DIST_FILL = 200.0


def _period_time_to_seconds(period_time) -> float:
    if pd.isna(period_time):
        return np.nan
    if isinstance(period_time, str) and ":" in period_time:
        mm, ss = period_time.split(":")
        return int(mm) * 60 + float(ss)
    return float(period_time)


def _polar_xy(x: float, y: float) -> tuple[float, float, float]:
    r = float(np.hypot(x, y))
    if r == 0:
        return 0.0, 0.0, 1.0
    return r, float(y / r), float(x / r)


def _radial_closing_speed(actor_xy, actor_v, target_xy) -> float:
    rel = np.array(target_xy, dtype=float) - np.array(actor_xy, dtype=float)
    dist = np.linalg.norm(rel)
    if dist == 0:
        return 0.0
    unit = rel / dist
    return float(np.dot(np.array(actor_v, dtype=float), unit))


def _lane_margin_seconds(passer_xy, recv_xy, def_xy, def_v) -> float:
    a = np.array(passer_xy, dtype=float)
    b = np.array(recv_xy, dtype=float)
    d = np.array(def_xy, dtype=float)
    v = np.array(def_v, dtype=float)

    ab = b - a
    lane_len = np.linalg.norm(ab)
    if lane_len == 0:
        return np.inf

    ab_hat = ab / lane_len
    s_along = float(np.clip(np.dot(d - a, ab_hat), 0.0, lane_len))
    closest = a + s_along * ab_hat

    perp = float(np.linalg.norm(d - closest))
    d_eff = max(0.0, perp - LANE_RADIUS_FT)

    to_lane = closest - d
    to_lane_norm = np.linalg.norm(to_lane)
    if to_lane_norm > 0:
        v_proj = float(np.dot(v, to_lane / to_lane_norm))
    else:
        v_proj = float(np.linalg.norm(v))
    v_eff = max(DEFENDER_FLOOR_SPEED_FTPS, v_proj)

    t_def = d_eff / max(v_eff, 1e-6) + DEFENDER_REACTION_S
    t_puck = (s_along / lane_len) * lane_len / PUCK_SPEED_FTPS
    return t_def - t_puck


def _build_sequence_level_controls(forechecks: pd.DataFrame, events: pd.DataFrame, stints: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    starts = forechecks[["fc_sequence_id", "game_id", "sl_event_id_start", "pressing_team_id", "puck_x_at_start", "puck_y_at_start"]].copy()
    start_events = events[["game_id", "sl_event_id", "game_stint"]].drop_duplicates()
    starts = starts.merge(start_events, left_on=["game_id", "sl_event_id_start"], right_on=["game_id", "sl_event_id"], how="left")

    stint_cols = ["game_id", "game_stint", "n_home_skaters", "n_away_skaters", "home_score", "away_score"]
    starts = starts.merge(stints[stint_cols].drop_duplicates(), on=["game_id", "game_stint"], how="left")
    starts = starts.merge(games[["game_id", "home_team_id"]], on="game_id", how="left")

    starts["home_rink"] = (starts["pressing_team_id"] == starts["home_team_id"]).astype(int)
    is_pressing_home = starts["home_rink"] == 1
    starts["pressing_score"] = np.where(is_pressing_home, starts["home_score"], starts["away_score"])
    starts["opp_score"] = np.where(is_pressing_home, starts["away_score"], starts["home_score"])
    starts["score_diff"] = starts["pressing_score"] - starts["opp_score"]
    starts["score_diff_bin"] = np.select(
        [starts["score_diff"] < 0, starts["score_diff"] == 0, starts["score_diff"] > 0],
        ["trailing", "tied", "leading"],
        default="unknown",
    )

    starts["manpower_state"] = starts["n_home_skaters"].astype("Int64").astype(str) + "v" + starts["n_away_skaters"].astype("Int64").astype(str)

    pr, psin, pcos = zip(*starts.apply(lambda r: _polar_xy(r["puck_x_at_start"], r["puck_y_at_start"]), axis=1))
    starts["puck_start_r"] = pr
    starts["puck_start_sinθ"] = psin
    starts["puck_start_cosθ"] = pcos

    return starts[[
        "fc_sequence_id",
        "manpower_state",
        "home_rink",
        "score_diff_bin",
        "puck_start_r",
        "puck_start_sinθ",
        "puck_start_cosθ",
    ]]


def build_hazard_rows(max_frames: int | None = None) -> pd.DataFrame:
    forechecks = pd.read_parquet(PROCESSED_DIR / "forechecks.parquet")
    fc_events = pd.read_parquet(PROCESSED_DIR / "forecheck_events.parquet")
    tracking = pd.read_parquet(PROCESSED_DIR / "forecheck_tracking.parquet")
    events = pd.read_parquet(RAW_DIR / "events.parquet")
    stints = pd.read_parquet(RAW_DIR / "stints.parquet")
    games = pd.read_parquet(RAW_DIR / "games.parquet")

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
    rows = rows.sort_values(["fc_sequence_id", "sl_event_id"]).reset_index(drop=True)
    if max_frames is not None:
        rows = rows.head(max_frames).copy()

    tracking_index = tracking.groupby(["fc_sequence_id", "sl_event_id"])
    out_records = []

    for r in rows.itertuples(index=False):
        key = (r.fc_sequence_id, r.sl_event_id)
        try:
            frame = tracking_index.get_group(key).copy()
        except KeyError:
            continue

        puck_rows = frame[frame["player_id"].isna()].copy()
        if not puck_rows.empty:
            if np.isfinite(r.x) and np.isfinite(r.y):
                d2 = (puck_rows["tracking_x"] - r.x) ** 2 + (puck_rows["tracking_y"] - r.y) ** 2
                puck = puck_rows.loc[d2.idxmin()]
            else:
                puck = puck_rows.iloc[0]
            puck_xy = (float(puck["tracking_x"]), float(puck["tracking_y"]))
            puck_v = (float(puck["tracking_vel_x"]), float(puck["tracking_vel_y"]))
        else:
            puck_xy = (float(r.x), float(r.y)) if np.isfinite(r.x) and np.isfinite(r.y) else (0.0, 0.0)
            puck_v = (0.0, 0.0)

        puck_r, puck_sin, puck_cos = _polar_xy(*puck_xy)
        puck_speed = float(np.hypot(*puck_v))

        forecheckers = frame[frame["team_id"] == r.pressing_team_id].copy()
        defenders = frame[frame["team_id"] == r.defending_team_id].copy()
        skaters = frame[frame["player_id"].notna()].copy()

        forecheckers["dist_to_puck"] = np.hypot(forecheckers["tracking_x"] - puck_xy[0], forecheckers["tracking_y"] - puck_xy[1])
        forecheckers = forecheckers.sort_values("dist_to_puck").head(5).reset_index(drop=True)

        feat = {
            "fc_sequence_id": r.fc_sequence_id,
            "sl_event_id": r.sl_event_id,
            "event_t": int(r.event_t),
            "time_since_start_s": float(r.time_since_start_s),
            "time_since_start_bin": int(r.time_since_start_bin),
            "puck_r": puck_r,
            "puck_sinθ": puck_sin,
            "puck_cosθ": puck_cos,
            "puck_speed": puck_speed,
        }

        fi_rows = {}
        for i in range(1, 6):
            if i <= len(forecheckers):
                p = forecheckers.iloc[i - 1]
                px, py = float(p["tracking_x"]), float(p["tracking_y"])
                pv = (float(p["tracking_vel_x"]), float(p["tracking_vel_y"]))
                relx, rely = px - puck_xy[0], py - puck_xy[1]
                dist = float(np.hypot(relx, rely))
                sinv = 0.0 if dist == 0 else rely / dist
                cosv = 1.0 if dist == 0 else relx / dist
                vr = _radial_closing_speed((px, py), pv, puck_xy)
                feat[f"F{i}_r"] = dist
                feat[f"F{i}_sinθ"] = sinv
                feat[f"F{i}_cosθ"] = cosv
                feat[f"F{i}_vr_puck"] = vr
                fi_rows[i] = p
            else:
                feat[f"F{i}_r"] = np.nan
                feat[f"F{i}_sinθ"] = np.nan
                feat[f"F{i}_cosθ"] = np.nan
                feat[f"F{i}_vr_puck"] = np.nan

        opp_skaters = skaters[skaters["team_id"] != r.pressing_team_id]
        for i in range(2, 6):
            if i in fi_rows and not opp_skaters.empty:
                p = fi_rows[i]
                d2 = (opp_skaters["tracking_x"] - p["tracking_x"]) ** 2 + (opp_skaters["tracking_y"] - p["tracking_y"]) ** 2
                nearest = opp_skaters.loc[d2.idxmin()]
                px, py = float(p["tracking_x"]), float(p["tracking_y"])
                pv = (float(p["tracking_vel_x"]), float(p["tracking_vel_y"]))
                nx, ny = float(nearest["tracking_x"]), float(nearest["tracking_y"])
                feat[f"F{i}_r_nearestOpp"] = float(np.hypot(px - nx, py - ny))
                feat[f"F{i}_vr_nearestOpp"] = _radial_closing_speed((px, py), pv, (nx, ny))
            else:
                feat[f"F{i}_r_nearestOpp"] = np.nan
                feat[f"F{i}_vr_nearestOpp"] = np.nan

        # Passing lanes and outlet availability
        poss_team = r.team_id
        carrier_id = r.player_id
        if pd.notna(poss_team):
            teammates = skaters[(skaters["team_id"] == poss_team)].copy()
            if pd.notna(carrier_id):
                carrier_rows = teammates[teammates["player_id"] == carrier_id]
            else:
                carrier_rows = pd.DataFrame()
            if carrier_rows.empty:
                passer_xy = puck_xy
            else:
                passer_xy = (float(carrier_rows.iloc[0]["tracking_x"]), float(carrier_rows.iloc[0]["tracking_y"]))
            candidates = teammates[teammates["player_id"] != carrier_id].copy() if pd.notna(carrier_id) else teammates.copy()
        else:
            passer_xy = puck_xy
            candidates = pd.DataFrame()

        lane_rows = []
        for c in candidates.itertuples(index=False):
            recv_xy = (float(c.tracking_x), float(c.tracking_y))
            dist = float(np.hypot(recv_xy[0] - passer_xy[0], recv_xy[1] - passer_xy[1]))
            center = abs(recv_xy[1]) <= CENTER_LANE_Y_MAX
            if dist == 0:
                continue
            best_margin = np.inf
            best_def_player = None
            for fd in forecheckers.itertuples(index=False):
                m = _lane_margin_seconds(
                    passer_xy,
                    recv_xy,
                    (float(fd.tracking_x), float(fd.tracking_y)),
                    (float(fd.tracking_vel_x), float(fd.tracking_vel_y)),
                )
                if m < best_margin:
                    best_margin = m
                    best_def_player = fd.player_id
            lane_rows.append({
                "receiver_id": c.player_id,
                "distance": dist,
                "center": center,
                "blocked": int(best_margin < 0),
                "severity": max(0.0, -best_margin),
                "blocker_player_id": best_def_player,
            })

        lanes = pd.DataFrame(lane_rows)
        feat["outlet_candidate_count"] = int(len(lanes))
        if lanes.empty:
            feat["unblocked_outlet_count"] = 0
            feat["center_open"] = 0
            unblocked_d = []
        else:
            unblocked = lanes[lanes["blocked"] == 0].sort_values("distance")
            feat["unblocked_outlet_count"] = int(len(unblocked))
            feat["center_open"] = int(((lanes["center"] == 1) & (lanes["blocked"] == 0)).any())
            unblocked_d = unblocked["distance"].tolist()

        for j in range(1, 4):
            feat[f"unblocked_outlet_d{j}"] = float(unblocked_d[j - 1]) if len(unblocked_d) >= j else MAX_OUTLET_DIST_FILL

        for i in range(1, 6):
            feat[f"F{i}_block_any"] = 0
            feat[f"F{i}_block_severity"] = 0.0
            feat[f"F{i}_block_center_any"] = 0
            feat[f"F{i}_block_center_severity"] = 0.0

        if not lanes.empty:
            player_to_rank = {row.player_id: rank for rank, row in fi_rows.items()}
            for player_id, grp in lanes.groupby("blocker_player_id"):
                rank = player_to_rank.get(player_id)
                if rank is None:
                    continue
                blocked_grp = grp[grp["blocked"] == 1]
                if blocked_grp.empty:
                    continue
                best_any = float(blocked_grp["severity"].max())
                feat[f"F{rank}_block_any"] = 1
                feat[f"F{rank}_block_severity"] = best_any

                blocked_center = blocked_grp[blocked_grp["center"] == 1]
                if not blocked_center.empty:
                    feat[f"F{rank}_block_center_any"] = 1
                    feat[f"F{rank}_block_center_severity"] = float(blocked_center["severity"].max())

        out_records.append(feat)

    out = pd.DataFrame(out_records)
    out = out.merge(seq_controls, on="fc_sequence_id", how="left")
    out = out.sort_values(["fc_sequence_id", "sl_event_id"]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build forecheck hazard features.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on number of time-step rows for quick validation runs.")
    parser.add_argument("--output", type=Path, default=OUT_PATH, help="Output parquet path.")
    args = parser.parse_args()

    features = build_hazard_rows(max_frames=args.max_frames)
    features.to_parquet(args.output, index=False)
    print(f"Wrote {len(features):,} rows to {args.output}")


if __name__ == "__main__":
    main()
