#!/usr/bin/env python3
"""
Pass lane features: distance, angle, and lane obstruction from tracking data.
"""

from pathlib import Path

import numpy as np
import pandas as pd

##############
### CONFIG ###
##############

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_data(data_dir: Path):
    """Load events, stints, games, players, tracking from parquet."""
    return {
        "events": pd.read_parquet(data_dir / "events.parquet"),
        "stints": pd.read_parquet(data_dir / "stints.parquet"),
        "games": pd.read_parquet(data_dir / "games.parquet"),
        "players": pd.read_parquet(data_dir / "players.parquet"),
        "tracking": pd.read_parquet(data_dir / "tracking.parquet"),
    }


def prepare_tracking_and_events(events: pd.DataFrame, tracking: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add game_event_id and flip-adjusted tracking columns."""
    events = events.copy()
    tracking = tracking.copy()
    events["game_event_id"] = events["game_id"].astype(str) + "_" + events["sl_event_id"].astype(str)
    tracking["game_event_id"] = tracking["game_id"].astype(str) + "_" + tracking["sl_event_id"].astype(str)
    events["flip"] = (
        (events["x_adj"] == -events["x"]) & (events["y_adj"] == -events["y"])
    )
    tracking = tracking.merge(events[["game_event_id", "flip"]], on="game_event_id")
    tracking["tracking_x_adj"] = tracking["tracking_x"]
    tracking["tracking_y_adj"] = tracking["tracking_y"]
    tracking["tracking_vel_x_adj"] = tracking["tracking_vel_x"]
    tracking["tracking_vel_y_adj"] = tracking["tracking_vel_y"]
    tracking.loc[tracking["flip"], "tracking_x_adj"] *= -1
    tracking.loc[tracking["flip"], "tracking_y_adj"] *= -1
    tracking.loc[tracking["flip"], "tracking_vel_x_adj"] *= -1
    tracking.loc[tracking["flip"], "tracking_vel_y_adj"] *= -1
    return events, tracking


def closest_point_on_segment(P, R, D):
    """
    Returns:
        s -> distance along PR segment (scalar)
        closest_point -> coordinates of closest point
    """
    P = np.array(P)
    R = np.array(R)
    D = np.array(D)
    PR = R - P
    pass_length = np.linalg.norm(PR)
    if pass_length == 0:
        return 0, P
    PR_unit = PR / pass_length
    s = np.dot(D - P, PR_unit)
    s = np.clip(s, 0, pass_length)
    closest = P + PR_unit * s
    return s, closest


def lane_obstructed(
    pass_row,
    tracking: pd.DataFrame,
    cone_half_angle_deg: float = 15.0,
    puck_speed: float = 30.0,
    intercept_radius: float = 3.0,
    dt: float = 0.05,
) -> bool:
    """Check if the pass lane is obstructed by a defender (cone + intercept logic)."""
    frame_id = pass_row["game_event_id"]
    poss_team = pass_row["event_team_id"]
    frame = tracking[tracking["game_event_id"] == frame_id]
    defenders = frame[frame["team_id"] != poss_team]
    P = np.array([pass_row["tracking_x_adj_i"], pass_row["tracking_y_adj_i"]])
    R = np.array([pass_row["tracking_x_adj_j"], pass_row["tracking_y_adj_j"]])
    PR = R - P
    pass_length = np.linalg.norm(PR)
    if pass_length == 0:
        return False
    PR_unit = PR / pass_length
    tan_half_angle = np.tan(np.deg2rad(cone_half_angle_deg))
    T_pass_total = pass_length / puck_speed
    for _, d in defenders.iterrows():
        D0 = np.array([d["tracking_x_adj"], d["tracking_y_adj"]])
        Vd = np.array([d["tracking_vel_x_adj"], d["tracking_vel_y_adj"]])
        s0, closest0 = closest_point_on_segment(P, R, D0)
        if 0 < s0 < pass_length:
            perp0 = np.linalg.norm(D0 - closest0)
            cone_radius0 = s0 * tan_half_angle
            if perp0 <= cone_radius0:
                return True
        t = 0.0
        while t <= T_pass_total:
            defender_pos = D0 + Vd * t
            s, closest = closest_point_on_segment(P, R, defender_pos)
            if s <= 0 or s >= pass_length:
                t += dt
                continue
            perp_dist = np.linalg.norm(defender_pos - closest)
            cone_radius = s * tan_half_angle
            within_intercept = perp_dist <= intercept_radius
            puck_depth = puck_speed * t
            puck_not_passed = puck_depth <= s
            if within_intercept and puck_not_passed:
                return True
            t += dt
    return False


def build_pass_options(events: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
    """Build pairwise pass options (same team, passer = event player) with distance, angle, lane_blocked."""
    out = []
    for gid, frame in tracking.groupby("game_event_id"):
        test = frame.merge(frame, on="game_event_id", suffixes=("_i", "_j"))
        test = test[test["player_id_i"] != test["player_id_j"]]
        if test.empty:
            continue
        dx = test["tracking_x_adj_j"] - test["tracking_x_adj_i"]
        dy = test["tracking_y_adj_j"] - test["tracking_y_adj_i"]
        test["distance"] = np.sqrt(dx**2 + dy**2)
        test["angle"] = np.degrees(np.arctan2(dy, dx))
        ev = events[events["game_event_id"] == gid][["game_event_id", "player_id", "team_id"]]
        if ev.empty:
            continue
        test = test.merge(ev, on="game_event_id")
        test = test.rename(columns={"player_id": "event_player_id", "team_id": "event_team_id"})
        pass_options = test[
            (test["team_id_i"] == test["event_team_id"])
            & (test["team_id_i"] == test["team_id_j"])
            & (test["player_id_i"] == test["event_player_id"])
        ].copy()
        out.append(pass_options)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def add_lane_blocked(pass_options: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
    """Add lane_blocked column using lane_obstructed."""
    pass_options = pass_options.copy()
    pass_options["lane_blocked"] = pass_options.apply(
        lambda row: lane_obstructed(row, tracking), axis=1
    )
    return pass_options


def plot_passing_cone(P, R, cone_angle_deg=12, base_radius=1.5, n_points=60):
    """Plot hybrid conical passing lane (for visualization). Requires matplotlib."""
    import matplotlib.pyplot as plt
    P = np.array(P)
    R = np.array(R)
    PR = R - P
    L = np.linalg.norm(PR)
    if L == 0:
        return
    PR_unit = PR / L
    theta = np.deg2rad(cone_angle_deg)
    perp = np.array([-PR_unit[1], PR_unit[0]])
    s_vals = np.linspace(0, L, n_points)
    left_boundary = []
    right_boundary = []
    for s in s_vals:
        center = P + PR_unit * s
        radius = base_radius + s * np.tan(theta)
        left_boundary.append(center + perp * radius)
        right_boundary.append(center - perp * radius)
    left_boundary = np.array(left_boundary)
    right_boundary = np.array(right_boundary)
    plt.plot([P[0], R[0]], [P[1], R[1]], linestyle="--", linewidth=2)
    plt.plot(left_boundary[:, 0], left_boundary[:, 1], linewidth=1)
    plt.plot(right_boundary[:, 0], right_boundary[:, 1], linewidth=1)
    cone_polygon = np.vstack((left_boundary, right_boundary[::-1]))
    plt.fill(cone_polygon[:, 0], cone_polygon[:, 1], alpha=0.15)


if __name__ == "__main__":
    data = load_data(DATA_DIR)
    events, tracking = prepare_tracking_and_events(data["events"], data["tracking"])
    pass_options = build_pass_options(events, tracking)
    pass_options = add_lane_blocked(pass_options, tracking)
    print("Pass options shape:", pass_options.shape)
    print("Lane blocked counts:\n", pass_options["lane_blocked"].value_counts())
