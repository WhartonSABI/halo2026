#!/usr/bin/env python3
"""
EDA visuals: slot-change audit, attribution spreads, contribution histograms,
player ranking charts, modeling-vs-other scatter plots, game/press animations.

Puck location colors (for animations):
- Red: Team A's defensive zone (x < -25)
- Blue: Team B's defensive zone (x > 25)
- Black: Neutral zone (-25 <= x <= 25)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
import pandas as pd

##############
### CONFIG ###
##############

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"
JP_DIR = PROJECT_ROOT / "jp"

FORECHECK_SLOTS = ["F1", "F2", "F3", "F4", "F5"]

# Rink dimensions (x: -100 to 100, y: -42.5 to 42.5)
RINK_X = (-100, 100)
RINK_Y = (-42.5, 42.5)
BLUE_LINE = 25


####################
### LOAD HELPERS ###
####################

def load_events(data_dir: Path) -> pd.DataFrame:
    """Load events from parquet or CSV."""
    pq = data_dir / "events.parquet"
    csv = JP_DIR / "events.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        df = pd.read_csv(csv)
        df["x"] = df["x"].astype(float)
        df["y"] = df["y"].astype(float)
        return df
    raise FileNotFoundError(f"No events at {pq} or {csv}")


def load_tracking(data_dir: Path) -> pd.DataFrame:
    """Load tracking from parquet or CSV."""
    pq = data_dir / "tracking.parquet"
    csv = JP_DIR / "tracking.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        df = pd.read_csv(csv)
        for c in ["tracking_x", "tracking_y"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    return pd.DataFrame()


#####################
### PUCK ZONE COLOR ###
#####################

def puck_color(x: float) -> str:
    """Return color for puck location by zone (using raw x coordinate)."""
    if x < -BLUE_LINE:
        return "red"   # Team A defensive zone
    if x > BLUE_LINE:
        return "blue"  # Team B defensive zone
    return "black"    # Neutral zone


#################
### RINK PLOT ###
#################

def draw_rink(ax: plt.Axes) -> None:
    """Draw simple rink outline and blue lines."""
    ax.set_xlim(RINK_X[0] - 5, RINK_X[1] + 5)
    ax.set_ylim(RINK_Y[0] - 5, RINK_Y[1] + 5)
    ax.set_aspect("equal")

    # Boards
    rect = mpatches.Rectangle(
        (RINK_X[0], RINK_Y[0]),
        RINK_X[1] - RINK_X[0],
        RINK_Y[1] - RINK_Y[0],
        fill=False,
        edgecolor="gray",
        linewidth=2,
    )
    ax.add_patch(rect)

    # Blue lines
    ax.axvline(-BLUE_LINE, color="blue", linestyle="-", linewidth=1, alpha=0.7)
    ax.axvline(BLUE_LINE, color="blue", linestyle="-", linewidth=1, alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.5, alpha=0.5)


#################
### ANIMATION ###
#################

def play_game(
    game_id: str | None = None,
    interval_ms: int = 400,
    max_events: int = 200,
) -> None:
    """
    Animate a game: step through events, showing players and puck.

    Puck is colored:
    - Red: x < -25 (Team A defensive zone)
    - Blue: x > 25 (Team B defensive zone)
    - Black: neutral zone
    """
    events = load_events(DATA_DIR)
    tracking = load_tracking(DATA_DIR)

    if game_id is None:
        game_id = events["game_id"].iloc[0]
    ev = events[events["game_id"] == game_id].sort_values("sl_event_id").reset_index(drop=True)
    if len(ev) == 0:
        raise ValueError(f"No events for game {game_id}")

    ev = ev.head(max_events)
    tr = tracking[tracking["game_id"] == game_id] if len(tracking) > 0 else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(12, 6))
    draw_rink(ax)

    # Persistent artists
    player_scatter = ax.scatter([], [], c="gray", s=60, alpha=0.7, label="Players", zorder=2)
    puck_scatter = ax.scatter([], [], s=200, marker="o", edgecolors="white", linewidths=2, zorder=3)

    title = ax.set_title("")
    ax.legend(loc="upper right")

    def init():
        player_scatter.set_offsets(np.empty((0, 2)))
        puck_scatter.set_offsets(np.empty((0, 2)))
        puck_scatter.set_facecolors([])
        return player_scatter, puck_scatter, title

    def update(frame: int):
        if frame >= len(ev):
            return player_scatter, puck_scatter, title

        row = ev.iloc[frame]
        px, py = float(row["x"]), float(row["y"])
        color = puck_color(px)

        # Players at this event
        if len(tr) > 0:
            t0 = tr[tr["sl_event_id"] == row["sl_event_id"]]
            t0 = t0.dropna(subset=["tracking_x", "tracking_y"])
            if len(t0) > 0:
                tx = t0["tracking_x"].values
                ty = t0["tracking_y"].values
                player_scatter.set_offsets(np.column_stack([tx, ty]))
            else:
                player_scatter.set_offsets(np.empty((0, 2)))
        else:
            player_scatter.set_offsets(np.empty((0, 2)))

        puck_scatter.set_offsets([[px, py]])
        puck_scatter.set_facecolors([color])

        et = row.get("event_type", "?")
        desc = row.get("description", "")
        title.set_text(f"Event {frame + 1}/{len(ev)} | {et} | {desc}")

        return player_scatter, puck_scatter, title

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(ev),
        init_func=init,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / f"game_play_{game_id[:8]}.gif"
    ani.save(out_path, writer="pillow")
    print(f"Saved: {out_path}")
    plt.show()


def _save_press_gif(
    ev: pd.DataFrame,
    tr: pd.DataFrame,
    out_path: Path,
    title_prefix: str = "",
    interval_ms: int = 350,
    x_min: float | None = None,
) -> None:
    """Animate a single forecheck and save to GIF."""
    fig, ax = plt.subplots(figsize=(12, 6))
    draw_rink(ax)
    if x_min is not None:
        ax.set_xlim(x_min, RINK_X[1] + 5)

    player_scatter = ax.scatter([], [], c="gray", s=60, alpha=0.7, label="Players", zorder=2)
    puck_scatter = ax.scatter([], [], s=200, marker="o", edgecolors="white", linewidths=2, zorder=3)
    title = ax.set_title("")
    ax.legend(loc="upper right")

    def init():
        player_scatter.set_offsets(np.empty((0, 2)))
        puck_scatter.set_offsets(np.empty((0, 2)))
        puck_scatter.set_facecolors([])
        return player_scatter, puck_scatter, title

    def update(frame: int):
        if frame >= len(ev):
            return player_scatter, puck_scatter, title

        row = ev.iloc[frame]
        px, py = float(row["x"]), float(row["y"])
        color = puck_color(px)

        t0 = tr[(tr["game_id"] == row["game_id"]) & (tr["sl_event_id"] == row["sl_event_id"])]
        t0 = t0.dropna(subset=["tracking_x", "tracking_y"])
        if len(t0) > 0:
            player_scatter.set_offsets(np.column_stack([t0["tracking_x"].values, t0["tracking_y"].values]))
        else:
            player_scatter.set_offsets(np.empty((0, 2)))

        puck_scatter.set_offsets([[px, py]])
        puck_scatter.set_facecolors([color])

        et = row.get("event_type", "?")
        desc = row.get("description", "")
        title.set_text(f"{title_prefix} | Event {frame + 1}/{len(ev)} | {et} | {desc}")

        return player_scatter, puck_scatter, title

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(ev),
        init_func=init,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    ani.save(out_path, writer="pillow")
    plt.close(fig)
    print(f"Saved: {out_path}")


#####################################
### POSSESSION TIME VS RECOVERY ###
#####################################

def possession_time_vs_recovery() -> None:
    """Does length of possession (time or event count) correlate with recovery rate?"""
    events = load_events(DATA_DIR)
    forechecks = pd.read_parquet(PROCESSED_DIR / "forechecks.parquet")

    start_times = (
        events[["game_id", "sl_event_id", "period", "period_time"]]
        .rename(columns={"sl_event_id": "sl_event_id_start", "period_time": "period_time_start", "period": "period_start"})
    )
    end_times = (
        events[["game_id", "sl_event_id", "period", "period_time"]]
        .rename(columns={"sl_event_id": "sl_event_id_end", "period_time": "period_time_end", "period": "period_end"})
    )
    fc = (
        forechecks
        .merge(start_times, on=["game_id", "sl_event_id_start"])
        .merge(end_times, on=["game_id", "sl_event_id_end"])
    )
    same_period = fc["period_start"] == fc["period_end"]
    fc = fc.loc[same_period].copy()
    fc["duration_sec"] = fc["period_time_end"].astype(float) - fc["period_time_start"].astype(float)
    fc["event_count"] = fc["sl_event_id_end"] - fc["sl_event_id_start"]

    r_time = fc["duration_sec"].corr(fc["y"])
    r_events = fc["event_count"].corr(fc["y"])
    print("Correlation with recovery (y):")
    print(f"  duration_sec:  {r_time:.4f}")
    print(f"  event_count:  {r_events:.4f}")

    max_sec = float(fc["duration_sec"].max())
    duration_edges = [0, 5, 10, 15, 20, 30, 60, max(61, max_sec + 0.01)]
    fc["duration_bin"] = pd.cut(
        fc["duration_sec"],
        bins=duration_edges,
        labels=["0-5s", "5-10s", "10-15s", "15-20s", "20-30s", "30-60s", "60s+"],
    )
    by_duration = fc.groupby("duration_bin", observed=True).agg(
        n=("y", "count"),
        recovery_rate=("y", "mean"),
    ).round(4)
    print("\nRecovery rate by possession length (seconds):")
    print(by_duration)

    max_ev = int(fc["event_count"].max())
    event_edges = [0, 5, 10, 20, 50, 100, max(101, max_ev + 1)]
    fc["event_bin"] = pd.cut(
        fc["event_count"],
        bins=event_edges,
        labels=["1-5", "6-10", "11-20", "21-50", "51-100", "100+"],
    )
    by_events = fc.groupby("event_bin", observed=True).agg(
        n=("y", "count"),
        recovery_rate=("y", "mean"),
    ).round(4)
    print("\nRecovery rate by event count in sequence:")
    print(by_events)


def save_longest_presses(interval_ms: int = 350) -> None:
    """Save longest successful press (long_success.gif) and longest failure (long_failure.gif)."""
    forechecks = pd.read_parquet(PROCESSED_DIR / "forechecks.parquet")
    fc_events = pd.read_parquet(PROCESSED_DIR / "forecheck_events.parquet")
    fc_tracking = pd.read_parquet(PROCESSED_DIR / "forecheck_tracking.parquet")

    forechecks["length"] = forechecks["sl_event_id_end"] - forechecks["sl_event_id_start"]
    long_success = forechecks[forechecks["y"] == 1].nlargest(1, "length").iloc[0]
    long_failure = forechecks[forechecks["y"] == 0].nlargest(1, "length").iloc[0]

    for fc_id, name, title in [
        (long_success["fc_sequence_id"], "long_success", "Longest success"),
        (long_failure["fc_sequence_id"], "long_failure", "Longest failure"),
    ]:
        ev = fc_events[fc_events["fc_sequence_id"] == fc_id].sort_values("sl_event_id").reset_index(drop=True)
        tr = fc_tracking[fc_tracking["fc_sequence_id"] == fc_id]
        _save_press_gif(ev, tr, PLOTS_DIR / f"{name}.gif", title_prefix=title, interval_ms=interval_ms, x_min=-40)


#####################################
### ATTRIBUTION & CONTRIBUTION EDA ###
#####################################

def slot_change_audit() -> None:
    """Fraction of sequences where any slot_id changes across rows (stint relevance)."""
    path = PROCESSED_DIR / "hazard_features.parquet"
    if not path.exists():
        print("slot_change_audit: hazard_features.parquet not found (run 02_features.py)")
        return

    df = pd.read_parquet(path)
    slot_ids = [f"{s}_id" for s in FORECHECK_SLOTS if f"{s}_id" in df.columns]
    if not slot_ids:
        print("slot_change_audit: no slot_id columns found")
        return

    frac_changing = (
        df.groupby("fc_sequence_id")[slot_ids]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
        .mean()
    )
    print("Slot-change audit (stint attribution relevance):")
    print(f"  Fraction of sequences with slot changes (any slot): {frac_changing:.4f}")


def attribution_spreads() -> None:
    """% positive, negative, zero for each attribution method and modeling columns."""
    print("\n--- Attribution spreads ---")

    # Participation
    p_path = RESULTS_DIR / "participation.csv"
    if p_path.exists():
        p = pd.read_csv(p_path)
        if "n_presses" in p.columns and "n_press" not in p.columns:
            p = p.rename(columns={"n_presses": "n_press"})
        col = "total"
        s = p[col].dropna()
        n = len(s)
        if n > 0:
            pos, neg, zero = (s > 0).sum(), (s < 0).sum(), (s == 0).sum()
            print(f"\nParticipation ({col}): + {100*pos/n:.1f}% ({pos:,}) | - {100*neg/n:.1f}% ({neg:,}) | 0 {100*zero/n:.1f}%")

    # Distance
    d_path = RESULTS_DIR / "distance.csv"
    if d_path.exists():
        d = pd.read_csv(d_path)
        if "n_presses" in d.columns and "n_press" not in d.columns:
            d = d.rename(columns={"n_presses": "n_press"})
        col = "total"
        s = d[col].dropna()
        n = len(s)
        if n > 0:
            pos, neg, zero = (s > 0).sum(), (s < 0).sum(), (s == 0).sum()
            print(f"Distance ({col}): + {100*pos/n:.1f}% ({pos:,}) | - {100*neg/n:.1f}% ({neg:,}) | 0 {100*zero/n:.1f}%")

    # Modeling
    m_path = RESULTS_DIR / "modeling.csv"
    if m_path.exists():
        m = pd.read_csv(m_path)
        for col in ["pos_total", "exec_total", "check_total"]:
            if col not in m.columns:
                continue
            s = m[col].dropna()
            n = len(s)
            if n > 0:
                pos, neg, zero = (s > 0).sum(), (s < 0).sum(), (s == 0).sum()
                print(f"Modeling ({col}): + {100*pos/n:.1f}% ({pos:,}) | - {100*neg/n:.1f}% ({neg:,}) | 0 {100*zero/n:.1f}%")


def contribution_distributions() -> None:
    """Histograms of modeling contribution columns (pos_total, exec_total, check_total)."""
    path = RESULTS_DIR / "modeling.csv"
    if not path.exists():
        print("contribution_distributions: modeling.csv not found (run 05_modeling.py)")
        return

    m = pd.read_csv(path)
    cols = [c for c in ["pos_total", "exec_total", "check_total"] if c in m.columns]
    if not cols:
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        s = m[col].dropna()
        ax.hist(s, bins=50, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linestyle="--", alpha=0.7)
        ax.set_title(col)
        ax.set_xlabel("value")

    plt.suptitle("Contribution distributions (player-level)")
    plt.tight_layout()
    out = PLOTS_DIR / "contribution_distributions.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def player_rankings_visual(top_n: int = 20) -> None:
    """Bar charts of top players by each attribution method."""
    paths = {
        "participation": (RESULTS_DIR / "participation.csv", "total"),
        "distance": (RESULTS_DIR / "distance.csv", "total"),
        "modeling": (RESULTS_DIR / "modeling.csv", "check_total"),
    }
    dfs = {}
    for name, (p, total_col) in paths.items():
        if p.exists():
            df = pd.read_csv(p)
            if "n_presses" in df.columns and "n_press" not in df.columns:
                df = df.rename(columns={"n_presses": "n_press"})
            if total_col not in df.columns and name == "modeling":
                total_col = "total"
            if total_col in df.columns and "player_name" in df.columns:
                dfs[name] = df.nlargest(top_n, total_col)[["player_name", total_col]]

    if not dfs:
        print("player_rankings_visual: no result CSVs found")
        return

    methods = list(dfs.keys())
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(methods), 1, figsize=(10, 3 * len(methods)))
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        sub = dfs[method]
        val_col = [c for c in sub.columns if c != "player_name"][0]
        vals = sub[val_col].values
        ax.barh(range(len(sub)), vals)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["player_name"].values, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"Top {top_n} — {method}")
        v_min, v_max = vals.min(), vals.max()
        margin = max((v_max - v_min) * 0.05, 1e-6) if v_max != v_min else 0.01
        ax.set_xlim(v_min - margin, v_max + margin)
        if v_min <= 0 <= v_max:
            ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel(val_col)

    plt.suptitle("Player rankings by attribution method")
    plt.tight_layout()
    out = PLOTS_DIR / "player_rankings.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def player_press_distributions() -> None:
    """Possession-level spread of pos/exec/total from player_press.parquet."""
    path = PROCESSED_DIR / "player_press.parquet"
    if not path.exists():
        print("player_press_distributions: data/processed/player_press.parquet not found (run 05_modeling.py)")
        return

    pp = pd.read_parquet(path)
    for col, name in [
        ("positioning", "pos"),
        ("execution", "exec"),
        ("total_in_press", "total"),
    ]:
        if col not in pp.columns:
            continue
        s = pp[col].dropna()
        if len(s) == 0:
            continue
        print(f"\n{name} (possession): mean={s.mean():.4f}, median={s.median():.4f}, "
              f"pct_pos={100*(s>0).mean():.1f}%, n={len(s):,}")

    if "execution" in pp.columns:
        ex = pp["execution"]
        pos_vals = ex[ex > 0]
        neg_vals = ex[ex < 0]
        if len(pos_vals) > 0 and len(neg_vals) > 0:
            print(f"\nexec magnitude: mean(pos)={pos_vals.mean():.4f}, mean(neg)={neg_vals.mean():.4f}")

    n_press_per_player = pp.groupby("player_id").size()
    pp = pp.copy()
    pp["player_n_press"] = pp["player_id"].map(n_press_per_player)
    high = pp["player_n_press"] > 10
    low = pp["player_n_press"] <= 10
    if high.sum() > 0:
        sh = pp.loc[high, "execution"]
        print(f"exec from high-n (n_press>10): n={high.sum():,}, pct_pos={100*(sh>0).mean():.1f}%")
    if low.sum() > 0:
        sl = pp.loc[low, "execution"]
        print(f"exec from low-n (n_press<=10): n={low.sum():,}, pct_pos={100*(sl>0).mean():.1f}%")


def ranking_comparison_scatter() -> None:
    """Scatter: model vs participation, model vs distance, distance vs participation."""
    paths = {
        "participation": RESULTS_DIR / "participation.csv",
        "distance": RESULTS_DIR / "distance.csv",
        "modeling": RESULTS_DIR / "modeling.csv",
    }
    dfs = {}
    for name, p in paths.items():
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "n_presses" in df.columns:
            df = df.rename(columns={"n_presses": "n_press"})
        total_col = "check_total" if name == "modeling" and "check_total" in df.columns else "total"
        if total_col in df.columns:
            dfs[name] = df[["player_id", total_col]].rename(columns={total_col: name})

    if len(dfs) < 2:
        print("ranking_comparison_scatter: need at least 2 of modeling, participation, distance")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("modeling", "participation", "model_vs_participation", "model", "participation"),
        ("modeling", "distance", "model_vs_distance", "model", "distance"),
        ("distance", "participation", "distance_vs_participation", "distance", "participation"),
    ]
    for x_name, y_name, out_name, x_label, y_label in pairs:
        if x_name not in dfs or y_name not in dfs:
            continue
        merged = dfs[x_name].merge(dfs[y_name], on="player_id", how="inner")
        if len(merged) < 5:
            continue
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(merged[x_name], merged[y_name], alpha=0.6)
        ax.axhline(0, color="gray", linestyle="--")
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{x_label} vs {y_label} (player totals)")
        plt.tight_layout()
        out = PLOTS_DIR / f"{out_name}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved: {out}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="EDA visuals for forecheck analysis")
    parser.add_argument("--all", action="store_true", help="Run all (default)")
    parser.add_argument("--possession", action="store_true", help="Possession time vs recovery")
    parser.add_argument("--gifs", action="store_true", help="Save longest press GIFs")
    parser.add_argument("--slot-audit", action="store_true", help="Slot-change audit")
    parser.add_argument("--spreads", action="store_true", help="Attribution positive/negative spreads")
    parser.add_argument("--distributions", action="store_true", help="Contribution histograms")
    parser.add_argument("--rankings", action="store_true", help="Top player bar charts")
    parser.add_argument("--player-press", action="store_true", help="Possession-level credit stats")
    parser.add_argument("--scatter", action="store_true", help="Modeling vs participation/distance scatter")
    args = parser.parse_args()

    run_all = args.all or not any([
        args.possession, args.gifs, args.slot_audit, args.spreads,
        args.distributions, args.rankings, args.player_press, args.scatter,
    ])

    if run_all or args.possession:
        possession_time_vs_recovery()
    if run_all or args.gifs:
        save_longest_presses(interval_ms=350)
    if run_all or args.slot_audit:
        slot_change_audit()
    if run_all or args.spreads:
        attribution_spreads()
    if run_all or args.distributions:
        contribution_distributions()
    if run_all or args.rankings:
        player_rankings_visual()
    if run_all or args.player_press:
        player_press_distributions()
    if run_all or args.scatter:
        ranking_comparison_scatter()


if __name__ == "__main__":
    main()
