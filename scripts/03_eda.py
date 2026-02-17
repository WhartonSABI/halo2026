#!/usr/bin/env python3
"""
EDA: Play through a game with puck location colored by zone.

Puck location colors:
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
PLOTS_DIR = PROJECT_ROOT / "plots"
JP_DIR = PROJECT_ROOT / "jp"

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


def main() -> None:
    save_longest_presses(interval_ms=350)


if __name__ == "__main__":
    main()
