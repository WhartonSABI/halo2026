#!/usr/bin/env python3
"""Evaluation: calibration diagnostics and benchmarking.

1. Calibration (out-of-sample): start model + hazard model reliability diagrams.
2. Benchmark: participation, distance, modeling against each other.
"""

from __future__ import annotations

import importlib.util
import numpy as np
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"
_SCRIPTS = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(PROJECT_ROOT))

from _preprocess import add_slot_imputed_indicators, build_feature_lists


def _load_modeling_module():
    spec = importlib.util.spec_from_file_location(
        "modeling", _SCRIPTS / "05_modeling.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_and_fit_hybrid():
    """Load data, fit+calibrate hybrid via 05_modeling, return models and test data."""
    mod = _load_modeling_module()
    df = mod.load_data()
    add_slot_imputed_indicators(df)
    train_df, test_df = mod.split_groups(df)
    numeric_cols, cat_cols = build_feature_lists(df)
    start_pipe, hazard_pipe = mod.fit_and_calibrate_hybrid(train_df, numeric_cols, cat_cols)
    _, test_is_start = mod._compute_start_meta(test_df)
    return start_pipe, hazard_pipe, test_df, test_is_start, numeric_cols, cat_cols


def plot_calibration(ax, y_true_binary, y_prob, label, color, n_bins=10):
    prob_true, prob_pred = calibration_curve(
        y_true_binary, y_prob, n_bins=n_bins, strategy="quantile"
    )
    ax.plot(prob_pred, prob_true, "s-", label=label, color=color)
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.5)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")


def run_calibration() -> None:
    """Out-of-sample calibration for start and hazard models."""
    print("Loading and fitting hybrid model (same split as 05_modeling)...")
    start_pipe, hazard_pipe, test_df, test_is_start, numeric_cols, cat_cols = load_and_fit_hybrid()

    mod = _load_modeling_module()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Start model calibration (test set) ---
    start_index, _ = mod._compute_start_meta(test_df)
    start_row_idx = start_index.values
    X_test_start = test_df.loc[start_row_idx, numeric_cols + cat_cols]
    seq_outcome = test_df.groupby("fc_sequence_id")["event_t"].max().reset_index()
    seq_outcome["success"] = (seq_outcome["event_t"] == 1).astype(np.int32)
    start_seq_ids = test_df.loc[start_row_idx, "fc_sequence_id"].values
    y_test_start = seq_outcome.set_index("fc_sequence_id").loc[start_seq_ids, "success"].values
    p_success = start_pipe.predict_proba(X_test_start)[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_calibration(ax, y_test_start, p_success, "P(sequence success)", "C0", n_bins=12)
    ax.set_title("Start model: P(sequence success) given start config")
    fig.suptitle("Start model calibration (test set)", fontsize=12)
    fig.tight_layout()
    path_start = PLOTS_DIR / "calibration_start.png"
    fig.savefig(path_start, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path_start}")
    print(f"  Start: pred mean={p_success.mean():.4f}, actual={y_test_start.mean():.4f}")

    # --- 2. Hazard model calibration (test exec rows) ---
    test_exec_mask = ~test_is_start
    X_test_exec = test_df.loc[test_exec_mask, numeric_cols + cat_cols]
    y_test_exec = test_df.loc[test_exec_mask, "target_class"]
    proba_exec = hazard_pipe.predict_proba(X_test_exec)
    h_success = proba_exec[:, 1]
    h_failure = proba_exec[:, 2]
    y_success = (y_test_exec.values == 1).astype(np.int32)
    y_failure = (y_test_exec.values == 2).astype(np.int32)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_calibration(axes[0], y_success, h_success, "Success hazard", "C0", n_bins=12)
    axes[0].set_title("Hazard exec: success")
    plot_calibration(axes[1], y_failure, h_failure, "Failure hazard", "C1", n_bins=12)
    axes[1].set_title("Hazard exec: failure")
    fig.suptitle("Hazard model calibration (test set, exec rows only)", fontsize=12)
    fig.tight_layout()
    path_hazard = PLOTS_DIR / "calibration_hazard.png"
    fig.savefig(path_hazard, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path_hazard}")
    print(f"  Hazard success: pred={h_success.mean():.4f}, actual={y_success.mean():.4f}")
    print(f"  Hazard failure: pred={h_failure.mean():.4f}, actual={y_failure.mean():.4f}")
    print(f"  Log loss (exec): {log_loss(y_test_exec, proba_exec, labels=[0, 1, 2]):.4f}")


def _participants_from_stints(forechecks, events, stints, players):
    """Pressing-team skaters on ice at terminal event. Same logic as 03_simple-attribution."""
    term_events = forechecks[["fc_sequence_id", "game_id", "sl_event_id_end", "pressing_team_id"]].merge(
        events[["game_id", "sl_event_id", "game_stint"]].drop_duplicates(),
        left_on=["game_id", "sl_event_id_end"],
        right_on=["game_id", "sl_event_id"],
        how="inner",
    )[["fc_sequence_id", "game_id", "game_stint", "pressing_team_id"]]

    pos_col = "primary_position" if "primary_position" in players.columns else "position"
    skater_ids = set(players.loc[~players[pos_col].isin({"G"}), "player_id"]) if pos_col in players.columns else None

    stint_players = stints[["game_id", "game_stint", "player_id", "team_id"]].dropna(subset=["player_id"])
    merged = term_events.merge(stint_players, on=["game_id", "game_stint"], how="left")
    merged = merged[merged["team_id"] == merged["pressing_team_id"]]
    if skater_ids:
        merged = merged[merged["player_id"].isin(skater_ids)]

    participants = (
        merged.groupby("fc_sequence_id")["player_id"]
        .apply(lambda x: x.dropna().unique().tolist())
        .reset_index()
        .rename(columns={"player_id": "participant_ids"})
    )
    all_seq = term_events[["fc_sequence_id"]].drop_duplicates()
    participants = all_seq.merge(participants, on="fc_sequence_id", how="left")
    participants["participant_ids"] = participants["participant_ids"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    return participants


def run_benchmark() -> None:
    """Out-of-sample prediction benchmark.

    Forecheck-level: predict success (0/1) from players-on-ice metrics. Fit on train
    games, evaluate log loss on test games. Which method best predicts turnover success?
    Team-game won: same split, predict win from team metric.
    """
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import log_loss

    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    RESULTS_DIR = PROJECT_ROOT / "data" / "results"
    FORECHECKS_PATH = PROJECT_ROOT / "data" / "processed" / "forechecks.parquet"

    for p in [RAW_DIR, RESULTS_DIR, FORECHECKS_PATH]:
        if not p.exists():
            print(f"  Missing {p}. Run pipeline first.")
            return

    forechecks = pd.read_parquet(FORECHECKS_PATH)
    events = pd.read_parquet(RAW_DIR / "events.parquet")
    stints = pd.read_parquet(RAW_DIR / "stints.parquet")
    players = pd.read_parquet(RAW_DIR / "players.parquet")
    games = pd.read_parquet(RAW_DIR / "games.parquet")

    participants = _participants_from_stints(forechecks, events, stints, players)
    fc = forechecks[["fc_sequence_id", "game_id", "pressing_team_id", "y"]].merge(
        participants, on="fc_sequence_id", how="left"
    )

    method_data = {}
    for name, path, total_col, n_col in [
        ("participation", RESULTS_DIR / "participation.csv", "total", "n_press"),
        ("distance", RESULTS_DIR / "distance.csv", "total", "n_press"),
        ("modeling", RESULTS_DIR / "modeling.csv", "check_total", "n_press"),
        ("modeling_rfcde", RESULTS_DIR / "rfcde_ghosts" / "modeling.csv", "check_total", "n_press"),
        ("modeling_rfcde_dist", RESULTS_DIR / "rfcde_distributional" / "modeling.csv", "check_total", "n_press"),
    ]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        tcol = total_col if total_col in df.columns else "total"
        ncol = n_col if n_col in df.columns else "n_press"
        if tcol not in df.columns or ncol not in df.columns:
            continue
        method_data[name] = df[["player_id", tcol, ncol]].dropna().rename(
            columns={tcol: "total", ncol: "n_press"}
        )

    if not method_data:
        print("  No method CSVs. Run 03 and 05.")
        return

    # Expand forechecks to (fc_sequence_id, player_id) for participants
    rows = []
    for _, r in fc.iterrows():
        for pid in (r["participant_ids"] if isinstance(r["participant_ids"], list) else []):
            rows.append({"fc_sequence_id": r["fc_sequence_id"], "game_id": r["game_id"], "y": r["y"], "player_id": pid})
    fc_exp = pd.DataFrame(rows)
    if fc_exp.empty:
        print("  No forechecks with participants.")
        return

    game_ids = forechecks["game_id"].unique()
    n_games = len(game_ids)
    kfold = GroupKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n[Benchmark] 5-fold CV over games (each game in test once)")
    print("  Turnover success: forecheck-level. Metric = sum(total)/sum(n_press) for players on ice.")
    print("  Won: team-game level. Mean ± std across folds.")
    print()

    results_turnover = {name: [] for name in method_data}
    for fold, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(n_games), np.zeros(n_games), groups=game_ids)):
        train_games = set(game_ids[train_idx])
        test_games = set(game_ids[test_idx])
        fc["_split"] = fc["game_id"].apply(lambda x: "train" if x in train_games else "test")

        for name, data in method_data.items():
            merged = fc_exp.merge(data, on="player_id", how="inner")
            agg = merged.groupby("fc_sequence_id").agg(
                total=("total", "sum"),
                n_press=("n_press", "sum"),
            ).reset_index()
            agg["metric"] = np.where(agg["n_press"] > 0, agg["total"] / agg["n_press"], np.nan)
            fc_with_metric = fc.merge(agg[["fc_sequence_id", "metric"]], on="fc_sequence_id", how="inner")
            fc_with_metric = fc_with_metric.dropna(subset=["metric"])
            fc_with_metric = fc_with_metric[fc_with_metric["participant_ids"].apply(len) > 0]

            train_fc = fc_with_metric[fc_with_metric["_split"] == "train"]
            test_fc = fc_with_metric[fc_with_metric["_split"] == "test"]

            if len(train_fc) < 10 or len(test_fc) < 10:
                continue

            mdl = LogisticRegression(random_state=42, max_iter=500).fit(
                train_fc[["metric"]].values, train_fc["y"].astype(int).values
            )
            pred = mdl.predict_proba(test_fc[["metric"]].values)[:, 1]
            ll = log_loss(test_fc["y"].astype(int), pred)
            results_turnover[name].append(ll)

    for name in sorted(method_data.keys()):
        vals = results_turnover.get(name, [])
        if vals:
            mean_ll, std_ll = np.mean(vals), np.std(vals)
            print(f"  Turnover success (log loss): {name} = {mean_ll:.5f} ± {std_ll:.5f}")
    if results_turnover:
        best = min(
            [(n, np.mean(v)) for n, v in results_turnover.items() if v],
            key=lambda x: x[1],
        )
        print(f"  Best: {best[0]}")
    print()

    # Won: team-game level, 5-fold CV
    tg_turnover = fc.groupby(["game_id", "pressing_team_id"]).size().reset_index(name="n_total")
    tg_turnover = tg_turnover.rename(columns={"pressing_team_id": "team_id"})
    g = games[["game_id", "home_team_id", "away_team_id", "game_outcome"]]
    home = g[["game_id", "home_team_id"]].rename(columns={"home_team_id": "team_id"})
    home["won"] = (g["game_outcome"] == "home_win").astype(int)
    away = g[["game_id", "away_team_id"]].rename(columns={"away_team_id": "team_id"})
    away["won"] = (g["game_outcome"] == "away_win").astype(int)
    tg_won = pd.concat([home, away], ignore_index=True)

    player_team = (
        stints[["game_id", "player_id", "team_id"]]
        .dropna(subset=["player_id", "team_id"])
        .drop_duplicates(["game_id", "player_id"])
    )

    tg = tg_turnover.merge(tg_won, on=["game_id", "team_id"], how="left")
    for name, data in method_data.items():
        merged = player_team.merge(data, on="player_id", how="inner")
        agg = merged.groupby(["game_id", "team_id"]).agg(
            total=("total", "sum"),
            n_press=("n_press", "sum"),
        ).reset_index()
        agg[f"{name}_metric"] = np.where(agg["n_press"] > 0, agg["total"] / agg["n_press"], np.nan)
        tg = tg.merge(agg[["game_id", "team_id", f"{name}_metric"]], on=["game_id", "team_id"], how="left")

    metric_cols = [c for c in tg.columns if c.endswith("_metric")]
    tg = tg.dropna(subset=["won"] + metric_cols)

    results_won = {col.replace("_metric", ""): [] for col in metric_cols}
    for fold, (train_idx, test_idx) in enumerate(kfold.split(np.zeros(n_games), np.zeros(n_games), groups=game_ids)):
        train_games = set(game_ids[train_idx])
        test_games = set(game_ids[test_idx])
        tg["_split"] = tg["game_id"].apply(lambda x: "train" if x in train_games else "test")
        train_tg = tg[tg["_split"] == "train"]
        test_tg = tg[tg["_split"] == "test"]

        if train_tg.empty or test_tg.empty:
            continue

        for col in metric_cols:
            name = col.replace("_metric", "")
            mdl = LogisticRegression(random_state=42, max_iter=500).fit(
                train_tg[[col]].values, train_tg["won"].astype(int).values
            )
            pred = mdl.predict_proba(test_tg[[col]].values)[:, 1]
            ll = log_loss(test_tg["won"].astype(int), pred)
            results_won[name].append(ll)

    for name in sorted(results_won.keys()):
        vals = results_won[name]
        if vals:
            mean_ll, std_ll = np.mean(vals), np.std(vals)
            print(f"  Won (log loss): {name} = {mean_ll:.5f} ± {std_ll:.5f}")
    if results_won:
        best = min(
            [(n, np.mean(v)) for n, v in results_won.items() if v],
            key=lambda x: x[1],
        )
        print(f"  Best: {best[0]}")


def main() -> None:
    run_calibration()
    run_benchmark()


if __name__ == "__main__":
    main()
