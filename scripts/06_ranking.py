#!/usr/bin/env python3
"""Combined forecheck ranking from participation, distance, and modeling outputs.

Merges results from:
- participation.csv (03_simple-attribution, equal split)
- distance.csv (03_simple-attribution, distance-weighted)
- modeling.csv (05_modeling, hazard-model counterfactual credits)

Ranks by per-press credit (total / n_press) so players are comparable across ice time.
Output: ranking.csv with player_id, player_name, n_press, n_rows, and per-method totals/rates/ranks.
"""

from pathlib import Path

import pandas as pd
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "data" / "results"


def _per_press_col(df: pd.DataFrame, total_col: str, n_press_col: str = "n_press") -> pd.Series:
    """Per-press credit; use total_per_press if present, else total / n_press."""
    if "total_per_press" in df.columns:
        return df["total_per_press"]
    if total_col in df.columns and n_press_col in df.columns:
        return np.where(df[n_press_col] > 0, df[total_col] / df[n_press_col], np.nan)
    return df[total_col]


def main() -> None:
    dfs = []
    if (RESULTS / "participation.csv").exists():
        p = pd.read_csv(RESULTS / "participation.csv")
        if "n_presses" in p.columns and "n_press" not in p.columns:
            p = p.rename(columns={"n_presses": "n_press"})
        p["per_press"] = _per_press_col(p, "total")
        p["rank_participation"] = p["per_press"].rank(ascending=False, method="min", na_option="bottom")
        cols = ["player_id", "total", "per_press", "rank_participation"]
        if "n_press" in p.columns:
            cols.insert(1, "n_press")
        dfs.append(("participation", p[[c for c in cols if c in p.columns]].rename(
            columns={"total": "participation_total", "per_press": "participation_per_press", "rank_participation": "participation_rank"})))
    if (RESULTS / "distance.csv").exists():
        d = pd.read_csv(RESULTS / "distance.csv")
        if "n_presses" in d.columns and "n_press" not in d.columns:
            d = d.rename(columns={"n_presses": "n_press"})
        d["per_press"] = _per_press_col(d, "total")
        d["rank_distance"] = d["per_press"].rank(ascending=False, method="min", na_option="bottom")
        cols = ["player_id", "total", "per_press", "rank_distance"]
        if "n_press" in d.columns:
            cols.insert(1, "n_press")
        dfs.append(("distance", d[[c for c in cols if c in d.columns]].rename(
            columns={"total": "distance_total", "per_press": "distance_per_press", "rank_distance": "distance_rank"})))
    if (RESULTS / "modeling.csv").exists():
        m = pd.read_csv(RESULTS / "modeling.csv")
        total_col = "check_total" if "check_total" in m.columns else "total_check"
        per_col = "check_per_press" if "check_per_press" in m.columns else None
        if per_col and per_col in m.columns:
            m["per_press"] = m[per_col]
        else:
            m["per_press"] = _per_press_col(m, total_col)
        m["rank_model"] = m["per_press"].rank(ascending=False, method="min", na_option="bottom")
        cols = ["player_id", total_col, "per_press", "rank_model"]
        if "n_press" in m.columns:
            cols.insert(1, "n_press")
        if "n_rows" in m.columns:
            cols.insert(2, "n_rows")
        dfs.append(("modeling", m[[c for c in cols if c in m.columns]].rename(
            columns={total_col: "check_total", "per_press": "check_per_press", "rank_model": "check_rank"})))

    if not dfs:
        print("No result CSVs found. Run 03_simple-attribution and 05_modeling first.")
        return

    merged = dfs[0][1]
    for _, df in dfs[1:]:
        merged = merged.merge(df, on="player_id", how="outer")

    n_press_cols = [c for c in merged.columns if c == "n_press" or c.startswith("n_press_")]
    if n_press_cols:
        merged["n_press"] = merged[n_press_cols].bfill(axis=1).iloc[:, 0]
        merged = merged.drop(columns=[c for c in n_press_cols if c != "n_press"], errors="ignore")

    for path in [RESULTS / "participation.csv", RESULTS / "distance.csv", RESULTS / "modeling.csv"]:
        if path.exists():
            src = pd.read_csv(path)
            name_col = next((c for c in src.columns if c in ("player_name", "name")), None)
            if name_col:
                names = src[["player_id", name_col]].drop_duplicates()
                merged = merged.merge(names, on="player_id", how="left")
                if name_col != "player_name":
                    merged = merged.rename(columns={name_col: "player_name"})
            break

    order = [
        "player_id", "player_name", "n_press", "n_rows",
        "participation_total", "participation_per_press", "participation_rank",
        "distance_total", "distance_per_press", "distance_rank",
        "check_total", "check_per_press", "check_rank",
    ]
    out_cols = [c for c in order if c in merged.columns]
    for c in merged.columns:
        if c not in out_cols and c not in ("avg_rank", "composite_rank"):
            out_cols.append(c)
    ranking = merged[out_cols].sort_values(
        "check_rank" if "check_rank" in merged.columns else "participation_rank",
        na_position="last",
    ).reset_index(drop=True)

    out_path = RESULTS / "ranking.csv"
    ranking.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print("\nTop 20:")
    print(ranking.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
