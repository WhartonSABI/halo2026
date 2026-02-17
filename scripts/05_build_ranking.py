#!/usr/bin/env python3
"""Build combined forecheck ranking from participation, distance, and modeling outputs.

Merges results from:
- participation.csv (04_simple_attribution, equal split)
- distance.csv (04_simple_attribution, distance-weighted)
- modeling.csv (04_modeling, hazard-model counterfactual credits)

Output: ranking.csv with composite_rank = mean of individual method ranks.
"""

from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "data" / "results"


def main() -> None:
    # ---- Load available result files ----
    dfs = []
    if (RESULTS / "participation.csv").exists():
        p = pd.read_csv(RESULTS / "participation.csv")
        p["rank_participation"] = p["total"].rank(ascending=False, method="min")  # higher total = better rank
        dfs.append(("participation", p[["player_id", "total", "rank_participation"]].rename(columns={"total": "total_participation"})))
    if (RESULTS / "distance.csv").exists():
        d = pd.read_csv(RESULTS / "distance.csv")
        d["rank_distance"] = d["total"].rank(ascending=False, method="min")
        dfs.append(("distance", d[["player_id", "total", "rank_distance"]].rename(columns={"total": "total_distance"})))
    if (RESULTS / "modeling.csv").exists():
        m = pd.read_csv(RESULTS / "modeling.csv")
        m["rank_model"] = m["total"].rank(ascending=False, method="min")
        dfs.append(("modeling", m[["player_id", "n_rows", "total", "rank_model"]].rename(columns={"total": "total_modeling"})))

    if not dfs:
        print("No result CSVs found. Run 04_simple_attribution and 04_modeling first.")
        return

    # ---- Outer merge on player_id; each method contributes its rank and total ----
    merged = dfs[0][1]
    for _, df in dfs[1:]:
        merged = merged.merge(df, on="player_id", how="outer")

    rank_cols = [c for c in merged.columns if c.startswith("rank_")]
    merged["avg_rank"] = merged[rank_cols].mean(axis=1)
    merged["composite_rank"] = merged["avg_rank"].rank(method="min").astype(int)  # lower avg_rank = better

    # ---- Enrich with player names from first available result file ----
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

    out_cols = ["composite_rank", "player_id", "player_name"] + [c for c in merged.columns if c not in ("composite_rank", "player_id", "player_name", "avg_rank")]
    out_cols = [c for c in out_cols if c in merged.columns]
    ranking = merged[out_cols].sort_values("composite_rank")

    out_path = RESULTS / "ranking.csv"
    ranking.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print("\nTop 20 by composite rank:")
    print(ranking.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
