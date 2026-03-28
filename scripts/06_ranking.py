#!/usr/bin/env python3
"""Combined forecheck ranking from participation, distance, and modeling outputs.

Merges results from:
- participation.csv (03_simple-attribution, equal split)
- distance.csv (03_simple-attribution, distance-weighted)
- modeling.csv (05_modeling, hazard-model counterfactual credits)

Ranks by per-forecheck credit so players are comparable across workload.
Output: ranking.csv with player_id, player_name, n_forechecks, n_rows,
and per-method totals/rates/ranks.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "data" / "results"


def _normalize_n_forechecks(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize forecheck-count column names across source files."""
    out = df.copy()
    for old in ("n_presses", "n_press"):
        if old in out.columns and "n_forechecks" not in out.columns:
            out = out.rename(columns={old: "n_forechecks"})
    return out


def _per_forecheck_col(df: pd.DataFrame, total_col: str, count_col: str = "n_forechecks") -> pd.Series:
    """Per-forecheck credit; use precomputed column if present, else total / count."""
    if "total_per_forecheck" in df.columns:
        return df["total_per_forecheck"]
    if "total_per_press" in df.columns:
        return df["total_per_press"]
    if total_col in df.columns and count_col in df.columns:
        return np.where(df[count_col] > 0, df[total_col] / df[count_col], np.nan)
    return df[total_col]


def _model_total_col(df: pd.DataFrame) -> str | None:
    for col in ("press_total", "check_total", "total_check"):
        if col in df.columns:
            return col
    return None


def _model_rate_col(df: pd.DataFrame) -> str | None:
    for col in ("press_per_forecheck", "check_per_press"):
        if col in df.columns:
            return col
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined ranking tables from method outputs.")
    parser.add_argument(
        "--min-n-forechecks-filter",
        type=int,
        default=20,
        help="Minimum n_forechecks for ranking-filtered.csv (set <=0 to disable filtered output).",
    )
    parser.add_argument(
        "--min-n-press-filter",
        dest="min_n_forechecks_filter",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--filtered-filename",
        type=str,
        default="ranking-filtered.csv",
        help="Filename for filtered ranking output under data/results/.",
    )
    args = parser.parse_args()

    dfs = []
    if (RESULTS / "participation.csv").exists():
        p = _normalize_n_forechecks(pd.read_csv(RESULTS / "participation.csv"))
        p["per_forecheck"] = _per_forecheck_col(p, "total")
        p["rank_participation"] = p["per_forecheck"].rank(ascending=False, method="min", na_option="bottom")
        cols = ["player_id", "total", "per_forecheck", "rank_participation"]
        if "n_forechecks" in p.columns:
            cols.insert(1, "n_forechecks")
        dfs.append(("participation", p[[c for c in cols if c in p.columns]].rename(
            columns={
                "n_forechecks": "n_forechecks_participation",
                "total": "participation_total",
                "per_forecheck": "participation_per_forecheck",
                "rank_participation": "participation_rank",
            })))

    if (RESULTS / "distance.csv").exists():
        d = _normalize_n_forechecks(pd.read_csv(RESULTS / "distance.csv"))
        d["per_forecheck"] = _per_forecheck_col(d, "total")
        d["rank_distance"] = d["per_forecheck"].rank(ascending=False, method="min", na_option="bottom")
        cols = ["player_id", "total", "per_forecheck", "rank_distance"]
        if "n_forechecks" in d.columns:
            cols.insert(1, "n_forechecks")
        dfs.append(("distance", d[[c for c in cols if c in d.columns]].rename(
            columns={
                "n_forechecks": "n_forechecks_distance",
                "total": "distance_total",
                "per_forecheck": "distance_per_forecheck",
                "rank_distance": "distance_rank",
            })))

    if (RESULTS / "modeling.csv").exists():
        m = _normalize_n_forechecks(pd.read_csv(RESULTS / "modeling.csv"))
        total_col = _model_total_col(m)
        per_col = _model_rate_col(m)
        if total_col is not None:
            if per_col is not None:
                m["per_forecheck"] = m[per_col]
            else:
                m["per_forecheck"] = _per_forecheck_col(m, total_col)
            m["rank_model"] = m["per_forecheck"].rank(ascending=False, method="min", na_option="bottom")
            cols = ["player_id", total_col, "per_forecheck", "rank_model"]
            if "n_forechecks" in m.columns:
                cols.insert(1, "n_forechecks")
            if "n_rows" in m.columns:
                cols.insert(2, "n_rows")
            dfs.append(("modeling", m[[c for c in cols if c in m.columns]].rename(
                columns={
                    "n_forechecks": "n_forechecks_modeling",
                    total_col: "press_total",
                    "per_forecheck": "press_per_forecheck",
                    "rank_model": "press_rank",
                })))

    if not dfs:
        print("No result CSVs found. Run 03_simple-attribution and 05_modeling first.")
        return

    merged = dfs[0][1]
    for _, df in dfs[1:]:
        merged = merged.merge(df, on="player_id", how="outer")

    n_forecheck_priority = [
        c for c in ["n_forechecks_modeling", "n_forechecks_participation", "n_forechecks_distance"]
        if c in merged.columns
    ]
    if n_forecheck_priority:
        merged["n_forechecks"] = merged[n_forecheck_priority].bfill(axis=1).iloc[:, 0]
        merged = merged.drop(columns=n_forecheck_priority, errors="ignore")
    if "n_forechecks" in merged.columns:
        merged["n_forechecks"] = merged["n_forechecks"].astype("Int64")

    for path in [RESULTS / "participation.csv", RESULTS / "distance.csv", RESULTS / "modeling.csv"]:
        if path.exists():
            src = pd.read_csv(path)
            name_col = next((c for c in src.columns if c in ("player_name", "name")), None)
            pos_col = next((c for c in src.columns if c in ("position", "primary_position")), None)
            if name_col:
                cols = ["player_id", name_col]
                if pos_col:
                    cols.append(pos_col)
                meta = src[cols].drop_duplicates(subset=["player_id"])
                merged = merged.merge(meta, on="player_id", how="left")
                if name_col != "player_name":
                    merged = merged.rename(columns={name_col: "player_name"})
                if pos_col and pos_col != "position":
                    merged = merged.rename(columns={pos_col: "position"})
            break

    if "position" in merged.columns:
        merged = merged[merged["position"] != "G"]
    if "n_rows" in merged.columns:
        merged["n_rows"] = merged["n_rows"].fillna(0).astype(int)

    order = [
        "player_id", "player_name", "n_forechecks", "n_rows",
        "participation_total", "participation_per_forecheck", "participation_rank",
        "distance_total", "distance_per_forecheck", "distance_rank",
        "press_total", "press_per_forecheck", "press_rank",
    ]
    out_cols = [c for c in order if c in merged.columns]
    for c in merged.columns:
        if c not in out_cols and c not in ("avg_rank", "composite_rank"):
            out_cols.append(c)
    ranking = merged[out_cols].sort_values(
        "press_rank" if "press_rank" in merged.columns else "participation_rank",
        na_position="last",
    ).reset_index(drop=True)

    out_path = RESULTS / "ranking.csv"
    ranking.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    if args.min_n_forechecks_filter > 0 and "n_forechecks" in ranking.columns:
        filtered = ranking[ranking["n_forechecks"] >= args.min_n_forechecks_filter].copy().reset_index(drop=True)
        filtered.insert(0, "filtered_rank", np.arange(1, len(filtered) + 1))
        filtered_path = RESULTS / args.filtered_filename
        filtered.to_csv(filtered_path, index=False)
        print(
            f"Saved: {filtered_path} "
            f"(n_forechecks >= {args.min_n_forechecks_filter}, {len(filtered)} of {len(ranking)} rows)"
        )

    print("\nTop 20:")
    print(ranking.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
