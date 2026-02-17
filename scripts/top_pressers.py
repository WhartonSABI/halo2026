#!/usr/bin/env python3
"""Print top pressers with names. Run: python scripts/top_pressers.py"""

from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
credit = pd.read_csv(PROJECT / "data/processed/player_press_credit.csv")
players = pd.read_parquet(PROJECT / "data/raw/players.parquet")

# Resolve player_id column name
pid_col = "player_id" if "player_id" in players.columns else "id"
name_col = "player_name" if "player_name" in players.columns else "name"
pos_col = "primary_position" if "primary_position" in players.columns else ("position" if "position" in players.columns else None)

top = credit[credit.n_rows >= 80].nlargest(20, "total_press_credit")
merged = top.merge(
    players[[pid_col, name_col] + ([pos_col] if pos_col else [])],
    left_on="player_id",
    right_on=pid_col,
    how="left",
)
cols = [name_col, pos_col] if pos_col else [name_col]
cols += ["n_rows", "total_press_credit", "avg_press_credit"]
print("Top 20 pressers (min 80 forecheck frames):\n")
print(merged[cols].to_string(index=False))
print("\n--- Context ---")
print(f"Mean avg_press_credit: {credit.avg_press_credit.mean():.4f}")
print(f"Median avg_press_credit: {credit.avg_press_credit.median():.4f}")
