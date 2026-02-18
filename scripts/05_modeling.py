#!/usr/bin/env python3
"""Train competing-risk hazard models for forecheck outcomes.

This script fits three event-level classifiers over hazard rows:
1) Multinomial logistic regression (baseline, interpretable)
2) Gradient boosting (nonlinear interactions)
3) Neural net (MLP)

Target classes:
- 0: no terminal event at this row
- 1: terminal success
- 2: terminal failure

Key modeling choice:
- Uses `time_since_start_s` as a continuous-time covariate with flexible basis
  expansions (log-time and spline basis), so each model learns hazard as a
  function of elapsed continuous time.

Player crediting logic (value of a press):
- For each hazard row, define press value = P(success) - P(failure).
- For each forechecker slot F1..F5, compute a leave-one-out counterfactual by
  removing the slot player ID and replacing slot pressure/lane features with
  contextual medians (same manpower/score/time-bin context when possible).
- Slot credit on the row is the drop in press value under this counterfactual.
  Positive credit means that player's pressure increased expected success
  relative to failure.
- Aggregate row credits to player-level totals/means for valuation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hazard_features.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

RANDOM_STATE = 7
FORECHECK_SLOTS = ["F1", "F2", "F3", "F4", "F5"]


class TimeAugmenter:
    """Simple sklearn-compatible transformer to add time basis features.

    Adds log and sqrt transforms of elapsed time so models can capture
    nonlinear hazard rates (e.g., higher risk early in a sequence).
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TimeAugmenter":
        """No-op: this transformer is stateless."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        t = X["time_since_start_s"].astype(float).clip(lower=0.0)  # avoid log(0)
        X["log_time_since_start_s"] = np.log1p(t)
        X["sqrt_time_since_start_s"] = np.sqrt(t)
        return X

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        if input_features is None:
            return np.array([])
        return np.asarray(list(input_features) + ["log_time_since_start_s", "sqrt_time_since_start_s"])


def load_data() -> pd.DataFrame:
    """Load hazard features and define 3-class target: ongoing(0), success(1), failure(2)."""
    df = pd.read_parquet(DATA_PATH)
    y = np.zeros(len(df), dtype=np.int64)
    y[df["event_t"].eq(1).values] = 1  # terminal success
    y[df["terminal_failure_t"].eq(1).values] = 2  # terminal failure
    df["target_class"] = y
    return df


def split_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test split by fc_sequence_id. No sequence appears in both sets (prevents leakage)."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(splitter.split(df, groups=df["fc_sequence_id"]))
    return df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()


def build_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Partition columns into numeric and categorical, excluding IDs and targets."""
    ignore = {
        "fc_sequence_id",
        "sl_event_id",
        "event_t",
        "terminal_failure_t",
        "target_class",
        "carrier_id",
        "F1_id",
        "F2_id",
        "F3_id",
        "F4_id",
        "F5_id",
    }
    feature_cols = [c for c in df.columns if c not in ignore]

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in numeric_cols]
    return numeric_cols, cat_cols


def build_preprocessor(numeric_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Build preprocessing pipeline: impute, spline time, scale numerics, one-hot encode categories."""
    time_spline_cols = ["time_since_start_s"]
    numeric_main_cols = [c for c in numeric_cols if c not in time_spline_cols]

    numeric_main = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    time_spline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "spline",
                SplineTransformer(n_knots=5, degree=3, include_bias=False),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_main, numeric_main_cols),
            ("time_spline", time_spline, time_spline_cols),
            ("cat", categorical, cat_cols),
        ]
    )


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute log loss, classification report, and mean predicted hazards. Return metrics dict."""
    proba = model.predict_proba(X_test)
    pred = np.argmax(proba, axis=1)

    ll = log_loss(y_test, proba, labels=[0, 1, 2])
    print(f"\n=== {name} ===")
    print(f"log_loss: {ll:.5f}")
    print(classification_report(y_test, pred, labels=[0, 1, 2], target_names=["ongoing", "success", "failure"]))

    success_hazard = proba[:, 1]
    failure_hazard = proba[:, 2]
    print(
        "mean predicted hazards | "
        f"success: {success_hazard.mean():.5f}, failure: {failure_hazard.mean():.5f}"
    )

    return {
        "model": name,
        "log_loss": ll,
        "mean_success_hazard": float(success_hazard.mean()),
        "mean_failure_hazard": float(failure_hazard.mean()),
    }


def contextual_median_slot_replacement(df: pd.DataFrame, slot: str) -> pd.DataFrame:
    """Counterfactual: remove slot identity and replace slot features by contextual medians.

    Used for leave-one-out attribution: simulate "this player wasn't here" by replacing
    their pressure/lane features with the median under the same manpower/score/time context.
    """
    cf = df.copy()

    # Erase player identity so the model cannot use "who" is in the slot
    slot_id = f"{slot}_id"
    if slot_id in cf.columns:
        cf[slot_id] = np.nan

    # Slot-specific pressure and geometry features to replace
    slot_cols = [
        f"{slot}_r",
        f"{slot}_vr_carrier",
        f"{slot}_sinθ",
        f"{slot}_cosθ",
        f"{slot}_block_severity",
        f"{slot}_block_center_severity",
        f"{slot}_r_nearestOpp",
        f"{slot}_vr_nearestOpp",
    ]

    # Replace with median within same game context (manpower, score, time bin)
    context_candidates = ["manpower_state", "score_diff_bin", "time_since_start_bin"]
    context_cols = [c for c in context_candidates if c in df.columns]

    for col in slot_cols:
        if col not in cf.columns:
            continue

        if context_cols:
            contextual = df.groupby(context_cols, dropna=False)[col].transform("median")
            replacement = contextual.fillna(df[col].median())  # fallback if context missing
        else:
            replacement = pd.Series(df[col].median(), index=df.index)

        cf[col] = replacement

    return cf


def build_player_press_credit(
    model: Pipeline,
    source_df: pd.DataFrame,
    X_source: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate player-level press value and split into start-state vs execution credit.

    Decomposition for each row follows:
    - start_positioning_credit: effect on sequence-start press value
    - execution_credit: remaining effect beyond sequence start
    - total_press_credit: start_positioning_credit + execution_credit
    """
    base_proba = model.predict_proba(X_source)
    base_press_value = base_proba[:, 1] - base_proba[:, 2]  # P(success) - P(failure)

    # Identify sequence-start row per fc_sequence_id (earliest elapsed time; tie-break by event id)
    ordering = pd.DataFrame(
        {
            "fc_sequence_id": source_df["fc_sequence_id"].values,
            "time_since_start_s": X_source["time_since_start_s"].astype(float).values,
            "sl_event_id": source_df["sl_event_id"].values,
        },
        index=source_df.index,
    )
    start_rows = (
        ordering.sort_values(["fc_sequence_id", "time_since_start_s", "sl_event_id"])
        .groupby("fc_sequence_id", as_index=False)
        .first()[["fc_sequence_id", "time_since_start_s", "sl_event_id"]]
    )
    start_index = (
        ordering.reset_index()
        .merge(start_rows, on=["fc_sequence_id", "time_since_start_s", "sl_event_id"], how="inner")
        .drop_duplicates(subset=["fc_sequence_id"])
        .set_index("fc_sequence_id")["index"]
    )

    base_start_by_seq = pd.Series(base_press_value, index=source_df.index).loc[start_index.values]
    base_start_by_seq.index = start_index.index

    # For each forechecker slot F1..F5, compute leave-one-out credit
    credit_rows: list[pd.DataFrame] = []
    for slot in FORECHECK_SLOTS:
        slot_id = f"{slot}_id"
        if slot_id not in source_df.columns:
            continue

        cf_features = contextual_median_slot_replacement(X_source, slot)
        cf_proba = model.predict_proba(cf_features)
        cf_press_value = cf_proba[:, 1] - cf_proba[:, 2]

        cf_start_by_seq = pd.Series(cf_press_value, index=source_df.index).loc[start_index.values]
        cf_start_by_seq.index = start_index.index

        total_credit = base_press_value - cf_press_value  # drop in press value when removing this slot

        # Decompose: start_positioning = effect at sequence start; execution = remainder
        base_start_aligned = source_df["fc_sequence_id"].map(base_start_by_seq).to_numpy(dtype=float)
        cf_start_aligned = source_df["fc_sequence_id"].map(cf_start_by_seq).to_numpy(dtype=float)
        start_positioning_credit = base_start_aligned - cf_start_aligned
        execution_credit = total_credit - start_positioning_credit

        slot_frame = pd.DataFrame(
            {
                "player_id": source_df[slot_id].values,
                "fc_sequence_id": source_df["fc_sequence_id"].values,
                "sl_event_id": source_df["sl_event_id"].values,
                "slot": slot,
                "start_positioning_credit": start_positioning_credit,
                "execution_credit": execution_credit,
                "total_press_credit": total_credit,
            }
        )
        slot_frame = slot_frame.dropna(subset=["player_id"])
        credit_rows.append(slot_frame)

    if not credit_rows:
        return pd.DataFrame(
            columns=[
                "player_id",
                "n_rows",
                "total_start_positioning_credit",
                "avg_start_positioning_credit",
                "total_execution_credit",
                "avg_execution_credit",
                "total_press_credit",
                "avg_press_credit",
            ]
        )

    per_row = pd.concat(credit_rows, ignore_index=True)

    # Cap extreme counterfactual deltas (can occur when model extrapolates)
    for col in ["start_positioning_credit", "execution_credit", "total_press_credit"]:
        per_row[col] = per_row[col].clip(lower=-1.0, upper=1.0)

    # Aggregate row-level credits to player totals and means
    summary = (
        per_row.groupby("player_id", as_index=False)
        .agg(
            n_rows=("total_press_credit", "size"),
            n_presses=("fc_sequence_id", "nunique"),
            total_start_positioning_credit=("start_positioning_credit", "sum"),
            avg_start_positioning_credit=("start_positioning_credit", "mean"),
            total_execution_credit=("execution_credit", "sum"),
            avg_execution_credit=("execution_credit", "mean"),
            total_press_credit=("total_press_credit", "sum"),
            avg_press_credit=("total_press_credit", "mean"),
        )
        .sort_values("total_press_credit", ascending=False)
    )
    return summary


def _write_clean_csv(credit: pd.DataFrame, out_path: Path) -> None:
    """Write human-readable CSV with player names/positions merged from raw player data."""
    players_df = pd.read_parquet(RAW_DIR / "players.parquet")
    pid_col = "player_id" if "player_id" in players_df.columns else "id"
    name_col = "player_name" if "player_name" in players_df.columns else "name"
    pos_col = "primary_position" if "primary_position" in players_df.columns else "position"
    merge_cols = {pid_col: "player_id", name_col: "player_name"}
    if pos_col in players_df.columns:
        merge_cols[pos_col] = "position"
    out = credit[["player_id", "n_rows", "n_presses", "total_start_positioning_credit", "total_execution_credit"]].copy()
    out = out.rename(columns={
        "total_start_positioning_credit": "start_positioning",
        "total_execution_credit": "execution",
    })
    out["total"] = out["start_positioning"] + out["execution"]
    out["total_per_press"] = np.where(out["n_presses"] > 0, out["total"] / out["n_presses"], np.nan)
    out = out.sort_values("total", ascending=False).reset_index(drop=True)
    out = out.merge(
        players_df[[c for c in [pid_col, name_col, pos_col] if c in players_df.columns]].rename(columns=merge_cols),
        on="player_id",
        how="left",
    )
    out_cols = ["player_id", "player_name", "position", "n_presses", "n_rows", "start_positioning", "execution", "total", "total_per_press"]
    out = out[[c for c in out_cols if c in out.columns]]
    out.to_csv(out_path, index=False)


def main() -> None:
    # ---- Data loading and splitting ----
    df = load_data()
    train_df, test_df = split_groups(df)

    numeric_cols, cat_cols = build_feature_lists(df)

    X_train = train_df[numeric_cols + cat_cols]
    y_train = train_df["target_class"]
    X_test = test_df[numeric_cols + cat_cols]
    y_test = test_df["target_class"]

    preprocessor = build_preprocessor(numeric_cols, cat_cols)

    # ---- Train competing models ----
    models = {
        "multinomial_logit": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "gbm_hist": HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.05,
            max_iter=300,
            random_state=RANDOM_STATE,
        ),
        "neural_net_mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=200,
            random_state=RANDOM_STATE,
            early_stopping=True,
        ),
    }

    rows = []
    fitted_models: dict[str, Pipeline] = {}
    for name, estimator in models.items():
        pipe = Pipeline(
            steps=[
                ("time_aug", TimeAugmenter()),
                ("prep", preprocessor),
                ("model", estimator),
            ]
        )
        pipe.fit(X_train, y_train)
        fitted_models[name] = pipe
        rows.append(evaluate_model(name, pipe, X_test, y_test))

    summary = pd.DataFrame(rows).sort_values("log_loss")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "model_summary.csv"
    summary.to_csv(out_path, index=False)

    # ---- Player attribution from best model ----
    best_model_name = summary.iloc[0]["model"]
    best_model = fitted_models[best_model_name]
    player_credit = build_player_press_credit(best_model, test_df, X_test)
    model_out = RESULTS_DIR / "modeling.csv"
    _write_clean_csv(player_credit, model_out)

    print("\nSaved model summary:", out_path)
    print("Saved modeling ranking:", model_out)
    print(summary)


if __name__ == "__main__":
    main()
