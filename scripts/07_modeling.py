#!/usr/bin/env python3
"""Train competing-risk hazard models for forecheck outcomes.

This script fits three event-level classifiers over hazard rows:
1) Multinomial logistic regression (baseline, interpretable)
2) Histogram gradient boosting (sklearn GBM)
3) XGBoost (tuned via 06_tuning.py; see README)

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
  replacing slot features with a ghost: sample from same RF leaf.
- Slot credit on the row is the drop in press value under this counterfactual.
  Positive credit means that player's pressure increased expected success
  relative to failure.
- Aggregate row credits to player-level totals/means for valuation.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
import xgboost as xgb

import importlib.util
_preprocess_path = Path(__file__).resolve().parent / "05_preprocess.py"
_spec = importlib.util.spec_from_file_location("preprocess", _preprocess_path)
_preprocess = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_preprocess)
FORECHECK_SLOTS = _preprocess.FORECHECK_SLOTS
SLOT_FEATURE_TEMPLATE = _preprocess.SLOT_FEATURE_TEMPLATE
TimeAugmenter = _preprocess.TimeAugmenter
add_slot_imputed_indicators = _preprocess.add_slot_imputed_indicators
build_feature_lists = _preprocess.build_feature_lists
build_preprocessor = _preprocess.build_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hazard_features.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

RANDOM_STATE = 7
N_JOBS = min(12, os.cpu_count() or 12)  # Parallel workers for row chunks and slots
GHOST_DRAWS = 10  # B draws per row; average credits for E[credit] under ghost distribution


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


def load_tuned_params() -> dict[str, dict]:
    """Load best hyperparameters from tuning_results.csv (from 06_tuning.py).

    Returns dict mapping model name to param dict (without model__ prefix).
    Model names: hist_gbm, xgboost. Falls back to empty dict if file missing.
    """
    path = RESULTS_DIR / "tuning_results.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        name = str(row["model"])
        try:
            raw = ast.literal_eval(str(row["best_params"]))
        except (ValueError, SyntaxError):
            continue
        params = {k.replace("model__", ""): v for k, v in raw.items()}
        out[name] = params
    return out


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


def _get_slot_cols(slot: str, df: pd.DataFrame) -> list[str]:
    """Slot feature columns that exist in df."""
    return [c.format(slot=slot) for c in SLOT_FEATURE_TEMPLATE if c.format(slot=slot) in df.columns]


class LeafResampleGhost:
    """Sample from conditional empirical distribution via RF leaf resampling."""

    def __init__(
        self,
        rf: RandomForestRegressor,
        y_train: np.ndarray,
        X_train_processed: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        self.rf = rf
        self.y_train = np.asarray(y_train)
        self.rng = rng
        self.leaf_to_idx: list[dict[int, list[int]]] = []
        for t in rf.estimators_:
            leaf_ids = t.apply(X_train_processed)
            m: dict[int, list[int]] = {}
            for i, leaf in enumerate(leaf_ids):
                m.setdefault(int(leaf), []).append(i)
            self.leaf_to_idx.append(m)
        self.n_trees = len(self.rf.estimators_)

    def sample(self, X_processed: np.ndarray, B: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Sample B ghost draws per row. Returns (samples, valid_mask).

        samples: (B, n, n_slot_features). Rows where valid_mask[b,i]=False are unused.
        valid_mask: (B, n) bool. True where we found a sample in the leaf.
        """
        n = X_processed.shape[0]
        n_feat = self.y_train.shape[1]
        out = np.zeros((B, n, n_feat))
        valid = np.zeros((B, n), dtype=bool)
        tree_order = np.arange(self.n_trees)

        for b in range(B):
            self.rng.shuffle(tree_order)
            for i in range(n):
                for j in tree_order:
                    leaf = self.rf.estimators_[j].apply(X_processed[i : i + 1])[0]
                    idxs = self.leaf_to_idx[j].get(int(leaf), [])
                    if idxs:
                        k = idxs[self.rng.integers(0, len(idxs))]
                        out[b, i, :] = self.y_train[k]
                        valid[b, i] = True
                        break

        return out, valid


def fit_slot_predictors(
    X_train: pd.DataFrame,
    numeric_cols: list[str],
    cat_cols: list[str],
) -> dict[str, tuple[Pipeline, LeafResampleGhost, list[str]]]:
    """Fit one multi-output RF per slot: predict slot features from X_without_slot.

    Uses leaf-resampling ghosts: sample from same RF leaf (any training row).
    Returns dict[slot] -> (pipeline, ghost, input_cols).
    """
    slot_predictors: dict[str, tuple[Pipeline, LeafResampleGhost, list[str]]] = {}
    rf_params = dict(n_estimators=100, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)

    for slot in tqdm(FORECHECK_SLOTS, desc="Slot predictors", unit="slot", file=sys.stdout):
        slot_id = f"{slot}_id"
        slot_cols = _get_slot_cols(slot, X_train)
        if not slot_cols:
            continue

        input_cols = [c for c in numeric_cols + cat_cols if c not in slot_cols and c != slot_id]
        input_cols = [c for c in input_cols if c in X_train.columns]
        if not input_cols:
            continue

        y_out = X_train[slot_cols].astype(float)
        valid = y_out.notna().all(axis=1)
        if valid.sum() < 100:
            continue
        X_in = X_train.loc[valid, input_cols]
        y_out = y_out.loc[valid]

        input_numeric = [c for c in input_cols if c in numeric_cols]
        input_cat = [c for c in input_cols if c in cat_cols]
        transformers = []
        if input_numeric:
            transformers.append(("num", SimpleImputer(strategy="median"), input_numeric))
        if input_cat:
            transformers.append(
                (
                    "cat",
                    Pipeline([
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]),
                    input_cat,
                )
            )
        preprocess = ColumnTransformer(transformers, remainder="drop")
        pipeline = Pipeline([
            ("preprocess", preprocess),
            ("rf", RandomForestRegressor(**rf_params)),
        ])
        pipeline.fit(X_in, y_out)
        preprocess = pipeline.named_steps["preprocess"]
        rf = pipeline.named_steps["rf"]
        X_processed = preprocess.transform(X_in)
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
        ghost = LeafResampleGhost(
            rf,
            y_out.to_numpy(),
            np.asarray(X_processed),
            np.random.default_rng(RANDOM_STATE),
        )
        slot_predictors[slot] = (pipeline, ghost, input_cols)

    return slot_predictors


def _rf_slot_replacement(
    df: pd.DataFrame,
    slot: str,
    slot_predictor: tuple[Pipeline, LeafResampleGhost, list[str]] | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Counterfactual: replace slot features with ghost from same leaf.

    Preserves missingness pattern: only overwrites values that were originally observed,
    leaves NaNs as NaNs so base and CF have identical missing indicators.

    Returns (cf, valid_mask). valid_mask[i]=True where we found a sample.
    """
    cf = df.copy()
    slot_id = f"{slot}_id"
    slot_cols = _get_slot_cols(slot, df)
    n = len(df)
    valid_mask = np.zeros(n, dtype=bool)

    if not slot_cols or slot_predictor is None:
        return cf, valid_mask
    if slot_id in cf.columns:
        cf[slot_id] = np.nan

    # Only process rows with at least one observed slot feature
    observed_any = cf[slot_cols].notna().any(axis=1)
    if not observed_any.any():
        return cf, valid_mask

    pipeline, ghost, input_cols = slot_predictor
    input_cols = [c for c in input_cols if c in df.columns]
    X_in = df.loc[observed_any, input_cols]
    X_processed = pipeline.named_steps["preprocess"].transform(X_in)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    samples, valid = ghost.sample(np.asarray(X_processed), B=1)
    sample_valid = valid[0]
    # Map valid back to full row index
    valid_mask[observed_any] = sample_valid

    if sample_valid.any():
        valid_indices = df.index[observed_any][sample_valid]
        pred = samples[0][sample_valid]  # (n_valid_rows, n_slot_features)
        pred_df = pd.DataFrame(pred, index=valid_indices, columns=slot_cols)
        # Only overwrite values that were originally present (keep NaNs as NaNs)
        for col in slot_cols:
            m = df[col].notna() & df.index.isin(valid_indices)
            if m.any():
                cf.loc[m, col] = pred_df.loc[df.index[m], col].values
    return cf, valid_mask


def _fill_missing_slots_with_rf(
    X: pd.DataFrame,
    slot_predictors: dict[str, tuple[Pipeline, LeafResampleGhost, list[str]]],
    exclude_slot: str | None = None,
) -> pd.DataFrame:
    """Fill missing slot features with conditional RF prediction (E[slot|X]). Returns a copy."""
    out = X.copy()
    for slot, pred in slot_predictors.items():
        if slot == exclude_slot:
            continue
        pipeline, _, input_cols = pred
        slot_cols = _get_slot_cols(slot, X)
        if not slot_cols:
            continue
        input_cols = [c for c in input_cols if c in X.columns]
        if not input_cols:
            continue
        is_missing = X[slot_cols].isna().any(axis=1)
        if not is_missing.any():
            continue
        X_in = out[input_cols]
        pred_vals = pipeline.predict(X_in)
        out.loc[is_missing, slot_cols] = pred_vals[is_missing]
    return out


def _credit_one_slot(
    slot_rank: int,
    slot: str,
    model: Pipeline,
    source_df: pd.DataFrame,
    X_source: pd.DataFrame,
    slot_predictors: dict,
    base_press_value: np.ndarray,
    base_start_by_seq: pd.Series,
    start_index: pd.Series,
) -> pd.DataFrame:
    """Compute credits for one slot (all ghost draws). Used by Parallel over slots."""
    slot_id = f"{slot}_id"
    pred = slot_predictors[slot]

    acc_total = np.zeros(len(source_df))
    acc_start = np.zeros(len(source_df))
    acc_exec = np.zeros(len(source_df))
    acc_count = np.zeros(len(source_df))

    for draw_idx in range(GHOST_DRAWS):
        cf_features, valid_mask = _rf_slot_replacement(X_source, slot, pred)
        # Fill remaining missing slots (others besides the crediting slot) with RF prediction
        cf_features = _fill_missing_slots_with_rf(
            cf_features, slot_predictors, exclude_slot=slot
        )

        cf_press_value = base_press_value.copy()
        if np.any(valid_mask):
            cf_proba = model.predict_proba(cf_features.loc[valid_mask])
            cf_press_value[valid_mask] = cf_proba[:, 1] - cf_proba[:, 2]

        cf_start_by_seq = pd.Series(cf_press_value, index=source_df.index).loc[start_index.values]
        cf_start_by_seq.index = start_index.index

        total_credit = base_press_value - cf_press_value
        base_start_aligned = source_df["fc_sequence_id"].map(base_start_by_seq).to_numpy(dtype=float)
        cf_start_aligned = source_df["fc_sequence_id"].map(cf_start_by_seq).to_numpy(dtype=float)
        start_positioning_credit = base_start_aligned - cf_start_aligned
        execution_credit = total_credit - start_positioning_credit

        acc_total += total_credit
        acc_start += start_positioning_credit
        acc_exec += execution_credit
        acc_count += valid_mask

    denom = np.maximum(acc_count, 1)
    total_credit = acc_total / denom
    start_positioning_credit = acc_start / denom
    execution_credit = acc_exec / denom

    # Fully-missing frames: no evidence, avoid artificial execution penalty (-start)
    slot_cols = _get_slot_cols(slot, X_source)
    if slot_cols:
        fully_missing = X_source[slot_cols].isna().all(axis=1).to_numpy()
        execution_credit[fully_missing] = np.nan
        total_credit[fully_missing] = np.nan

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
    return slot_frame.dropna(subset=["player_id"])


def build_player_press_credit(
    model: Pipeline,
    source_df: pd.DataFrame,
    X_source: pd.DataFrame,
    slot_predictors: dict[str, tuple[Pipeline, LeafResampleGhost, list[str]]] | None = None,
) -> pd.DataFrame:
    """Estimate player-level press value and split into start-state vs execution credit.

    Decomposition for each row follows:
    - start_positioning_credit: effect on sequence-start press value
    - execution_credit: remaining effect beyond sequence start
    - total_press_credit: start_positioning_credit + execution_credit
    """
    # Fill missing slot features with conditional RF prediction before scoring
    if slot_predictors:
        X_for_base = _fill_missing_slots_with_rf(X_source, slot_predictors)
    else:
        X_for_base = X_source

    base_proba = model.predict_proba(X_for_base)
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

    # For each forechecker slot F1..F5, compute leave-one-out credit in parallel
    slot_tasks = [
        (rank, slot)
        for rank, slot in enumerate(FORECHECK_SLOTS)
        if f"{slot}_id" in source_df.columns
        and slot_predictors.get(slot) is not None
    ]
    tqdm.write(f"  Crediting {len(slot_tasks)} slots in parallel (n_jobs={N_JOBS})...")
    credit_rows = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_credit_one_slot)(
            slot_rank=rank,
            slot=slot,
            model=model,
            source_df=source_df,
            X_source=X_source,
            slot_predictors=slot_predictors,
            base_press_value=base_press_value,
            base_start_by_seq=base_start_by_seq,
            start_index=start_index,
        )
        for rank, slot in slot_tasks
    )

    if not credit_rows:
        return pd.DataFrame(
            columns=["player_id", "n_rows", "n_press", "positioning", "execution", "total", "total_per_press"]
        )

    per_row = pd.concat(credit_rows, ignore_index=True)

    # Cap extreme counterfactual deltas (can occur when model extrapolates)
    for col in ["start_positioning_credit", "execution_credit", "total_press_credit"]:
        per_row[col] = per_row[col].clip(lower=-1.0, upper=1.0)

    # Per (player, press): positioning = start value (once per press); execution = mean over frames
    per_press = (
        per_row.groupby(["player_id", "fc_sequence_id"], as_index=False)
        .agg(
            positioning=("start_positioning_credit", "first"),  # same for all rows in press
            execution=("execution_credit", "mean"),  # avg gain/loss over frames player is in
        )
    )
    # execution=nan for fully-missing frames (no evidence); treat as 0 in total
    per_press["execution"] = per_press["execution"].fillna(0.0)
    per_press["total_in_press"] = per_press["positioning"] + per_press["execution"]

    # Sum across presses for player totals
    n_rows = per_row.groupby("player_id").size().reset_index(name="n_rows")
    summary = (
        per_press.groupby("player_id", as_index=False)
        .agg(
            n_press=("fc_sequence_id", "nunique"),
            positioning=("positioning", "sum"),
            execution=("execution", "sum"),
            total=("total_in_press", "sum"),
        )
    )
    summary = summary.merge(n_rows, on="player_id", how="left")
    summary["total_per_press"] = np.where(summary["n_press"] > 0, summary["total"] / summary["n_press"], np.nan)
    summary = summary.sort_values("total", ascending=False).reset_index(drop=True)

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
    out = credit[["player_id", "n_rows", "n_press", "positioning", "execution", "total", "total_per_press"]].copy()
    out = out.rename(columns={
        "positioning": "total_positioning",
        "execution": "total_execution",
        "total": "total_check",
        "total_per_press": "check_per_press",
    })
    out = out.sort_values("check_per_press", ascending=False).reset_index(drop=True)
    out = out.merge(
        players_df[[c for c in [pid_col, name_col, pos_col] if c in players_df.columns]].rename(columns=merge_cols),
        on="player_id",
        how="left",
    )
    out_cols = ["player_id", "player_name", "position", "n_press", "n_rows", "total_positioning", "total_execution", "total_check", "check_per_press"]
    out = out[[c for c in out_cols if c in out.columns]]
    out.to_csv(out_path, index=False)


def main() -> None:
    # ---- Data loading and splitting ----
    print("[1/5] Loading data and splitting...")
    df = load_data()
    add_slot_imputed_indicators(df)
    train_df, test_df = split_groups(df)
    print(f"  Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")

    numeric_cols, cat_cols = build_feature_lists(df)

    X_train = train_df[numeric_cols + cat_cols]
    y_train = train_df["target_class"]
    X_test = test_df[numeric_cols + cat_cols]
    y_test = test_df["target_class"]

    preprocessor = build_preprocessor(numeric_cols, cat_cols)

    # ---- Train competing models ----
    print("\n[2/5] Training hazard models...")
    tuned = load_tuned_params()
    if tuned:
        print("Loaded tuned params from tuning_results.csv")
    hist_defaults = {"max_depth": 8, "learning_rate": 0.05, "max_iter": 300}
    xgb_defaults = {
        "max_depth": 8,
        "learning_rate": 0.05,
        "n_estimators": 800,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
    }

    hist_params = {**hist_defaults, **(tuned.get("hist_gbm") or {})}
    xgb_params = {**xgb_defaults, **(tuned.get("xgboost") or {})}

    models = {
        "multinomial_logit": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "gbm_hist": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            **{k: v for k, v in hist_params.items() if k in {"max_depth", "learning_rate", "max_iter", "min_samples_leaf", "l2_regularization", "max_bins"}},
        ),
        "xgboost": xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **{k: v for k, v in xgb_params.items() if k in {"max_depth", "learning_rate", "n_estimators", "min_child_weight", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"}},
        ),
    }

    rows = []
    fitted_models: dict[str, Pipeline] = {}
    for name, estimator in tqdm(models.items(), desc="  Models", unit="model", file=sys.stdout):
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
    print(f"  Best: {summary.iloc[0]['model']} (log_loss={summary.iloc[0]['log_loss']:.4f})")

    # ---- Fit per-slot RF predictors for counterfactual replacement ----
    print("\n[3/5] Fitting slot predictors (RF per slot)...")
    slot_predictors = fit_slot_predictors(X_train, numeric_cols, cat_cols)
    print(f"  Fitted: {list(slot_predictors.keys())}")
    print(f"  Averaging credits over {GHOST_DRAWS} ghost draws per row")

    # ---- Player attribution from best model ----
    print("\n[4/5] Computing player credits...")
    best_model_name = summary.iloc[0]["model"]
    best_model = fitted_models[best_model_name]
    player_credit = build_player_press_credit(best_model, test_df, X_test, slot_predictors)
    model_out = RESULTS_DIR / "modeling.csv"
    _write_clean_csv(player_credit, model_out)

    print("\n[5/5] Done.")
    print("  Saved model summary:", out_path)
    print("Saved modeling ranking:", model_out)
    print(summary)


if __name__ == "__main__":
    main()
