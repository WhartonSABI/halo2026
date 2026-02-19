#!/usr/bin/env python3
"""Train competing-risk hazard models for forecheck outcomes.

Uses the best model from tuning_results.csv (06_tuning.py) when available;
otherwise trains and compares logit, HistGradientBoosting, and XGBoost.

Target classes:
- 0: no terminal event at this row
- 1: terminal success
- 2: terminal failure

Key modeling choice:
- Uses `time_since_start_s` as a continuous-time covariate with flexible basis
  expansions (log-time and spline basis), so each model learns hazard as a
  function of elapsed continuous time.

Player crediting logic (possession-level value):
- Attribution uses P(success) - P(failure) over the whole possession (CIF, cumulative incidence).
- For each forechecker slot, compute leave-one-out counterfactual: replace slot with ghost from RF leaf.
- Credit = drop in possession-level (CIF_success - CIF_failure) under counterfactual.
- Positioning = effect of start-row slot; execution = effect of non-start rows.
- Aggregate to player-level totals and per-press rates for valuation.
"""

from __future__ import annotations

import argparse
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
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestRegressor
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
    Model names: hist_gbm, gbm, xgboost. Falls back to empty dict if file missing.
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


def load_best_model_from_tuning() -> tuple[str, dict] | None:
    """Load best model name and params from tuning_results.csv (sorted by test_log_loss).

    Returns (model_name, params) or None if tuning_results missing/empty.
    Model names: hist_gbm, gbm, xgboost.
    """
    path = RESULTS_DIR / "tuning_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "model" not in df.columns:
        return None
    df = df.sort_values("test_log_loss").reset_index(drop=True)
    row = df.iloc[0]
    name = str(row["model"])
    try:
        raw = ast.literal_eval(str(row["best_params"]))
    except (ValueError, SyntaxError):
        return None
    params = {k.replace("model__", ""): v for k, v in raw.items()}
    return (name, params)


def _build_model(name: str, params: dict, preprocessor: ColumnTransformer) -> Pipeline:
    """Build a single hazard model pipeline. name: hist_gbm, gbm, or xgboost."""
    hist_keys = {"max_depth", "learning_rate", "max_iter", "min_samples_leaf", "l2_regularization", "max_bins"}
    gbm_keys = {"max_depth", "learning_rate", "n_estimators", "min_samples_leaf", "subsample", "max_features"}
    xgb_keys = {"max_depth", "learning_rate", "n_estimators", "min_child_weight", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"}

    if name == "hist_gbm":
        estimator = HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            **{k: v for k, v in params.items() if k in hist_keys},
        )
    elif name == "gbm":
        estimator = GradientBoostingClassifier(
            random_state=RANDOM_STATE,
            **{k: v for k, v in params.items() if k in gbm_keys},
        )
    elif name == "xgboost":
        estimator = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **{k: v for k, v in params.items() if k in xgb_keys},
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    return Pipeline([
        ("time_aug", TimeAugmenter()),
        ("prep", preprocessor),
        ("model", estimator),
    ])


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


def _compute_start_meta(
    df: pd.DataFrame, time_col: str = "time_since_start_s"
) -> tuple[pd.Series, np.ndarray]:
    """Compute start_index (fc_sequence_id -> row index) and is_start_row (bool per row)."""
    ordering = pd.DataFrame(
        {
            "fc_sequence_id": df["fc_sequence_id"].values,
            "time_since_start_s": df[time_col].astype(float).values,
            "sl_event_id": df["sl_event_id"].values,
        },
        index=df.index,
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
    is_start_row = np.asarray(df.index.isin(start_index.values), dtype=bool)
    return start_index, is_start_row


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
    is_start_row: np.ndarray,
) -> dict[str, tuple[Pipeline, LeafResampleGhost, LeafResampleGhost, list[str]]]:
    """Fit one multi-output RF per slot: predict slot features from X_without_slot.

    Trains two ghost samplers per slot:
    - ghost_start: trained only on sequence-start rows (for positioning credit)
    - ghost_exec: trained on non-start rows (for execution credit)

    Returns dict[slot] -> (pipeline, ghost_start, ghost_exec, input_cols).
    """
    slot_predictors: dict[str, tuple[Pipeline, LeafResampleGhost, LeafResampleGhost, list[str]]] = {}
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
        start_mask = np.asarray(is_start_row, dtype=bool)[valid.values]

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
        X_processed = preprocess.transform(X_in)
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
        X_processed = np.asarray(X_processed)

        # ghost_start: trained only on sequence-start rows
        start_valid = start_mask
        if start_valid.sum() >= 30:
            X_start = X_processed[start_valid]
            y_start = y_out.iloc[start_valid].to_numpy()
            rf_start = RandomForestRegressor(**rf_params).fit(X_start, y_start)
            ghost_start = LeafResampleGhost(rf_start, y_start, X_start, np.random.default_rng(RANDOM_STATE))
        else:
            rf_all = pipeline.named_steps["rf"]
            ghost_start = LeafResampleGhost(rf_all, y_out.to_numpy(), X_processed, np.random.default_rng(RANDOM_STATE))

        # ghost_exec: trained on non-start rows
        exec_mask = ~start_mask
        if exec_mask.sum() >= 30:
            X_exec = X_processed[exec_mask]
            y_exec = y_out.iloc[exec_mask].to_numpy()
            rf_exec = RandomForestRegressor(**rf_params).fit(X_exec, y_exec)
            ghost_exec = LeafResampleGhost(
                rf_exec, y_exec, X_exec, np.random.default_rng(RANDOM_STATE + 1)
            )
        else:
            rf_all = pipeline.named_steps["rf"]
            ghost_exec = LeafResampleGhost(
                rf_all, y_out.to_numpy(), X_processed, np.random.default_rng(RANDOM_STATE + 1)
            )

        slot_predictors[slot] = (pipeline, ghost_start, ghost_exec, input_cols)

    return slot_predictors


def _compute_cif(
    proba: np.ndarray,
    fc_sequence_id: np.ndarray | pd.Series,
    source_df: pd.DataFrame,
) -> pd.DataFrame:
    """Cumulative incidence of success and failure over each possession.

    proba: (n_rows, 3) with columns [ongoing, success, failure]
    Returns DataFrame with fc_sequence_id, cif_success, cif_failure.
    """
    df = source_df.copy()
    df["_h_success"] = proba[:, 1]
    df["_h_failure"] = proba[:, 2]
    df["_h_ongoing"] = 1 - df["_h_success"] - df["_h_failure"]
    out = []
    for seq_id, grp in df.groupby("fc_sequence_id", sort=False):
        grp = grp.sort_values(["time_since_start_s", "sl_event_id"])
        h_s = grp["_h_success"].values
        h_f = grp["_h_failure"].values
        S = 1.0
        cif_s = 0.0
        cif_f = 0.0
        for t in range(len(grp)):
            cif_s += h_s[t] * S
            cif_f += h_f[t] * S
            S *= 1 - h_s[t] - h_f[t]
            if S <= 0:
                break
        out.append({"fc_sequence_id": seq_id, "cif_success": cif_s, "cif_failure": cif_f})
    return pd.DataFrame(out)


def _rf_slot_replacement(
    X_filled: pd.DataFrame,
    X_raw: pd.DataFrame,
    slot: str,
    slot_predictor: tuple[Pipeline, LeafResampleGhost, LeafResampleGhost, list[str]] | None,
    is_start_row: np.ndarray,
    replace_start_only: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Counterfactual: replace observed slot features with ghost from same leaf.

    Starts from X_filled (all NaNs filled). Uses X_raw to determine which positions
    were originally observed. Only overwrites observed positions with ghost values.
    Missing entries stay as RF fill in both base and CF - no asymmetry.

    Returns (cf, valid_mask). valid_mask[i]=True where we found a sample.
    """
    cf = X_filled.copy()
    slot_id = f"{slot}_id"
    slot_cols = _get_slot_cols(slot, X_raw)
    n = len(X_raw)
    valid_mask = np.zeros(n, dtype=bool)

    if not slot_cols or slot_predictor is None:
        return cf, valid_mask
    if slot_id in cf.columns:
        cf[slot_id] = np.nan

    # Only process rows with at least one observed slot feature (from raw)
    observed_any = X_raw[slot_cols].notna().any(axis=1)
    if not observed_any.any():
        return cf, valid_mask

    pipeline, ghost_start, ghost_exec, input_cols = slot_predictor
    input_cols = [c for c in input_cols if c in X_filled.columns]
    X_in = X_filled.loc[observed_any, input_cols]
    X_processed = pipeline.named_steps["preprocess"].transform(X_in)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = np.asarray(X_processed)

    obs_bool = np.asarray(observed_any)
    is_start_obs = np.asarray(is_start_row, dtype=bool)[obs_bool]
    start_pos = np.flatnonzero(is_start_obs)
    exec_pos = np.flatnonzero(~is_start_obs)

    all_valid_indices = []
    all_preds = []

    if len(start_pos) > 0:
        samples_s, valid_s = ghost_start.sample(X_processed[start_pos], B=1)
        sv = valid_s[0]
        if sv.any():
            obs_indices = np.flatnonzero(obs_bool)
            all_valid_indices.append(obs_indices[start_pos][sv])
            all_preds.append(samples_s[0][sv])

    if not replace_start_only and len(exec_pos) > 0:
        samples_e, valid_e = ghost_exec.sample(X_processed[exec_pos], B=1)
        ev = valid_e[0]
        if ev.any():
            obs_indices = np.flatnonzero(obs_bool)
            all_valid_indices.append(obs_indices[exec_pos][ev])
            all_preds.append(samples_e[0][ev])

    if all_valid_indices:
        valid_indices = np.concatenate(all_valid_indices)
        pred = np.vstack(all_preds)
        valid_mask[valid_indices] = True
        valid_labels = X_raw.index[valid_indices]
        pred_df = pd.DataFrame(pred, index=valid_labels, columns=slot_cols)
        for col in slot_cols:
            m = X_raw[col].notna() & X_raw.index.isin(valid_labels)
            if m.any():
                cf.loc[m, col] = pred_df.loc[X_raw.index[m], col].values

    return cf, valid_mask


def _fill_missing_slots_with_rf(
    X: pd.DataFrame,
    slot_predictors: dict[str, tuple[Pipeline, LeafResampleGhost, LeafResampleGhost, list[str]]],
    exclude_slot: str | None = None,
) -> pd.DataFrame:
    """Fill missing slot features with conditional RF prediction (E[slot|X]). Returns a copy."""
    out = X.copy()
    for slot, pred in slot_predictors.items():
        if slot == exclude_slot:
            continue
        pipeline, _, _, input_cols = pred
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


def _get_stints_per_sequence(
    source_df: pd.DataFrame,
    slot_id: str,
    start_index: pd.Series,
    seq_ids: np.ndarray,
) -> dict:
    """Get stint allocations per sequence for a slot.

    Stints = consecutive rows (within sequence, sorted by time) with same non-NaN slot_id.
    Returns dict: seq_id -> list of (player_id, n_non_start_rows, contains_start_row).
    """
    if slot_id not in source_df.columns:
        return {int(sid) if isinstance(sid, (np.integer, np.int64)) else sid: [] for sid in seq_ids}

    out = {}
    for seq_id, grp in source_df.groupby("fc_sequence_id", sort=False):
        if seq_id not in seq_ids:
            continue
        grp = grp.sort_values(["time_since_start_s", "sl_event_id"])
        start_idx = start_index.loc[seq_id] if seq_id in start_index.index else None
        if hasattr(start_idx, "__iter__") and not isinstance(start_idx, str):
            start_idx = start_idx[0] if len(start_idx) else None

        stints = []
        current_player = None
        current_n_non_start = 0
        current_contains_start = False

        for idx, row in grp.iterrows():
            player = row[slot_id]
            is_start = (idx == start_idx) if start_idx is not None else False

            if pd.isna(player):
                if current_player is not None:
                    stints.append((current_player, current_n_non_start, current_contains_start))
                    current_player = None
                    current_n_non_start = 0
                    current_contains_start = False
                continue

            if player == current_player:
                current_n_non_start += 0 if is_start else 1
                current_contains_start = current_contains_start or is_start
            else:
                if current_player is not None:
                    stints.append((current_player, current_n_non_start, current_contains_start))
                current_player = player
                current_n_non_start = 0 if is_start else 1
                current_contains_start = is_start

        if current_player is not None:
            stints.append((current_player, current_n_non_start, current_contains_start))

        out[seq_id] = stints

    for sid in seq_ids:
        if sid not in out:
            out[sid] = []
    return out


def _credit_one_slot(
    slot_rank: int,
    slot: str,
    model: Pipeline,
    source_df: pd.DataFrame,
    X_source: pd.DataFrame,
    X_filled: pd.DataFrame,
    slot_predictors: dict,
    base_proba: np.ndarray,
    start_index: pd.Series,
    is_start_row: np.ndarray,
) -> pd.DataFrame:
    """Possession-level credits for one slot. Credit = effect on P(success)-P(failure) over whole possession.

    Uses CIF (cumulative incidence) over each press. Positioning = effect of start-row slot;
    execution = effect of non-start rows.
    """
    slot_id = f"{slot}_id"
    pred = slot_predictors[slot]
    base_cif = _compute_cif(base_proba, source_df["fc_sequence_id"], source_df)
    base_val = base_cif["cif_success"].values - base_cif["cif_failure"].values

    acc_pos = np.zeros(len(base_cif))
    acc_exec = np.zeros(len(base_cif))
    seq_ids = base_cif["fc_sequence_id"].values

    for draw_idx in range(GHOST_DRAWS):
        cf_start_features, _ = _rf_slot_replacement(
            X_filled, X_source, slot, pred, is_start_row, replace_start_only=True
        )
        cf_full_features, _ = _rf_slot_replacement(
            X_filled, X_source, slot, pred, is_start_row, replace_start_only=False
        )
        cf_start_proba = model.predict_proba(cf_start_features)
        cf_full_proba = model.predict_proba(cf_full_features)
        cf_start_cif = _compute_cif(cf_start_proba, source_df["fc_sequence_id"], source_df)
        cf_full_cif = _compute_cif(cf_full_proba, source_df["fc_sequence_id"], source_df)
        cf_start_val = cf_start_cif["cif_success"].values - cf_start_cif["cif_failure"].values
        cf_full_val = cf_full_cif["cif_success"].values - cf_full_cif["cif_failure"].values
        positioning = base_val - cf_start_val
        execution = cf_start_val - cf_full_val
        acc_pos += positioning
        acc_exec += execution

    denom = GHOST_DRAWS
    positioning = acc_pos / denom
    execution = acc_exec / denom

    # Attribute by stint: positioning to start-stint player, execution split by non-start row share
    stints_by_seq = _get_stints_per_sequence(source_df, slot_id, start_index, seq_ids)
    rows = []
    for i, sid in enumerate(seq_ids):
        stints = stints_by_seq.get(sid, [])
        total_non_start = sum(s[1] for s in stints)
        pos_i = positioning[i]
        exec_i = execution[i]
        for player_id, n_non_start, contains_start in stints:
            pos_credit = pos_i if contains_start else 0.0
            exec_share = n_non_start / total_non_start if total_non_start > 0 else 0.0
            exec_credit = exec_i * exec_share
            total = pos_credit + exec_credit
            if player_id is not None and (pos_credit != 0 or exec_credit != 0):
                rows.append({
                    "player_id": player_id,
                    "fc_sequence_id": sid,
                    "slot": slot,
                    "start_positioning_credit": pos_credit,
                    "execution_credit": exec_credit,
                    "total_press_credit": total,
                })
    slot_frame = pd.DataFrame(rows)
    return slot_frame.dropna(subset=["player_id"]) if len(rows) > 0 else pd.DataFrame(
        columns=["player_id", "fc_sequence_id", "slot", "start_positioning_credit", "execution_credit", "total_press_credit"]
    )


def build_player_press_credit(
    model: Pipeline,
    source_df: pd.DataFrame,
    X_source: pd.DataFrame,
    slot_predictors: dict[str, tuple[Pipeline, LeafResampleGhost, LeafResampleGhost, list[str]]] | None = None,
    save_per_press_path: Path | None = None,
) -> pd.DataFrame:
    """Estimate player-level press value and split into start-state vs execution credit.

    Decomposition for each row follows:
    - start_positioning_credit: effect on sequence-start press value
    - execution_credit: remaining effect beyond sequence start
    - total_press_credit: start_positioning_credit + execution_credit
    """
    if slot_predictors:
        X_filled = _fill_missing_slots_with_rf(X_source, slot_predictors)
    else:
        X_filled = X_source

    base_proba = model.predict_proba(X_filled)
    start_index, is_start_row = _compute_start_meta(source_df)

    slot_tasks = [
        (rank, slot)
        for rank, slot in enumerate(FORECHECK_SLOTS)
        if f"{slot}_id" in source_df.columns
        and slot_predictors.get(slot) is not None
    ]
    tqdm.write(f"  Crediting {len(slot_tasks)} slots in parallel (possession-level CIF)...")
    credit_rows = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_credit_one_slot)(
            slot_rank=rank,
            slot=slot,
            model=model,
            source_df=source_df,
            X_source=X_source,
            X_filled=X_filled,
            slot_predictors=slot_predictors,
            base_proba=base_proba,
            start_index=start_index,
            is_start_row=is_start_row,
        )
        for rank, slot in slot_tasks
    )

    if not credit_rows:
        return pd.DataFrame(
            columns=["player_id", "n_rows", "n_press", "positioning", "execution", "total", "total_per_press"]
        )

    per_slot = pd.concat(credit_rows, ignore_index=True)

    # Cap extreme counterfactual deltas (can occur when model extrapolates)
    for col in ["start_positioning_credit", "execution_credit", "total_press_credit"]:
        per_slot[col] = per_slot[col].clip(lower=-1.0, upper=1.0)

    # Sum across slots when player in multiple slots same press
    per_press = (
        per_slot.groupby(["player_id", "fc_sequence_id"], as_index=False)
        .agg(
            positioning=("start_positioning_credit", "sum"),
            execution=("execution_credit", "sum"),
            total_in_press=("total_press_credit", "sum"),
        )
    )
    if save_per_press_path is not None:
        per_press.to_parquet(save_per_press_path, index=False)
        tqdm.write(f"  Saved player_press to {save_per_press_path}")

    # Diagnostic: possession-level distribution
    if len(per_press) > 0:
        n_press_per_player = per_press.groupby("player_id").size()
        per_press_with_n = per_press.copy()
        per_press_with_n["player_n_press"] = per_press_with_n["player_id"].map(n_press_per_player)
        for col, name in [
            ("positioning", "pos"),
            ("execution", "exec"),
            ("total_in_press", "total"),
        ]:
            s = per_press[col].dropna()
            if len(s) > 0:
                tqdm.write(
                    f"  {name} (possession): mean={s.mean():.4f}, median={s.median():.4f}, "
                    f"pct_pos={100 * (s > 0).mean():.1f}%, n={len(s):,}"
                )
        high = per_press_with_n["player_n_press"] > 10
        low = per_press_with_n["player_n_press"] <= 10
        if high.sum() > 0:
            sh = per_press_with_n.loc[high, "execution"]
            tqdm.write(
                f"  exec from high-n (n_press>10): n={high.sum():,}, pct_pos={100*(sh>0).mean():.1f}%"
            )
        if low.sum() > 0:
            sl = per_press_with_n.loc[low, "execution"]
            tqdm.write(
                f"  exec from low-n (n_press<=10): n={low.sum():,}, pct_pos={100*(sl>0).mean():.1f}%"
            )
        ex = per_press["execution"]
        pos_vals = ex[ex > 0]
        neg_vals = ex[ex < 0]
        if len(pos_vals) > 0 and len(neg_vals) > 0:
            tqdm.write(
                f"  exec magnitude: mean(pos)={pos_vals.mean():.4f}, mean(neg)={neg_vals.mean():.4f}"
            )

    # Player totals; n_rows = frames player participated in (unique rows where player in any slot)
    slot_cols = [f"{s}_id" for s in FORECHECK_SLOTS if f"{s}_id" in source_df.columns]
    with_idx = source_df[["fc_sequence_id"] + slot_cols].reset_index()
    melted = with_idx.melt(
        id_vars=["index", "fc_sequence_id"], value_vars=slot_cols, var_name="_", value_name="player_id"
    ).dropna(subset=["player_id"])
    n_rows = melted.groupby("player_id")["index"].nunique().reset_index(name="n_rows")
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
    summary["execution"] = summary["execution"].fillna(0)
    summary["check_total"] = summary["positioning"] + summary["execution"]
    summary["check_per_press"] = np.where(
        summary["n_press"] > 0, summary["check_total"] / summary["n_press"], np.nan
    )
    summary = summary.sort_values("check_per_press", ascending=False).reset_index(drop=True)

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
    keep = ["player_id", "n_rows", "n_press", "positioning", "execution", "check_total", "check_per_press"]
    out = credit[[c for c in keep if c in credit.columns]].copy()
    out = out.rename(columns={
        "positioning": "pos_total",
        "execution": "exec_total",
    })
    out = out.sort_values("check_per_press", ascending=False).reset_index(drop=True)
    out = out.merge(
        players_df[[c for c in [pid_col, name_col, pos_col] if c in players_df.columns]].rename(columns=merge_cols),
        on="player_id",
        how="left",
    )
    out_cols = [
        "player_id", "player_name", "position", "n_press", "n_rows",
        "pos_total", "exec_total", "check_total", "check_per_press",
    ]
    out = out[[c for c in out_cols if c in out.columns]]
    out.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hazard models and compute player credits")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Subsample to ~N rows (by sequence) for smaller eval. E.g. 5000 for previous setup.",
    )
    args = parser.parse_args()

    # ---- Data loading and splitting ----
    print("[1/5] Loading data and splitting...")
    df = load_data()

    if args.max_rows is not None:
        seq_ids = df["fc_sequence_id"].unique()
        rng = np.random.default_rng(RANDOM_STATE)
        mean_per_seq = len(df) / len(seq_ids)
        n_seq = max(1, min(len(seq_ids), int(args.max_rows / mean_per_seq)))
        keep_seq = rng.choice(seq_ids, size=n_seq, replace=False)
        df = df[df["fc_sequence_id"].isin(keep_seq)].copy()
        print(f"  Subsampled to {len(df):,} rows ({n_seq} sequences)")

    add_slot_imputed_indicators(df)
    train_df, test_df = split_groups(df)
    print(f"  Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")

    numeric_cols, cat_cols = build_feature_lists(df)

    X_train = train_df[numeric_cols + cat_cols]
    y_train = train_df["target_class"]
    X_test = test_df[numeric_cols + cat_cols]
    y_test = test_df["target_class"]

    preprocessor = build_preprocessor(numeric_cols, cat_cols)

    # ---- Train hazard model ----
    # Use tuning winner when available; else train all three and pick best.
    best_model_name: str
    fitted_models: dict[str, Pipeline] = {}
    summary: pd.DataFrame

    tuning_choice = load_best_model_from_tuning()
    if tuning_choice is not None:
        best_model_name, best_params = tuning_choice
        print(f"\n[2/5] Training {best_model_name} (from tuning_results.csv)...")
        pipe = _build_model(best_model_name, best_params, preprocessor)
        pipe.fit(X_train, y_train)
        fitted_models[best_model_name] = pipe
        proba = pipe.predict_proba(X_test)
        ll = log_loss(y_test, proba, labels=[0, 1, 2])
        summary = pd.DataFrame([{
            "model": best_model_name,
            "log_loss": ll,
            "mean_success_hazard": float(proba[:, 1].mean()),
            "mean_failure_hazard": float(proba[:, 2].mean()),
        }])
    else:
        print("\n[2/5] Training hazard models (no tuning_results.csv; comparing all)...")
        tuned = load_tuned_params()
        hist_defaults = {"max_depth": 8, "learning_rate": 0.05, "max_iter": 300}
        xgb_defaults = {
            "max_depth": 8, "learning_rate": 0.05, "n_estimators": 800,
            "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 1.0,
            "reg_alpha": 1.0, "reg_lambda": 1.0,
        }
        hist_params = {**hist_defaults, **(tuned.get("hist_gbm") or {})}
        xgb_params = {**xgb_defaults, **(tuned.get("xgboost") or {})}

        models = {
            "multinomial_logit": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            "gbm_hist": HistGradientBoostingClassifier(
                random_state=RANDOM_STATE,
                **{k: v for k, v in hist_params.items() if k in {"max_depth", "learning_rate", "max_iter", "min_samples_leaf", "l2_regularization", "max_bins"}},
            ),
            "xgboost": xgb.XGBClassifier(
                objective="multi:softprob", num_class=3, random_state=RANDOM_STATE, n_jobs=-1,
                **{k: v for k, v in xgb_params.items() if k in {"max_depth", "learning_rate", "n_estimators", "min_child_weight", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"}},
            ),
        }
        rows = []
        for name, estimator in tqdm(models.items(), desc="  Models", unit="model", file=sys.stdout):
            pipe = Pipeline([
                ("time_aug", TimeAugmenter()),
                ("prep", preprocessor),
                ("model", estimator),
            ])
            pipe.fit(X_train, y_train)
            fitted_models[name] = pipe
            rows.append(evaluate_model(name, pipe, X_test, y_test))
        summary = pd.DataFrame(rows).sort_values("log_loss")
        best_model_name = summary.iloc[0]["model"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "model_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"  Model: {best_model_name} (log_loss={summary.iloc[0]['log_loss']:.4f})")

    # ---- Fit per-slot RF predictors for counterfactual replacement ----
    print("\n[3/5] Fitting slot predictors (RF per slot, start vs exec ghosts)...")
    _, train_is_start = _compute_start_meta(train_df)
    slot_predictors = fit_slot_predictors(X_train, numeric_cols, cat_cols, train_is_start)
    print(f"  Fitted: {list(slot_predictors.keys())}")
    print(f"  Averaging credits over {GHOST_DRAWS} ghost draws per row")

    # ---- Slot-change audit (stint attribution relevance) ----
    slot_ids = [f"{s}_id" for s in FORECHECK_SLOTS if f"{s}_id" in df.columns]
    if slot_ids:
        frac_changing = (
            df.groupby("fc_sequence_id")[slot_ids]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
            .mean()
        )
        print(f"  Audit: fraction of sequences with slot changes (any slot): {frac_changing:.4f}")

    # ---- Player attribution from best model (on full data, like participation/distance) ----
    print("\n[4/5] Computing player credits on full data...")
    best_model_name = summary.iloc[0]["model"]
    best_model = fitted_models[best_model_name]
    X_all = df[numeric_cols + cat_cols]
    player_credit = build_player_press_credit(
        best_model, df, X_all, slot_predictors,
        save_per_press_path=RESULTS_DIR / "player_press.parquet",
    )
    model_out = RESULTS_DIR / "modeling.csv"
    _write_clean_csv(player_credit, model_out)

    print("\n[5/5] Done.")
    print("  Saved model summary:", out_path)
    print("Saved modeling ranking:", model_out)
    print(summary)


if __name__ == "__main__":
    main()
