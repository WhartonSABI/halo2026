#!/usr/bin/env python3
"""Train hybrid forecheck model and compute player credits.

Pipeline:
1. Start model (XGBoost): P(sequence success) given start config. Calibrated by construction.
2. Hazard model (XGBoost, exec rows only): event-level hazards q_s(t), q_f(t). CIF over exec phase.
3. Slot predictors (RF): per-slot ghost samplers for counterfactual replacement.

Crediting:
- Positioning total = p₀ - p̄ (deviation from population avg). Allocate by ghost shares.
- Execution total = outcome - p₀ (surprise vs start expectation). Allocate by hazard shares.
- Ghosts define counterfactuals → deltas → shares. Totals are fixed (p₀-p̄, outcome-p₀).
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
)

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
import xgboost as xgb

from _preprocess import (
    FORECHECK_SLOTS,
    RANDOM_STATE,
    SLOT_FEATURE_TEMPLATE,
    add_slot_imputed_indicators,
    build_feature_lists,
    build_preprocessor,
    compute_start_meta,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hazard_features.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
N_JOBS = min(12, os.cpu_count() or 12)
GHOST_DRAWS = 10


def load_data() -> pd.DataFrame:
    """Load hazard features and define 3-class target."""
    df = pd.read_parquet(DATA_PATH)
    y = np.zeros(len(df), dtype=np.int64)
    y[df["event_t"].eq(1).values] = 1
    y[df["terminal_failure_t"].eq(1).values] = 2
    df["target_class"] = y
    return df


def split_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test split by fc_sequence_id."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(splitter.split(df, groups=df["fc_sequence_id"]))
    return df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()


def _read_tuning_csv() -> pd.DataFrame | None:
    """Load tuning results from tuning_quick.csv or tuning_full.csv."""
    for fname in ("tuning_full.csv", "tuning_quick.csv"):
        p = RESULTS_DIR / fname
        if p.exists():
            df = pd.read_csv(p)
            return df if not df.empty and "model" in df.columns else None
    return None


def load_best_model_from_tuning() -> tuple[str, dict] | None:
    """Load best hazard model from tuning. Excludes p0_xgb. Returns (model_name, params) or None."""
    df = _read_tuning_csv()
    if df is None:
        return None
    hazard_df = df[~df["model"].eq("p0_xgb")]
    if hazard_df.empty:
        return None
    hazard_df = hazard_df.sort_values("test_log_loss").reset_index(drop=True)
    row = hazard_df.iloc[0]
    name = str(row["model"])
    try:
        raw = ast.literal_eval(str(row["best_params"]))
    except (ValueError, SyntaxError):
        return None
    params = {k.replace("model__", ""): v for k, v in raw.items()}
    return (name, params)


def load_best_start_model_from_tuning() -> dict | None:
    """Load p₀ (start) model params from tuning. Returns params dict or None."""
    df = _read_tuning_csv()
    if df is None:
        return None
    p0_row = df[df["model"].eq("p0_xgb")]
    if p0_row.empty:
        return None
    try:
        raw = ast.literal_eval(str(p0_row.iloc[0]["best_params"]))
    except (ValueError, SyntaxError):
        return None
    return {k.replace("model__", ""): v for k, v in raw.items()}


def _build_model(name: str, params: dict, preprocessor: ColumnTransformer) -> Pipeline:
    """Build hazard model pipeline. name: rf, hist_gbm, or xgboost."""
    from _preprocess import TimeAugmenter

    hist_keys = {"max_depth", "learning_rate", "max_iter", "min_samples_leaf", "l2_regularization", "max_bins"}
    rf_keys = {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}
    xgb_keys = {"max_depth", "learning_rate", "n_estimators", "min_child_weight", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"}

    if name == "hist_gbm":
        estimator = HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            **{k: v for k, v in params.items() if k in hist_keys},
        )
    elif name == "rf":
        estimator = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **{k: v for k, v in params.items() if k in rf_keys},
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


def fit_hybrid(
    train_df: pd.DataFrame,
    numeric_cols: list[str],
    cat_cols: list[str],
    calibrate: bool = True,
) -> tuple[Pipeline, Pipeline]:
    """Fit start + hazard models. If calibrate=True, use 80% fit / 20% calibrate. Returns (start_pipe, hazard_pipe)."""
    from _preprocess import TimeAugmenter

    if calibrate:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE + 100)
        fit_idx, cal_idx = next(splitter.split(train_df, groups=train_df["fc_sequence_id"]))
        train_fit = train_df.iloc[fit_idx]
        train_cal = train_df.iloc[cal_idx]
    else:
        train_fit = train_df
        train_cal = None
    _, train_fit_is_start = _compute_start_meta(train_fit)
    if train_cal is not None:
        _, train_cal_is_start = _compute_start_meta(train_cal)

    # Start model
    seq_outcome = train_fit.groupby("fc_sequence_id")["event_t"].max().reset_index()
    seq_outcome["success"] = (seq_outcome["event_t"] == 1).astype(np.int32)
    start_index, _ = _compute_start_meta(train_fit)
    start_row_idx = start_index.values
    start_seq_ids = train_fit.loc[start_row_idx, "fc_sequence_id"].values
    y_train_start = seq_outcome.set_index("fc_sequence_id").loc[start_seq_ids, "success"].values
    X_train_start = train_fit.loc[start_row_idx, numeric_cols + cat_cols]
    preprocessor_start = build_preprocessor(numeric_cols, cat_cols)
    preprocessor_start.fit(X_train_start)
    start_params = load_best_start_model_from_tuning()
    start_defaults = {
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
        "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
    }
    if start_params:
        start_defaults.update({k: v for k, v in start_params.items() if k in start_defaults})
    start_xgb = xgb.XGBClassifier(
        objective="binary:logistic",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **start_defaults,
    )
    start_pipe = Pipeline([
        ("time_aug", TimeAugmenter()),
        ("prep", preprocessor_start),
        ("model", start_xgb),
    ])
    start_pipe.fit(X_train_start, y_train_start)

    if calibrate and train_cal is not None:
        start_cal_idx = _compute_start_meta(train_cal)[0].values
        X_cal_start = train_cal.loc[start_cal_idx, numeric_cols + cat_cols]
        seq_cal = train_cal.groupby("fc_sequence_id")["event_t"].max().reset_index()
        seq_cal["success"] = (seq_cal["event_t"] == 1).astype(np.int32)
        y_cal_start = seq_cal.set_index("fc_sequence_id").loc[
            train_cal.loc[start_cal_idx, "fc_sequence_id"].values, "success"
        ].values
        start_pipe = CalibratedClassifierCV(FrozenEstimator(start_pipe), method="isotonic", cv=5)
        start_pipe.fit(X_cal_start, y_cal_start)

    # Hazard model
    train_fit_exec_mask = ~train_fit_is_start
    X_train_exec = train_fit.loc[train_fit_exec_mask, numeric_cols + cat_cols]
    y_train_exec = train_fit.loc[train_fit_exec_mask, "target_class"]
    preprocessor_exec = build_preprocessor(numeric_cols, cat_cols)
    preprocessor_exec.fit(X_train_exec)
    tuning_choice = load_best_model_from_tuning()
    if tuning_choice is None:
        best_model_name, best_params = "xgboost", {
            "max_depth": 6, "learning_rate": 0.05, "n_estimators": 300,
            "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
        }
    else:
        best_model_name, best_params = tuning_choice
    hazard_pipe = _build_model(best_model_name, best_params, preprocessor_exec)
    hazard_pipe.fit(X_train_exec, y_train_exec)

    if calibrate and train_cal is not None:
        train_cal_exec_mask = ~train_cal_is_start
        X_cal_exec = train_cal.loc[train_cal_exec_mask, numeric_cols + cat_cols]
        y_cal_exec = train_cal.loc[train_cal_exec_mask, "target_class"]
        hazard_pipe = CalibratedClassifierCV(FrozenEstimator(hazard_pipe), method="isotonic", cv=5)
        hazard_pipe.fit(X_cal_exec, y_cal_exec)

    return start_pipe, hazard_pipe


def fit_and_calibrate_hybrid(train_df, numeric_cols, cat_cols):
    """Thin wrapper: always calibrate."""
    return fit_hybrid(train_df, numeric_cols, cat_cols, calibrate=True)


def _get_slot_cols(slot: str, df: pd.DataFrame) -> list[str]:
    return [c.format(slot=slot) for c in SLOT_FEATURE_TEMPLATE if c.format(slot=slot) in df.columns]


def _compute_start_meta(df: pd.DataFrame, time_col: str = "time_since_start_s") -> tuple[pd.Series, np.ndarray]:
    return compute_start_meta(df, time_col)


class LeafResampleGhost:
    def __init__(self, rf, y_train, X_train_processed, rng):
        self.rf = rf
        self.y_train = np.asarray(y_train)
        self.rng = rng
        self.leaf_to_idx = []
        for t in rf.estimators_:
            leaf_ids = t.apply(X_train_processed)
            m = {}
            for i, leaf in enumerate(leaf_ids):
                m.setdefault(int(leaf), []).append(i)
            self.leaf_to_idx.append(m)
        self.n_trees = len(self.rf.estimators_)

    def sample(self, X_processed, B=1):
        n, n_feat = X_processed.shape[0], self.y_train.shape[1]
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


def fit_slot_predictors(X_train, numeric_cols, cat_cols, is_start_row):
    slot_predictors = {}
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
            transformers.append((
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                input_cat,
            ))
        preprocess = ColumnTransformer(transformers, remainder="drop")
        pipeline = Pipeline([("preprocess", preprocess), ("rf", RandomForestRegressor(**rf_params))])
        pipeline.fit(X_in, y_out)
        preprocess = pipeline.named_steps["preprocess"]
        X_processed = preprocess.transform(X_in)
        if hasattr(X_processed, "toarray"):
            X_processed = np.asarray(X_processed)
        else:
            X_processed = np.asarray(X_processed)
        if start_mask.sum() >= 30:
            X_start = X_processed[start_mask]
            y_start = y_out.iloc[start_mask].to_numpy()
            rf_start = RandomForestRegressor(**rf_params).fit(X_start, y_start)
            ghost_start = LeafResampleGhost(rf_start, y_start, X_start, np.random.default_rng(RANDOM_STATE))
        else:
            ghost_start = LeafResampleGhost(
                pipeline.named_steps["rf"], y_out.to_numpy(), X_processed, np.random.default_rng(RANDOM_STATE)
            )
        exec_mask = ~start_mask
        if exec_mask.sum() >= 30:
            X_exec = X_processed[exec_mask]
            y_exec = y_out.iloc[exec_mask].to_numpy()
            rf_exec = RandomForestRegressor(**rf_params).fit(X_exec, y_exec)
            ghost_exec = LeafResampleGhost(rf_exec, y_exec, X_exec, np.random.default_rng(RANDOM_STATE + 1))
        else:
            ghost_exec = LeafResampleGhost(
                pipeline.named_steps["rf"], y_out.to_numpy(), X_processed, np.random.default_rng(RANDOM_STATE + 1)
            )
        slot_predictors[slot] = (pipeline, ghost_start, ghost_exec, input_cols)
    return slot_predictors


def _compute_cif_exec(proba, source_df, is_start_row):
    """CIF over execution rows only. S=1 at exec start; integrates hazards to P(success), P(failure)."""
    df = source_df.copy()
    df["_h_success"] = proba[:, 1]
    df["_h_failure"] = proba[:, 2]
    df["_is_start"] = is_start_row
    exec_df = df[~df["_is_start"]]
    all_seq_ids = source_df["fc_sequence_id"].unique()
    out = {sid: {"cif_success": 0.0, "cif_failure": 0.0} for sid in all_seq_ids}
    if exec_df.empty:
        return pd.DataFrame([{"fc_sequence_id": s, "cif_success": 0.0, "cif_failure": 0.0} for s in all_seq_ids])
    for seq_id, grp in exec_df.groupby("fc_sequence_id", sort=False):
        grp = grp.sort_values(["time_since_start_s", "sl_event_id"])
        h_s, h_f = grp["_h_success"].values, grp["_h_failure"].values
        S, cif_s, cif_f = 1.0, 0.0, 0.0
        for t in range(len(grp)):
            cif_s += h_s[t] * S
            cif_f += h_f[t] * S
            S *= 1 - h_s[t] - h_f[t]
            if S <= 0:
                break
        out[seq_id] = {"cif_success": cif_s, "cif_failure": cif_f}
    return pd.DataFrame([{"fc_sequence_id": s, "cif_success": out[s]["cif_success"], "cif_failure": out[s]["cif_failure"]} for s in all_seq_ids])


def _rf_slot_replacement(X_filled, X_raw, slot, slot_predictor, is_start_row, replace_start_only=False):
    """Replace slot features with ghost sample. Ghost = draw from RF leaf (conditional on context)."""
    cf = X_filled.copy()
    slot_id = f"{slot}_id"
    slot_cols = _get_slot_cols(slot, X_raw)
    n = len(X_raw)
    valid_mask = np.zeros(n, dtype=bool)
    if not slot_cols or slot_predictor is None:
        return cf, valid_mask
    if slot_id in cf.columns:
        cf[slot_id] = np.nan
    observed_any = X_raw[slot_cols].notna().any(axis=1)
    if not observed_any.any():
        return cf, valid_mask
    pipeline, ghost_start, ghost_exec, input_cols = slot_predictor
    input_cols = [c for c in input_cols if c in X_filled.columns]
    X_in = X_filled.loc[observed_any, input_cols]
    X_processed = pipeline.named_steps["preprocess"].transform(X_in)
    if hasattr(X_processed, "toarray"):
        X_processed = np.asarray(X_processed)
    else:
        X_processed = np.asarray(X_processed)
    obs_bool = np.asarray(observed_any)
    is_start_obs = np.asarray(is_start_row, dtype=bool)[obs_bool]
    start_pos, exec_pos = np.flatnonzero(is_start_obs), np.flatnonzero(~is_start_obs)
    all_valid_indices, all_preds = [], []
    if len(start_pos) > 0:
        samples_s, valid_s = ghost_start.sample(X_processed[start_pos], B=1)
        if valid_s[0].any():
            obs_indices = np.flatnonzero(obs_bool)
            all_valid_indices.append(obs_indices[start_pos][valid_s[0]])
            all_preds.append(samples_s[0][valid_s[0]])
    if not replace_start_only and len(exec_pos) > 0:
        samples_e, valid_e = ghost_exec.sample(X_processed[exec_pos], B=1)
        if valid_e[0].any():
            obs_indices = np.flatnonzero(obs_bool)
            all_valid_indices.append(obs_indices[exec_pos][valid_e[0]])
            all_preds.append(samples_e[0][valid_e[0]])
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


def _fill_missing_slots_with_rf(X, slot_predictors, exclude_slot=None):
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
        pred_vals = pipeline.predict(out[input_cols])
        out.loc[is_missing, slot_cols] = pred_vals[is_missing]
    return out


def _get_stints_per_sequence(source_df, slot_id, start_index, seq_ids):
    if slot_id not in source_df.columns:
        return {sid: [] for sid in seq_ids}
    out = {}
    for seq_id, grp in source_df.groupby("fc_sequence_id", sort=False):
        if seq_id not in seq_ids:
            continue
        grp = grp.sort_values(["time_since_start_s", "sl_event_id"])
        start_idx = start_index.loc[seq_id] if seq_id in start_index.index else None
        if hasattr(start_idx, "__iter__") and not isinstance(start_idx, str):
            start_idx = start_idx[0] if len(start_idx) else None
        stints, current_player, current_n_non_start, current_contains_start = [], None, 0, False
        for idx, row in grp.iterrows():
            player, is_start = row[slot_id], (idx == start_idx) if start_idx is not None else False
            if pd.isna(player):
                if current_player is not None:
                    stints.append((current_player, current_n_non_start, current_contains_start))
                    current_player, current_n_non_start, current_contains_start = None, 0, False
                continue
            if player == current_player:
                current_n_non_start += 0 if is_start else 1
                current_contains_start = current_contains_start or is_start
            else:
                if current_player is not None:
                    stints.append((current_player, current_n_non_start, current_contains_start))
                current_player, current_n_non_start, current_contains_start = player, (0 if is_start else 1), is_start
        if current_player is not None:
            stints.append((current_player, current_n_non_start, current_contains_start))
        out[seq_id] = stints
    for sid in seq_ids:
        if sid not in out:
            out[sid] = []
    return out


def _credit_one_slot(slot_rank, slot, hazard_model, start_model, source_df, X_source, X_filled,
                    slot_predictors, base_start_val, base_exec_cif, start_index, is_start_row,
                    start_row_idx):
    """Compute Δ_pos and Δ_exec for one slot (ghost-averaged). Deltas used later for shares."""
    slot_id = f"{slot}_id"
    pred = slot_predictors[slot]
    seq_ids = base_exec_cif["fc_sequence_id"].values
    acc_delta_pos = np.zeros(len(seq_ids))
    acc_delta_exec = np.zeros(len(seq_ids))
    base_exec_val = base_exec_cif["cif_success"].values - base_exec_cif["cif_failure"].values
    for _ in range(GHOST_DRAWS):
        cf_start_features, _ = _rf_slot_replacement(X_filled, X_source, slot, pred, is_start_row, replace_start_only=True)
        cf_full_features, _ = _rf_slot_replacement(X_filled, X_source, slot, pred, is_start_row, replace_start_only=False)
        X_start_cf = cf_start_features.loc[start_row_idx]
        cf_start_val = start_model.predict_proba(X_start_cf)[:, 1]
        cf_full_proba = hazard_model.predict_proba(cf_full_features)
        cf_exec_cif = _compute_cif_exec(cf_full_proba, source_df, is_start_row)
        cf_exec_aligned = cf_exec_cif.set_index("fc_sequence_id").reindex(seq_ids).fillna(0)
        cf_exec_val = cf_exec_aligned["cif_success"].values - cf_exec_aligned["cif_failure"].values
        acc_delta_pos += base_start_val - cf_start_val
        acc_delta_exec += base_exec_val - cf_exec_val
    delta_pos = acc_delta_pos / GHOST_DRAWS
    delta_exec = acc_delta_exec / GHOST_DRAWS
    return (slot, seq_ids, delta_pos, delta_exec)


def build_player_press_credit(hazard_model, start_model, source_df, X_source, slot_predictors,
                              save_per_press_path=None):
    """Crediting: positioning total=p₀-p̄ (allocate by ghost shares), exec total=outcome-p₀ (allocate by hazard shares)."""
    X_filled = _fill_missing_slots_with_rf(X_source, slot_predictors) if slot_predictors else X_source
    start_index, is_start_row = _compute_start_meta(source_df)

    # p₀, p̄, outcome per sequence
    seq_ids_ordered = source_df["fc_sequence_id"].drop_duplicates().values
    start_row_idx = start_index.values
    X_start = X_filled.loc[start_row_idx]
    base_start_proba = start_model.predict_proba(X_start)[:, 1]
    start_seq_ids = source_df.loc[start_row_idx, "fc_sequence_id"].values
    base_start_by_seq = pd.Series(base_start_proba, index=start_seq_ids)
    p0 = base_start_by_seq.reindex(seq_ids_ordered).fillna(0).values

    seq_outcome = source_df.groupby("fc_sequence_id")["event_t"].max()
    outcome = (seq_outcome.reindex(seq_ids_ordered).fillna(0) == 1).astype(np.float64).values
    p_bar = float(outcome.mean())

    base_proba = hazard_model.predict_proba(X_filled)
    base_exec_cif = _compute_cif_exec(base_proba, source_df, is_start_row)
    base_exec_cif = base_exec_cif.set_index("fc_sequence_id").reindex(seq_ids_ordered).fillna(0).reset_index()
    base_start_val = base_start_by_seq.reindex(seq_ids_ordered).fillna(0).values
    start_row_idx_ordered = start_index.reindex(seq_ids_ordered).astype(np.int64).values

    slot_tasks = [(r, s) for r, s in enumerate(FORECHECK_SLOTS)
                  if f"{s}_id" in source_df.columns and slot_predictors and slot_predictors.get(s)]
    tqdm.write(f"  Crediting {len(slot_tasks)} slots (p₀-p̄ positioning, outcome-p₀ exec, ghost shares)...")
    task_args = [
        (r, s, hazard_model, start_model, source_df, X_source, X_filled,
         slot_predictors, base_start_val, base_exec_cif, start_index, is_start_row,
         start_row_idx_ordered)
        for r, s in slot_tasks
    ]
    slot_results = [None] * len(slot_tasks)
    n_workers = min(N_JOBS, len(slot_tasks))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        future_to_idx = {ex.submit(_credit_one_slot, *args): i for i, args in enumerate(task_args)}
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx),
                           desc="  Slots", unit="slot", leave=False):
            idx = future_to_idx[future]
            slot_results[idx] = future.result()
    if not slot_results:
        return pd.DataFrame(columns=["player_id", "n_rows", "n_press", "positioning", "execution", "total", "total_per_press"])

    # Stack deltas: [n_slots x n_seq]
    n_seq = len(seq_ids_ordered)
    n_slots = len(slot_results)
    delta_pos = np.zeros((n_slots, n_seq))
    delta_exec = np.zeros((n_slots, n_seq))
    slot_names = []
    for idx, (slot, seq_ids, dp, de) in enumerate(slot_results):
        slot_names.append(slot)
        delta_pos[idx, :] = dp
        delta_exec[idx, :] = de

    # Allocate fixed totals by ghost-based shares
    _eps = 1e-12
    sum_pos = np.sum(delta_pos, axis=0)
    sum_exec = np.sum(delta_exec, axis=0)
    share_pos = delta_pos / (sum_pos + _eps)
    share_exec = delta_exec / (sum_exec + _eps)

    # pos_total = p₀ - p̄ ∈ [-p̄, 1-p̄] (e.g. [-0.35, 0.65] if p̄≈0.35); exec_total = outcome - p₀ ∈ [-1, 1]
    pos_total = p0 - p_bar
    exec_total = outcome - p0
    pos_credit_slot = share_pos * pos_total
    exec_credit_slot = share_exec * exec_total

    # Fix 4: Reallocate empty-slot credit to filled slots (e.g. F5 in 4v4 has no player → reallocate to F1-F4)
    stints_by_slot_seq = {}
    for slot_idx, slot in enumerate(slot_names):
        slot_id = f"{slot}_id"
        stints_by_slot_seq[slot_idx] = _get_stints_per_sequence(source_df, slot_id, start_index, seq_ids_ordered)
    has_stints = np.zeros((n_slots, n_seq), dtype=bool)
    for slot_idx in range(n_slots):
        stints_by_seq = stints_by_slot_seq[slot_idx]
        for i, sid in enumerate(seq_ids_ordered):
            stints = stints_by_seq.get(sid, [])
            has_stints[slot_idx, i] = len(stints) > 0
    for i in range(n_seq):
        filled = has_stints[:, i]
        empty = ~filled
        if empty.any():
            orphaned_pos = pos_credit_slot[empty, i].sum()
            orphaned_exec = exec_credit_slot[empty, i].sum()
            if filled.any():
                delta_pos_filled = np.where(filled, delta_pos[:, i], 0.0)
                delta_exec_filled = np.where(filled, delta_exec[:, i], 0.0)
                sum_pos_f = delta_pos_filled.sum()
                sum_exec_f = delta_exec_filled.sum()
                share_pos_f = np.where(filled, delta_pos[:, i] / (sum_pos_f + _eps), 0.0)
                share_exec_f = np.where(filled, delta_exec[:, i] / (sum_exec_f + _eps), 0.0)
                pos_credit_slot[:, i] = np.where(empty, 0.0, pos_credit_slot[:, i] + share_pos_f * orphaned_pos)
                exec_credit_slot[:, i] = np.where(empty, 0.0, exec_credit_slot[:, i] + share_exec_f * orphaned_exec)
            else:
                pos_credit_slot[:, i] = 0.0
                exec_credit_slot[:, i] = 0.0

    # Stint attribution: positioning to start-row player; exec split by n_non_start share
    rows = []
    for slot_idx, slot in enumerate(slot_names):
        stints_by_seq = stints_by_slot_seq[slot_idx]
        for i, sid in enumerate(seq_ids_ordered):
            stints = stints_by_seq.get(sid, [])
            total_non_start = sum(s[1] for s in stints)
            pos_i = pos_credit_slot[slot_idx, i]
            exec_i = exec_credit_slot[slot_idx, i]
            for player_id, n_non_start, contains_start in stints:
                pos_credit = pos_i if contains_start else 0.0
                exec_share = (
                    n_non_start / total_non_start
                    if total_non_start > 0
                    else 1.0 / len(stints)  # no tracking for exec: even split among stints
                )
                exec_credit = exec_i * exec_share
                total = pos_credit + exec_credit
                if player_id is not None and (pos_credit != 0 or exec_credit != 0):
                    rows.append({
                        "player_id": player_id, "fc_sequence_id": sid, "slot": slot,
                        "start_positioning_credit": pos_credit, "execution_credit": exec_credit, "total_press_credit": total,
                    })
    if not rows:
        return pd.DataFrame(columns=["player_id", "n_rows", "n_press", "positioning", "execution", "total", "total_per_press"])
    per_slot = pd.DataFrame(rows)
    for col in ["start_positioning_credit", "execution_credit", "total_press_credit"]:
        per_slot[col] = per_slot[col].clip(lower=-1.0, upper=1.0)
    if len(per_slot) > 0:
        exec_per_slot = per_slot.groupby(["fc_sequence_id", "slot"], as_index=False).agg(
            exec=("execution_credit", "sum"),
        )
        total_exec = exec_per_slot["exec"].sum()
        tqdm.write(f"  SUM(exec) = {total_exec:.6f}  (should be ≈0: outcome - p₀ by construction)")
    per_press = per_slot.groupby(["player_id", "fc_sequence_id"], as_index=False).agg(
        positioning=("start_positioning_credit", "sum"),
        execution=("execution_credit", "sum"),
        total_in_press=("total_press_credit", "sum"),
    )
    if save_per_press_path:
        per_press.to_parquet(save_per_press_path, index=False)
        tqdm.write(f"  Saved player_press to {save_per_press_path}")
    if len(per_press) > 0:
        n_press_per_player = per_press.groupby("player_id").size()
        per_press_with_n = per_press.copy()
        per_press_with_n["player_n_press"] = per_press_with_n["player_id"].map(n_press_per_player)
        for col, name in [("positioning", "pos"), ("execution", "exec"), ("total_in_press", "total")]:
            s = per_press[col].dropna()
            if len(s) > 0:
                tqdm.write(f"  {name} (possession): mean={s.mean():.4f}, median={s.median():.4f}, pct_pos={100*(s>0).mean():.1f}%, n={len(s):,}")
        high = per_press_with_n["player_n_press"] > 10
        if high.sum() > 0:
            tqdm.write(f"  exec from high-n (n_press>10): n={high.sum():,}, pct_pos={100*(per_press_with_n.loc[high,'execution']>0).mean():.1f}%")
        low = per_press_with_n["player_n_press"] <= 10
        if low.sum() > 0:
            tqdm.write(f"  exec from low-n (n_press<=10): n={low.sum():,}, pct_pos={100*(per_press_with_n.loc[low,'execution']>0).mean():.1f}%")
        ex = per_press["execution"]
        pos_vals, neg_vals = ex[ex > 0], ex[ex < 0]
        if len(pos_vals) > 0 and len(neg_vals) > 0:
            tqdm.write(f"  exec magnitude: mean(pos)={pos_vals.mean():.4f}, mean(neg)={neg_vals.mean():.4f}")
    slot_cols = [f"{s}_id" for s in FORECHECK_SLOTS if f"{s}_id" in source_df.columns]
    with_idx = source_df[["fc_sequence_id"] + slot_cols].reset_index()
    melted = with_idx.melt(id_vars=["index", "fc_sequence_id"], value_vars=slot_cols, var_name="_", value_name="player_id").dropna(subset=["player_id"])
    n_rows = melted.groupby("player_id")["index"].nunique().reset_index(name="n_rows")
    summary = per_press.groupby("player_id", as_index=False).agg(
        n_press=("fc_sequence_id", "nunique"),
        positioning=("positioning", "sum"),
        execution=("execution", "sum"),
        total=("total_in_press", "sum"),
    )
    summary = summary.merge(n_rows, on="player_id", how="left")
    summary["execution"] = summary["execution"].fillna(0)
    summary["check_total"] = summary["positioning"] + summary["execution"]
    summary["check_per_press"] = np.where(summary["n_press"] > 0, summary["check_total"] / summary["n_press"], np.nan)
    return summary.sort_values("check_per_press", ascending=False).reset_index(drop=True)


def _write_clean_csv(credit, out_path):
    players_df = pd.read_parquet(RAW_DIR / "players.parquet")
    pid_col = "player_id" if "player_id" in players_df.columns else "id"
    name_col = "player_name" if "player_name" in players_df.columns else "name"
    pos_col = "primary_position" if "primary_position" in players_df.columns else "position"
    merge_cols = {pid_col: "player_id", name_col: "player_name"}
    if pos_col in players_df.columns:
        merge_cols[pos_col] = "position"
    keep = ["player_id", "n_rows", "n_press", "positioning", "execution", "check_total", "check_per_press"]
    out = credit[[c for c in keep if c in credit.columns]].rename(columns={"positioning": "pos_total", "execution": "exec_total"})
    out = out.sort_values("check_per_press", ascending=False).merge(
        players_df[[c for c in [pid_col, name_col, pos_col] if c in players_df.columns]].rename(columns=merge_cols),
        on="player_id", how="left",
    )
    out_cols = ["player_id", "player_name", "position", "n_press", "n_rows", "pos_total", "exec_total", "check_total", "check_per_press"]
    out[[c for c in out_cols if c in out.columns]].to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train hazard models and compute player credits")
    parser.add_argument("--max-rows", type=int, default=None, help="Subsample for smaller eval")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip isotonic calibration; use raw model outputs")
    args = parser.parse_args()

    print("[1/5] Loading data and splitting...")
    df = load_data()
    if args.max_rows:
        seq_ids = df["fc_sequence_id"].unique()
        rng = np.random.default_rng(RANDOM_STATE)
        n_seq = max(1, min(len(seq_ids), int(args.max_rows / (len(df) / len(seq_ids)))))
        keep_seq = rng.choice(seq_ids, size=n_seq, replace=False)
        df = df[df["fc_sequence_id"].isin(keep_seq)].copy()
        print(f"  Subsampled to {len(df):,} rows ({n_seq} sequences)")
    add_slot_imputed_indicators(df)
    train_df, test_df = split_groups(df)
    print(f"  Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")
    numeric_cols, cat_cols = build_feature_lists(df)
    _, train_is_start = _compute_start_meta(train_df)

    # Fit start + hazard models (optionally calibrate on held-out 20%)
    cal_msg = " (no calibration)" if args.no_calibrate else " and calibrating"
    print(f"\n[2/5] Training{cal_msg} start + hazard models...")
    start_pipe, hazard_pipe = fit_hybrid(train_df, numeric_cols, cat_cols, calibrate=not args.no_calibrate)
    start_proba = start_pipe.predict_proba(train_df.loc[_compute_start_meta(train_df)[0].values, numeric_cols + cat_cols])
    print(f"  Start model: P(success) mean={start_proba[:, 1].mean():.4f}")

    # Hazard log loss on test
    print("\n[3/5] Evaluating hazard model...")
    _, test_is_start = _compute_start_meta(test_df)
    test_exec_mask = ~test_is_start
    X_test_exec = test_df.loc[test_exec_mask, numeric_cols + cat_cols]
    y_test_exec = test_df.loc[test_exec_mask, "target_class"]
    proba_exec = hazard_pipe.predict_proba(X_test_exec)
    ll = log_loss(y_test_exec, proba_exec, labels=[0, 1, 2])
    tuning_choice = load_best_model_from_tuning()
    best_model_name = tuning_choice[0] if tuning_choice else "xgboost"
    summary = pd.DataFrame([{
        "model": "hybrid",
        "start_model": "p0_xgb" if load_best_start_model_from_tuning() else "xgboost",
        "hazard_model": best_model_name,
        "hazard_log_loss": ll,
    }])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "model_summary.csv", index=False)
    print(f"  Hazard (exec rows): {best_model_name} log_loss={ll:.4f}")

    print("\n[4/5] Fitting slot predictors...")
    slot_predictors = fit_slot_predictors(
        train_df[numeric_cols + cat_cols], numeric_cols, cat_cols, train_is_start
    )
    print(f"  Fitted: {list(slot_predictors.keys())}, {GHOST_DRAWS} ghost draws")

    print("\n[5/5] Computing player credits (hybrid: start + hazard exec)...")
    X_all = df[numeric_cols + cat_cols]
    player_credit = build_player_press_credit(
        hazard_pipe, start_pipe, df, X_all, slot_predictors,
        save_per_press_path=OUT_DIR / "player_press.parquet",
    )
    _write_clean_csv(player_credit, RESULTS_DIR / "modeling.csv")
    print("\nDone.")
    print(f"  Saved: {RESULTS_DIR / 'model_summary.csv'}, {RESULTS_DIR / 'modeling.csv'}")


if __name__ == "__main__":
    main()
