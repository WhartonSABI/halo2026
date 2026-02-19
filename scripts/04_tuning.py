#!/usr/bin/env python3
"""Hyperparameter tuning for forecheck hazard prediction.

Tunes RandomForest, HistGradientBoosting, and XGBoost via ParameterSampler
with group-based cross-validation (by fc_sequence_id).
Preprocessor fit on train only (no leakage).

Target: 3-class classification (ongoing=0, success=1, failure=2); metric = log loss.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
)

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, log_loss
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    ParameterSampler,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from tqdm import tqdm

import xgboost as xgb

from _preprocess import (
    RANDOM_STATE,
    TimeAugmenter,
    add_slot_imputed_indicators,
    build_feature_lists,
    build_preprocessor,
    compute_start_meta,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hazard_features.parquet"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
N_CV_FOLDS = 5
N_ITER_RANDOM = 50
N_JOBS = max(1, (os.cpu_count() or 1) - 1)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Load hazard features, split train/test, return raw data and feature cols.
    No preprocessing here; preprocessor is fit per CV fold inside the pipeline.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Run 02_features.py first to create {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    y = np.zeros(len(df), dtype=np.int64)
    y[df["event_t"].eq(1).values] = 1
    y[df["terminal_failure_t"].eq(1).values] = 2
    df["target_class"] = y
    add_slot_imputed_indicators(df)

    numeric_cols, cat_cols = build_feature_lists(df)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(splitter.split(df, groups=df["fc_sequence_id"]))
    train_df = df.iloc[tr_idx]
    test_df = df.iloc[te_idx]

    X_raw_train = train_df[numeric_cols + cat_cols]
    X_raw_test = test_df[numeric_cols + cat_cols]
    y_train = train_df["target_class"].values
    y_test = test_df["target_class"].values
    groups = train_df["fc_sequence_id"].values

    return X_raw_train, X_raw_test, y_train, y_test, groups, numeric_cols, cat_cols


def load_data_start() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Load start-row data for p₀ model: binary sequence success given start config."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Run 02_features.py first to create {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    add_slot_imputed_indicators(df)
    numeric_cols, cat_cols = build_feature_lists(df)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(splitter.split(df, groups=df["fc_sequence_id"]))
    train_df = df.iloc[tr_idx]
    test_df = df.iloc[te_idx]

    _, train_is_start = compute_start_meta(train_df)
    _, test_is_start = compute_start_meta(test_df)

    X_train = train_df.loc[train_is_start, numeric_cols + cat_cols]
    X_test = test_df.loc[test_is_start, numeric_cols + cat_cols]
    seq_outcome_train = train_df.groupby("fc_sequence_id")["event_t"].max()
    seq_outcome_test = test_df.groupby("fc_sequence_id")["event_t"].max()
    start_seq_train = train_df.loc[train_is_start, "fc_sequence_id"].values
    start_seq_test = test_df.loc[test_is_start, "fc_sequence_id"].values
    y_train = (seq_outcome_train.reindex(start_seq_train).values == 1).astype(np.int64)
    y_test = (seq_outcome_test.reindex(start_seq_test).values == 1).astype(np.int64)
    groups = start_seq_train
    return X_train, X_test, y_train, y_test, groups, numeric_cols, cat_cols


def get_p0_config() -> tuple[str, object, dict]:
    """Return (name, estimator, param_distributions) for start model (p₀)."""
    return (
        "p0_xgb",
        xgb.XGBClassifier(
            objective="binary:logistic",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        {
            "max_depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "n_estimators": [200, 300, 500, 800],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [0.1, 1.0, 10.0],
        },
    )


def get_model_configs() -> list[tuple[str, object, dict]]:
    """Return list of (name, estimator, param_distributions) for hazard models."""
    configs: list[tuple[str, object, dict]] = []

    configs.append((
        "rf",
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [4, 6, 8, 10, 12, None],
            "min_samples_leaf": [5, 10, 20],
            "max_features": ["sqrt", "log2", None],
        },
    ))
    configs.append((
        "hist_gbm",
        HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "max_depth": [4, 6, 8, 10, 12],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_iter": [200, 300, 500, 800],
            "min_samples_leaf": [10, 20, 50],
            "l2_regularization": [0.0, 0.1, 1.0],
            "max_bins": [128, 255],
        },
    ))
    configs.append((
        "xgboost",
        xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        {
            "max_depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "n_estimators": [200, 300, 500, 800],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [0.1, 1.0, 10.0],
        },
    ))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning (RF, HistGBM, XGBoost) for forecheck hazard prediction"
    )
    parser.add_argument("--full", action="store_true", help="Use 50 iterations per model (default: 10)")
    args = parser.parse_args()

    n_iter = N_ITER_RANDOM if args.full else 10
    if not args.full:
        print("Quick mode: 10 iterations per model (use --full for 50)")

    cv = GroupKFold(n_splits=N_CV_FOLDS)
    all_results: list[dict] = []
    best_loss = np.inf
    best_name = ""

    # --- p₀ (start model) tuning first ---
    print("\n" + "=" * 60)
    print("Tuning p₀ (start model, binary sequence success)")
    print("=" * 60)
    print("Loading start-row data...")
    X_start_train, X_start_test, y_start_train, y_start_test, start_groups, numeric_cols, cat_cols = load_data_start()
    preprocessor = build_preprocessor(numeric_cols, cat_cols)
    name, estimator, param_dist = get_p0_config()
    base_pipeline = Pipeline([
        ("time_aug", TimeAugmenter()),
        ("prep", clone(preprocessor)),
        ("model", estimator),
    ])
    param_list = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=RANDOM_STATE))
    best_cv_score = -np.inf
    best_params: dict | None = None
    for params in tqdm(param_list, desc="p0_xgb", unit="candidate", leave=False):
        prefixed = {f"model__{k}": v for k, v in params.items()}
        pipe = clone(base_pipeline)
        pipe.set_params(**prefixed)
        scores = cross_val_score(
            pipe, X_start_train, y_start_train,
            cv=cv, groups=start_groups,
            scoring="neg_log_loss", n_jobs=N_JOBS,
        )
        if scores.mean() > best_cv_score:
            best_cv_score = scores.mean()
            best_params = params
    cv_ll = -best_cv_score
    prefixed = {f"model__{k}": v for k, v in (best_params or {}).items()}
    best_pipeline = clone(base_pipeline)
    best_pipeline.set_params(**prefixed)
    best_pipeline.fit(X_start_train, y_start_train)
    proba = best_pipeline.predict_proba(X_start_test)
    test_ll = log_loss(y_start_test, proba, labels=[0, 1])
    pred = np.argmax(proba, axis=1)
    print(f"\np0_xgb best CV log_loss: {cv_ll:.5f}")
    print(f"p0_xgb test log_loss: {test_ll:.5f}")
    print("Best params:", best_params)
    print(classification_report(y_start_test, pred, labels=[0, 1], target_names=["failure", "success"]))
    all_results.append({"model": "p0_xgb", "cv_log_loss": cv_ll, "test_log_loss": test_ll, "best_params": str(best_params)})

    # --- Hazard models (3-class) ---
    print("\n" + "=" * 60)
    print("Tuning hazard models (3-class)")
    print("=" * 60)
    print("Loading hazard data...")
    X_raw_train, X_raw_test, y_train, y_test, groups, numeric_cols, cat_cols = load_data()
    preprocessor = build_preprocessor(numeric_cols, cat_cols)
    configs = get_model_configs()

    for name, estimator, param_dist in configs:
        print(f"\n{'='*60}")
        print(f"Tuning {name} ({n_iter} candidates × {N_CV_FOLDS} folds)")
        print("=" * 60)

        base_pipeline = Pipeline([
            ("time_aug", TimeAugmenter()),
            ("prep", clone(preprocessor)),
            ("model", estimator),
        ])
        param_list = list(
            ParameterSampler(
                param_dist, n_iter=n_iter, random_state=RANDOM_STATE
            )
        )
        best_cv_score = -np.inf
        best_params: dict | None = None

        for params in tqdm(param_list, desc=name, unit="candidate", leave=False):
            prefixed = {f"model__{k}": v for k, v in params.items()}
            pipe = clone(base_pipeline)
            pipe.set_params(**prefixed)
            scores = cross_val_score(
                pipe,
                X_raw_train,
                y_train,
                cv=cv,
                groups=groups,
                scoring="neg_log_loss",
                n_jobs=N_JOBS,
            )
            mean_score = scores.mean()
            if mean_score > best_cv_score:
                best_cv_score = mean_score
                best_params = params

        cv_ll = -best_cv_score
        prefixed = {f"model__{k}": v for k, v in best_params.items()}
        best_pipeline = clone(base_pipeline)
        best_pipeline.set_params(**prefixed)
        best_pipeline.fit(X_raw_train, y_train)

        proba = best_pipeline.predict_proba(X_raw_test)
        test_ll = log_loss(y_test, proba, labels=[0, 1, 2])
        pred = np.argmax(proba, axis=1)

        print(f"\n{name} best CV log_loss: {cv_ll:.5f}")
        print(f"{name} test log_loss: {test_ll:.5f}")
        print("Best params:", best_params)
        print(classification_report(
            y_test, pred, labels=[0, 1, 2], target_names=["ongoing", "success", "failure"]
        ))

        all_results.append({
            "model": name,
            "cv_log_loss": cv_ll,
            "test_log_loss": test_ll,
            "best_params": str(best_params),
        })

        if test_ll < best_loss:
            best_loss = test_ll
            best_name = name

    results_df = pd.DataFrame(all_results)
    # Save with p0 first, then hazard models by test_log_loss
    p0_row = results_df[results_df["model"].eq("p0_xgb")]
    hazard_rows = results_df[~results_df["model"].eq("p0_xgb")].sort_values("test_log_loss")
    results_df = pd.concat([p0_row, hazard_rows], ignore_index=True)

    hazard_df = results_df[~results_df["model"].eq("p0_xgb")]
    hazard_df = hazard_df.sort_values("test_log_loss").reset_index(drop=True)
    print("\n" + "=" * 60)
    print("FINAL RANKING – hazard models (by test log_loss)")
    print("=" * 60)
    print(hazard_df[["model", "cv_log_loss", "test_log_loss"]].to_string(index=False))
    if len(hazard_df) > 0:
        best_name = hazard_df.iloc[0]["model"]
        best_loss = hazard_df.iloc[0]["test_log_loss"]
        print(f"\nBest hazard: {best_name} (test log_loss={best_loss:.5f})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_filename = "tuning_full.csv" if args.full else "tuning_quick.csv"
    out_path = RESULTS_DIR / out_filename
    results_df.to_csv(out_path, index=False)
    print(f"\np₀ (start) params saved in {out_path} (model=p0_xgb)")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
