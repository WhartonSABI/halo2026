#!/usr/bin/env python3
"""Hyperparameter tuning for forecheck hazard prediction.

Tunes RandomForest, HistGradientBoosting, and XGBoost via ParameterSampler
with group-based cross-validation (by fc_sequence_id).

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
from tqdm import tqdm

import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_features.parquet"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

META_COLS = {"fc_sequence_id", "sl_event_id", "event_t", "terminal_failure_t", "target_class"}

RANDOM_STATE = 7
N_CV_FOLDS = 5
N_ITER_RANDOM = 50  # RandomizedSearch iterations per model
N_JOBS = max(1, (os.cpu_count() or 1) - 1)  # all cores minus one


def load_data() -> pd.DataFrame:
    """Load preprocessed model features and define 3-class target."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Run 05_preprocess.py first to create {DATA_PATH}"
        )
    df = pd.read_parquet(DATA_PATH)
    y = np.zeros(len(df), dtype=np.int64)
    y[df["event_t"].eq(1).values] = 1
    y[df["terminal_failure_t"].eq(1).values] = 2
    df["target_class"] = y
    return df


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Column names for model input (excludes metadata)."""
    return [c for c in df.columns if c not in META_COLS]


def split_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test split by fc_sequence_id."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(splitter.split(df, groups=df["fc_sequence_id"]))
    return df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()


def get_model_configs() -> list[tuple[str, object, dict]]:
    """Return list of (name, estimator, param_distributions) for tuning.
    Estimators expect preprocessed X from model_features.parquet.
    """
    configs: list[tuple[str, object, dict]] = []

    # RandomForest
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

    # HistGradientBoosting
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

    # XGBoost
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
    parser = argparse.ArgumentParser(description="Hyperparameter tuning (RF, HistGBM, XGBoost) for forecheck hazard prediction")
    parser.add_argument("--full", action="store_true", help="Use 50 iterations per model (default: 10)")
    args = parser.parse_args()

    n_iter = N_ITER_RANDOM if args.full else 10
    if not args.full:
        print("Quick mode: 10 iterations per model (use --full for 50)")

    print("Loading data...")
    df = load_data()
    train_df, test_df = split_groups(df)

    feat_cols = _feature_columns(df)
    X_train = train_df[feat_cols]
    y_train = train_df["target_class"]
    X_test = test_df[feat_cols]
    y_test = test_df["target_class"]
    groups = train_df["fc_sequence_id"].values

    configs = get_model_configs()
    cv = GroupKFold(n_splits=N_CV_FOLDS)
    all_results: list[dict] = []
    best_loss = np.inf
    best_name = ""

    for name, estimator, param_dist in configs:
        print(f"\n{'='*60}")
        print(f"Tuning {name} ({n_iter} candidates × {N_CV_FOLDS} folds)")
        print("=" * 60)

        param_list = list(
            ParameterSampler(
                param_dist, n_iter=n_iter, random_state=RANDOM_STATE
            )
        )
        best_cv_score = -np.inf
        best_params: dict | None = None

        for params in tqdm(param_list, desc=name, unit="candidate", leave=False):
            est = clone(estimator)
            est.set_params(**params)
            scores = cross_val_score(
                est,
                X_train,
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
        best_estimator = clone(estimator)
        best_estimator.set_params(**best_params)
        best_estimator.fit(X_train, y_train)

        proba = best_estimator.predict_proba(X_test)
        test_ll = log_loss(y_test, proba, labels=[0, 1, 2])
        pred = np.argmax(proba, axis=1)

        print(f"\n{name} best CV log_loss: {cv_ll:.5f}")
        print(f"{name} test log_loss: {test_ll:.5f}")
        print("Best params:", best_params)
        print(classification_report(y_test, pred, labels=[0, 1, 2], target_names=["ongoing", "success", "failure"]))

        all_results.append({
            "model": name,
            "cv_log_loss": cv_ll,
            "test_log_loss": test_ll,
            "best_params": str(best_params),
        })

        if test_ll < best_loss:
            best_loss = test_ll
            best_name = name

    results_df = pd.DataFrame(all_results).sort_values("test_log_loss").reset_index(drop=True)

    print("\n" + "=" * 60)
    print("FINAL RANKING (by test log_loss)")
    print("=" * 60)
    print(results_df[["model", "cv_log_loss", "test_log_loss"]].to_string(index=False))
    print(f"\nBest: {best_name} (test log_loss={best_loss:.5f})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_filename = "tuning_full.csv" if args.full else "tuning_quick.csv"
    out_path = RESULTS_DIR / out_filename
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
