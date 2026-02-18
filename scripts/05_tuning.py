#!/usr/bin/env python3
"""Gradient-boosting hyperparameter tuning for forecheck hazard prediction.

Tunes HistGradientBoosting, GradientBoosting, and XGBoost via RandomizedSearchCV
with group-based cross-validation (by fc_sequence_id).

Target: 3-class classification (ongoing=0, success=1, failure=2); metric = log loss.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, log_loss
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hazard_features.parquet"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

RANDOM_STATE = 7
N_CV_FOLDS = 5
N_ITER_RANDOM = 50  # RandomizedSearch iterations per model
N_JOBS = -1


class TimeAugmenter:
    """Add log and sqrt time basis features."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TimeAugmenter":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        t = X["time_since_start_s"].astype(float).clip(lower=0.0)
        X["log_time_since_start_s"] = np.log1p(t)
        X["sqrt_time_since_start_s"] = np.sqrt(t)
        return X

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        if input_features is None:
            return np.array([])
        return np.asarray(list(input_features) + ["log_time_since_start_s", "sqrt_time_since_start_s"])


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


def build_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Partition columns into numeric and categorical."""
    ignore = {
        "fc_sequence_id", "sl_event_id", "event_t", "terminal_failure_t",
        "target_class", "carrier_id", "F1_id", "F2_id", "F3_id", "F4_id", "F5_id",
    }
    feature_cols = [c for c in df.columns if c not in ignore]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in numeric_cols]
    return numeric_cols, cat_cols


def build_preprocessor(numeric_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Build preprocessing: impute, spline time, scale, one-hot."""
    time_spline_cols = ["time_since_start_s"]
    numeric_main_cols = [c for c in numeric_cols if c not in time_spline_cols]

    numeric_main = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    time_spline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("spline", SplineTransformer(n_knots=5, degree=3, include_bias=False)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_main, numeric_main_cols),
        ("time_spline", time_spline, time_spline_cols),
        ("cat", categorical, cat_cols),
    ])


def get_model_configs(preprocessor: ColumnTransformer) -> list[tuple[str, Pipeline, dict]]:
    """Return list of (name, estimator, param_distributions) for tuning."""
    time_aug = TimeAugmenter()
    configs: list[tuple[str, Pipeline, dict]] = []

    # HistGradientBoosting (histogram-based GBM)
    configs.append((
        "hist_gbm",
        Pipeline([
            ("time_aug", time_aug),
            ("prep", preprocessor),
            ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
        {
            "model__max_depth": [4, 6, 8, 10, 12],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_iter": [200, 300, 500, 800],
            "model__min_samples_leaf": [10, 20, 50],
            "model__l2_regularization": [0.0, 0.1, 1.0],
            "model__max_bins": [128, 255],
        },
    ))

    # GradientBoosting (traditional sklearn GBM)
    configs.append((
        "gbm",
        Pipeline([
            ("time_aug", time_aug),
            ("prep", preprocessor),
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
        {
            "model__max_depth": [4, 6, 8, 10],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__n_estimators": [100, 200, 300, 500],
            "model__min_samples_leaf": [5, 10, 20],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__max_features": ["sqrt", "log2", None],
        },
    ))

    # XGBoost
    configs.append((
        "xgboost",
        Pipeline([
            ("time_aug", time_aug),
            ("prep", preprocessor),
            ("model", xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        {
            "model__max_depth": [4, 6, 8, 10],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__n_estimators": [200, 300, 500, 800],
            "model__min_child_weight": [1, 3, 5],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
            "model__reg_alpha": [0.0, 0.1, 1.0],
            "model__reg_lambda": [0.1, 1.0, 10.0],
        },
    ))

    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradient-boosting hyperparameter tuning for forecheck hazard prediction")
    parser.add_argument("--quick", action="store_true", help="Use 10 iterations per model for fast testing")
    args = parser.parse_args()

    n_iter = 10 if args.quick else N_ITER_RANDOM
    if args.quick:
        print("Quick mode: 10 iterations per model")

    print("Loading data...")
    df = load_data()
    train_df, test_df = split_groups(df)

    numeric_cols, cat_cols = build_feature_lists(df)
    preprocessor = build_preprocessor(numeric_cols, cat_cols)

    X_train = train_df[numeric_cols + cat_cols]
    y_train = train_df["target_class"]
    X_test = test_df[numeric_cols + cat_cols]
    y_test = test_df["target_class"]
    groups = train_df["fc_sequence_id"].values

    configs = get_model_configs(preprocessor)
    cv = GroupKFold(n_splits=N_CV_FOLDS)
    all_results: list[dict] = []
    best_loss = np.inf
    best_name = ""

    for name, estimator, param_dist in configs:
        print(f"\n{'='*60}")
        print(f"Tuning {name} (RandomizedSearchCV, n_iter={n_iter})")
        print("=" * 60)

        search = RandomizedSearchCV(
            estimator,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring="neg_log_loss",
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
            refit=True,
            verbose=1,
        )

        search.fit(X_train, y_train, groups=groups)

        cv_ll = -search.best_score_
        best_params = search.best_params_
        proba = search.best_estimator_.predict_proba(X_test)
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
    out_path = RESULTS_DIR / "tuning_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
