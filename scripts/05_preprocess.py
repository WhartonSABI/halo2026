#!/usr/bin/env python3
"""Hazard-model preprocessing and featurizing (pipeline step 5).

Used by 06_tuning.py and 07_modeling.py. Encodes:
- Time basis transforms (log, sqrt of elapsed time)
- Column partitioning (numeric vs categorical, slot vs other)
- Slot imputation with missingness indicators
- Spline basis for time_since_start_s
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

FORECHECK_SLOTS = ["F1", "F2", "F3", "F4", "F5"]
SLOT_FEATURE_TEMPLATE = [
    "{slot}_r",
    "{slot}_vr_carrier",
    "{slot}_sinθ",
    "{slot}_cosθ",
    "{slot}_block_severity",
    "{slot}_block_center_severity",
    "{slot}_r_nearestOpp",
    "{slot}_vr_nearestOpp",
]

IGNORE_COLS = frozenset({
    "fc_sequence_id", "sl_event_id", "event_t", "terminal_failure_t",
    "target_class", "carrier_id",
    "F1_id", "F2_id", "F3_id", "F4_id", "F5_id",
})


class TimeAugmenter:
    """Add log and sqrt time basis features for hazard models."""

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


def add_slot_imputed_indicators(df: pd.DataFrame) -> None:
    """Add F1_imputed, F2_imputed, ... columns (1 where that slot has any NaN). In-place."""
    for slot in FORECHECK_SLOTS:
        slot_cols = [t.format(slot=slot) for t in SLOT_FEATURE_TEMPLATE]
        existing = [c for c in slot_cols if c in df.columns]
        if not existing:
            continue
        col = f"{slot}_imputed"
        df[col] = (df[existing].isna().any(axis=1)).astype(np.float64)


def build_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Partition hazard-feature columns into numeric and categorical."""
    feature_cols = [c for c in df.columns if c not in IGNORE_COLS]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in numeric_cols]
    return numeric_cols, cat_cols


def build_preprocessor(numeric_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Build preprocessing pipeline: impute, spline time, scale, one-hot.

    Slot features get mean imputation + missingness indicators (treat missing as "unknown").
    Other numerics keep median imputation.
    """
    time_spline_cols = ["time_since_start_s"]
    slot_numeric = [
        c
        for s in FORECHECK_SLOTS
        for c in [t.format(slot=s) for t in SLOT_FEATURE_TEMPLATE]
        if c in numeric_cols
    ]
    other_numeric = [
        c
        for c in numeric_cols
        if c not in slot_numeric and c != "time_since_start_s"
    ]

    slot_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    other_num_pipe = Pipeline([
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

    transformers = []
    if slot_numeric:
        transformers.append(("slot_num", slot_pipe, slot_numeric))
    if other_numeric:
        transformers.append(("num", other_num_pipe, other_numeric))
    transformers.extend([
        ("time_spline", time_spline, time_spline_cols),
        ("cat", categorical, cat_cols),
    ])
    return ColumnTransformer(transformers)


def build_model_prep_pipeline(
    numeric_cols: list[str], cat_cols: list[str]
) -> Pipeline:
    """Build full preprocessing pipeline: TimeAugmenter + ColumnTransformer."""
    return Pipeline([
        ("time_aug", TimeAugmenter()),
        ("prep", build_preprocessor(numeric_cols, cat_cols)),
    ])


def run_preprocess() -> None:
    """Produce model_features.parquet and model_preprocessor.joblib. Run once before 06/07."""
    from pathlib import Path
    from joblib import dump

    project_root = Path(__file__).resolve().parent.parent
    in_path = project_root / "data" / "processed" / "hazard_features.parquet"
    out_path = project_root / "data" / "processed" / "model_features.parquet"
    preproc_path = project_root / "data" / "processed" / "model_preprocessor.joblib"

    if not in_path.exists():
        raise FileNotFoundError(f"Run 03_features.py first: {in_path} not found")

    df = pd.read_parquet(in_path)
    add_slot_imputed_indicators(df)

    numeric_cols, cat_cols = build_feature_lists(df)
    pipeline = build_model_prep_pipeline(numeric_cols, cat_cols)

    X_raw = df[numeric_cols + cat_cols]
    X_processed = pipeline.fit_transform(X_raw)

    feature_names = pipeline.named_steps["prep"].get_feature_names_out()
    out_df = pd.DataFrame(X_processed, columns=list(feature_names))
    for col in ["fc_sequence_id", "sl_event_id", "event_t", "terminal_failure_t"]:
        out_df[col] = df[col].values

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    dump(pipeline, preproc_path)

    print(f"Saved {out_path} ({len(out_df):,} rows)")
    print(f"Saved {preproc_path}")


if __name__ == "__main__":
    run_preprocess()
