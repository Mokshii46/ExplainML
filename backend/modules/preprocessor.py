"""
preprocessor.py
Deterministic preprocessing engine.
Rules are applied in fixed order and every decision is logged.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple


# Thresholds (deterministic rules)
HIGH_MISSING_THRESHOLD = 0.50   # Drop column if > 50% missing
CAT_CARDINALITY_LIMIT = 20      # Drop categorical col if > 20 unique vals (too sparse)


def preprocess(
    df: pd.DataFrame, target_col: str, problem_type: str
) -> Tuple[pd.DataFrame, pd.Series, dict]:

    log = []
    df = df.copy()

    # ── Step 1: Separate target ─────────────────────────────
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    y_raw = df[target_col].copy()
    X = df.drop(columns=[target_col])

    log.append({
        "step": "Target Separation",
        "action": f"Removed '{target_col}' from feature matrix.",
        "columns_before": list(df.columns),
        "columns_after": list(X.columns),
    })

    # ── Step 2: Drop high-missing columns ───────────────────
    missing_ratio = X.isnull().mean()
    high_missing_cols = missing_ratio[missing_ratio > HIGH_MISSING_THRESHOLD].index.tolist()

    if high_missing_cols:
        X.drop(columns=high_missing_cols, inplace=True)
        log.append({
            "step": "Drop High-Missing Columns",
            "dropped": high_missing_cols,
        })

    # ── Step 3: Identify column types ───────────────────────
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()

    # ── Step 4: Drop high-cardinality categoricals ──────────
    high_card = [c for c in categorical_cols if X[c].nunique() > CAT_CARDINALITY_LIMIT]

    if high_card:
        X.drop(columns=high_card, inplace=True)
        categorical_cols = [c for c in categorical_cols if c not in high_card]
        log.append({
            "step": "Drop High-Cardinality Categoricals",
            "dropped": high_card,
        })

    # ── Step 5: Impute numeric columns ──────────────────────
    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

    # ── Step 6: Encode categorical columns ──────────────────
    if categorical_cols:
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # ── Step 7: Encode target (FIXED) ───────────────────────
    if problem_type == "classification":
        target_encoder = LabelEncoder()
        y = pd.Series(
            target_encoder.fit_transform(y_raw.astype(str)),
            name=target_col
        )

        log.append({
            "step": "Target Encoding",
            "mapping": dict(zip(
                target_encoder.classes_,
                target_encoder.transform(target_encoder.classes_)
            )),
        })

    else:
        y = y_raw.astype(float)

    # ── Step 8: Scale numeric features ──────────────────────
    if numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    log.append({
        "step": "Final Feature Matrix",
        "shape": list(X.shape),
        "features": list(X.columns),
        "target_dtype": str(y.dtype),
    })

    return X, y, {"steps": log, "feature_names": list(X.columns)}