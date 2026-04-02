"""
problem_detector.py
Deterministic rule-based engine to detect whether the target column
represents a classification or regression problem.
"""

import pandas as pd
import numpy as np


CLASSIFICATION_UNIQUE_RATIO_THRESHOLD = 0.05   # < 5% unique → likely classification
CLASSIFICATION_MAX_UNIQUE_COUNT = 20            # Absolute cap for classification
REGRESSION_MIN_UNIQUE_COUNT = 15               # Minimum uniques for regression


def detect_problem_type(df: pd.DataFrame, target_col: str) -> dict:
    """
    Rule engine to determine problem type from the target column.

    Rules (in priority order):
    1. If dtype is object/bool/category → CLASSIFICATION
    2. If unique count ≤ 2 → CLASSIFICATION (binary)
    3. If unique count ≤ CLASSIFICATION_MAX_UNIQUE_COUNT AND
       unique_ratio ≤ CLASSIFICATION_UNIQUE_RATIO_THRESHOLD → CLASSIFICATION
    4. Otherwise → REGRESSION

    Returns a dict with problem_type, confidence, unique_count,
    unique_ratio, dtype, and rule_triggered.
    """
    series = df[target_col].dropna()
    n_total = len(series)

    dtype_str = str(series.dtype)
    unique_vals = series.nunique()
    unique_ratio = unique_vals / n_total if n_total > 0 else 0
    sample_vals = series.unique()[:5].tolist()

    # Rule 1: Non-numeric dtype
    if series.dtype == object or series.dtype.name in ("bool", "category"):
        return _result(
            problem_type="classification",
            confidence="high",
            rule_triggered="Rule 1: Non-numeric dtype detected",
            unique_count=unique_vals,
            unique_ratio=round(unique_ratio, 4),
            dtype=dtype_str,
            sample_values=sample_vals,
            class_distribution=_class_distribution(series),
        )

    # Rule 2: Binary target
    if unique_vals <= 2:
        return _result(
            problem_type="classification",
            confidence="high",
            rule_triggered="Rule 2: Binary target (≤2 unique values)",
            unique_count=unique_vals,
            unique_ratio=round(unique_ratio, 4),
            dtype=dtype_str,
            sample_values=sample_vals,
            class_distribution=_class_distribution(series),
        )

    # Rule 3: Low cardinality integer-like column
    if (
        unique_vals <= CLASSIFICATION_MAX_UNIQUE_COUNT
        and unique_ratio <= CLASSIFICATION_UNIQUE_RATIO_THRESHOLD
    ):
        return _result(
            problem_type="classification",
            confidence="medium",
            rule_triggered=(
                f"Rule 3: Low cardinality "
                f"({unique_vals} unique values, {round(unique_ratio*100,1)}% ratio)"
            ),
            unique_count=unique_vals,
            unique_ratio=round(unique_ratio, 4),
            dtype=dtype_str,
            sample_values=sample_vals,
            class_distribution=_class_distribution(series),
        )

    # Rule 4: Default → regression
    return _result(
        problem_type="regression",
        confidence="high",
        rule_triggered=(
            f"Rule 4: High cardinality numeric "
            f"({unique_vals} unique values, {round(unique_ratio*100,1)}% ratio)"
        ),
        unique_count=unique_vals,
        unique_ratio=round(unique_ratio, 4),
        dtype=dtype_str,
        sample_values=sample_vals,
        target_stats=_target_stats(series),
    )


def _result(problem_type, confidence, rule_triggered, **kwargs) -> dict:
    base = {
        "problem_type": problem_type,
        "confidence": confidence,
        "rule_triggered": rule_triggered,
    }
    base.update(kwargs)
    return base


def _class_distribution(series: pd.Series) -> dict:
    counts = series.value_counts(normalize=True).round(4).to_dict()
    return {str(k): v for k, v in counts.items()}


def _target_stats(series: pd.Series) -> dict:
    return {
        "mean": round(float(series.mean()), 4),
        "std": round(float(series.std()), 4),
        "min": round(float(series.min()), 4),
        "max": round(float(series.max()), 4),
        "skew": round(float(series.skew()), 4),
    }