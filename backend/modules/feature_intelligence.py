"""
feature_intelligence.py
Deterministic feature analysis layer.
- Correlation-based pruning (removes redundant features)
- Mutual information ranking (deterministic scoring)
- Low-variance filtering
Every decision is rule-based and fully explainable.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from typing import Tuple


CORRELATION_THRESHOLD = 0.90    # Drop one of two features correlated > 90%
VARIANCE_THRESHOLD = 0.001      # Drop near-zero variance features


def analyze_features(
    X: pd.DataFrame, y: pd.Series, problem_type: str
) -> Tuple[pd.DataFrame, dict]:
    """
    Returns (X_selected, feature_report).
    feature_report contains pruning decisions and importance scores.
    """
    log = []
    X = X.copy()
    original_features = list(X.columns)

    # ── Step 1: Low variance filter ──────────────────────────────────────────
    variances = X.var()
    low_var_cols = variances[variances < VARIANCE_THRESHOLD].index.tolist()
    if low_var_cols:
        X.drop(columns=low_var_cols, inplace=True)
        log.append({
            "step": "Low Variance Filter",
            "rule": f"Drop features with variance < {VARIANCE_THRESHOLD}",
            "dropped_features": low_var_cols,
            "reason": "Near-constant features carry no predictive signal.",
        })

    # ── Step 2: Correlation pruning ───────────────────────────────────────────
    dropped_corr = []
    corr_pairs = []
    if X.shape[1] > 1:
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            col for col in upper_tri.columns
            if any(upper_tri[col] > CORRELATION_THRESHOLD)
        ]
        # Log the pairs
        for col in to_drop:
            correlated_with = upper_tri.index[upper_tri[col] > CORRELATION_THRESHOLD].tolist()
            corr_pairs.append({
                "dropped": col,
                "correlated_with": correlated_with,
                "correlation": round(float(upper_tri[col][upper_tri[col] > CORRELATION_THRESHOLD].max()), 4),
            })
        if to_drop:
            X.drop(columns=to_drop, inplace=True)
            dropped_corr = to_drop
            log.append({
                "step": "Correlation Pruning",
                "rule": f"Drop one feature from pairs with |correlation| > {CORRELATION_THRESHOLD}",
                "dropped_features": dropped_corr,
                "correlated_pairs": corr_pairs,
                "reason": "Redundant features inflate dimensionality without adding information.",
            })

    # ── Step 3: Mutual information scoring ────────────────────────────────────
    if X.shape[1] > 0:
        if problem_type == "classification":
            mi_scores = mutual_info_classif(X, y, random_state=42)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=42)

        mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        importance_ranking = [
            {
                "feature": feat,
                "mutual_info_score": round(float(score), 5),
                "rank": i + 1,
                "signal_level": _signal_level(score, mi_series.max()),
            }
            for i, (feat, score) in enumerate(mi_series.items())
        ]
        log.append({
            "step": "Mutual Information Ranking",
            "rule": "Score each feature by mutual information with the target.",
            "rule_rationale": "Mutual information is model-agnostic and captures non-linear relationships.",
            "importance_ranking": importance_ranking,
        })
    else:
        importance_ranking = []

    report = {
        "original_feature_count": len(original_features),
        "selected_feature_count": X.shape[1],
        "selected_features": list(X.columns),
        "removed_features": {
            "low_variance": low_var_cols,
            "redundant_correlations": dropped_corr,
        },
        "analysis_steps": log,
        "importance_ranking": importance_ranking,
    }

    return X, report


def _signal_level(score: float, max_score: float) -> str:
    if max_score == 0:
        return "none"
    ratio = score / max_score
    if ratio >= 0.66:
        return "high"
    elif ratio >= 0.33:
        return "medium"
    elif ratio > 0:
        return "low"
    return "none"