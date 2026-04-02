"""
explainability.py
Generates traceable explanations using SHAP and feature importance.
SHAP is deterministic given a fixed model and data — no LLMs.
"""

import numpy as np
import pandas as pd
from typing import Optional


def generate_explanations(
    winner_model,
    X: pd.DataFrame,
    feature_names: list,
    problem_type: str,
    winner_name: str,
) -> dict:
    """
    Returns a dict with:
    - shap_values: per-feature SHAP importance (mean absolute)
    - feature_importance: model's native feature importances (if available)
    - explanation_method: which method was used and why
    """
    explanations = {
        "model_name": winner_name,
        "shap_available": False,
        "feature_importance_available": False,
        "shap_values": [],
        "feature_importance": [],
        "explanation_method": "",
        "top_features": [],
    }

    X_arr = X.values if hasattr(X, "values") else X
    # Use a sample for speed (SHAP can be slow on large datasets)
    sample_size = min(200, X_arr.shape[0])
    X_sample = X_arr[:sample_size]

    # ── Try SHAP ──────────────────────────────────────────────────────────────
    try:
        import shap

        # Choose explainer based on model type
        model_type = type(winner_model).__name__
        if "Forest" in model_type or "Boosting" in model_type or "Tree" in model_type:
            explainer = shap.TreeExplainer(winner_model)
            shap_vals = explainer.shap_values(X_sample)
        else:
            explainer = shap.LinearExplainer(winner_model, X_sample)
            shap_vals = explainer.shap_values(X_sample)

        # For multi-class classification, sum across classes
        if isinstance(shap_vals, list):
            shap_arr = np.abs(np.array(shap_vals)).mean(axis=0)
        else:
            shap_arr = np.abs(shap_vals)

        mean_shap = np.mean(shap_arr, axis=0)
        shap_ranked = sorted(
            zip(feature_names, mean_shap.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        explanations["shap_values"] = [
            {
                "feature": feat,
                "mean_abs_shap": round(float(val), 6),
                "rank": i + 1,
            }
            for i, (feat, val) in enumerate(shap_ranked)
        ]
        explanations["shap_available"] = True
        explanations["explanation_method"] = "SHAP TreeExplainer / LinearExplainer (deterministic given fixed model weights)"

    except Exception as e:
        explanations["shap_error"] = str(e)

    # ── Try native feature importance ─────────────────────────────────────────
    if hasattr(winner_model, "feature_importances_"):
        importances = winner_model.feature_importances_
        fi_ranked = sorted(
            zip(feature_names, importances.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        explanations["feature_importance"] = [
            {
                "feature": feat,
                "importance": round(float(val), 6),
                "rank": i + 1,
            }
            for i, (feat, val) in enumerate(fi_ranked)
        ]
        explanations["feature_importance_available"] = True
        if not explanations["explanation_method"]:
            explanations["explanation_method"] = "Gini-based feature importance (built into tree models)"

    elif hasattr(winner_model, "coef_"):
        coef = winner_model.coef_
        if coef.ndim > 1:
            coef_vals = np.abs(coef).mean(axis=0)
        else:
            coef_vals = np.abs(coef)
        fi_ranked = sorted(
            zip(feature_names, coef_vals.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        explanations["feature_importance"] = [
            {
                "feature": feat,
                "importance": round(float(val), 6),
                "rank": i + 1,
            }
            for i, (feat, val) in enumerate(fi_ranked)
        ]
        explanations["feature_importance_available"] = True
        if not explanations["explanation_method"]:
            explanations["explanation_method"] = "Coefficient magnitude from linear model"

    # ── Build top features summary ────────────────────────────────────────────
    source = explanations["shap_values"] if explanations["shap_available"] else explanations["feature_importance"]
    explanations["top_features"] = source[:5]

    return explanations