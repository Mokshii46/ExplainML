"""
model_trainer.py
Trains a fixed set of candidate models and collects structured metrics.
Uses cross-validation for honest evaluation. No randomness in scoring.
"""

import numpy as np
import time
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import make_scorer, f1_score, r2_score, mean_absolute_error
from typing import Tuple
import pandas as pd


CV_FOLDS = 5
RANDOM_STATE = 42


# Interpretability scores — deterministic, hand-designed rubric
INTERPRETABILITY = {
    "Decision Tree": 10,
    "Logistic Regression": 9,
    "Ridge Regression": 9,
    "Random Forest": 6,
    "Gradient Boosting": 4,
    "K-Nearest Neighbors": 5,
}


def get_models(problem_type: str) -> dict:
    if problem_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        }
    else:
        return {
            "Ridge Regression": Ridge(alpha=1.0),
            "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
            "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
        }


def train_and_evaluate(X, y, problem_type: str) -> Tuple[dict, dict]:
    """
    Returns (metrics_per_model, trained_models).
    All metrics are collected via cross-validation — no data leakage.
    """
    models = get_models(problem_type)
    results = {}
    trained_models = {}

    if problem_type == "classification":
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scoring = {
            "accuracy": "accuracy",
            "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        }
    else:
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scoring = {
            "r2": "r2",
            "neg_mae": "neg_mean_absolute_error",
        }

    X_arr = X.values if hasattr(X, "values") else X
    y_arr = y.values if hasattr(y, "values") else y

    for name, model in models.items():
        start = time.time()
        try:
            cv_results = cross_validate(
                model, X_arr, y_arr,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1,
            )
            elapsed = round(time.time() - start, 3)

            if problem_type == "classification":
                test_acc = float(np.mean(cv_results["test_accuracy"]))
                train_acc = float(np.mean(cv_results["train_accuracy"]))
                test_f1 = float(np.mean(cv_results["test_f1_macro"]))
                train_f1 = float(np.mean(cv_results["train_f1_macro"]))
                overfit_score = max(0.0, train_acc - test_acc)

                results[name] = {
                    "model_name": name,
                    "problem_type": "classification",
                    "test_accuracy": round(test_acc, 4),
                    "train_accuracy": round(train_acc, 4),
                    "test_f1_macro": round(test_f1, 4),
                    "train_f1_macro": round(train_f1, 4),
                    "overfit_gap": round(overfit_score, 4),
                    "training_time_sec": elapsed,
                    "interpretability_score": INTERPRETABILITY.get(name, 5),
                    "cv_folds": CV_FOLDS,
                    "status": "success",
                }
            else:
                test_r2 = float(np.mean(cv_results["test_r2"]))
                train_r2 = float(np.mean(cv_results["train_r2"]))
                test_mae = -float(np.mean(cv_results["test_neg_mae"]))
                train_mae = -float(np.mean(cv_results["train_neg_mae"]))
                overfit_score = max(0.0, train_r2 - test_r2)

                results[name] = {
                    "model_name": name,
                    "problem_type": "regression",
                    "test_r2": round(test_r2, 4),
                    "train_r2": round(train_r2, 4),
                    "test_mae": round(test_mae, 4),
                    "train_mae": round(train_mae, 4),
                    "overfit_gap": round(overfit_score, 4),
                    "training_time_sec": elapsed,
                    "interpretability_score": INTERPRETABILITY.get(name, 5),
                    "cv_folds": CV_FOLDS,
                    "status": "success",
                }

            # Train final model on full data for SHAP
            model.fit(X_arr, y_arr)
            trained_models[name] = model

        except Exception as e:
            results[name] = {
                "model_name": name,
                "status": "failed",
                "error": str(e),
            }

    return results, trained_models