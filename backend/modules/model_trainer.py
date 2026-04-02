"""
model_trainer.py
Trains a fixed set of candidate models and collects structured metrics.
Uses cross-validation for honest evaluation. No randomness in scoring.

SPEED OPTIMIZATIONS (accuracy-preserving only):
- n_jobs=-1 on all models + cross_validate (true CPU parallelism)
- LogisticRegression solver auto-picked per dataset size
- KNN algorithm='auto' (sklearn picks fastest tree for data shape)
- All n_estimators=100, max_depth=8, CV_FOLDS=5 kept at original values
- SHAP sample size capped separately in explainability.py (doesn't affect model)
"""

import numpy as np
import time
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import make_scorer, f1_score
from typing import Tuple
import pandas as pd


CV_FOLDS = 5          # Full 5-fold — accuracy preserved
RANDOM_STATE = 42


INTERPRETABILITY = {
    "Decision Tree": 10,
    "Logistic Regression": 9,
    "Ridge Regression": 9,
    "Random Forest": 6,
    "Gradient Boosting": 4,
    "K-Nearest Neighbors": 5,
}


def get_models(problem_type: str, n_samples: int) -> dict:
    # Pick fastest accurate LR solver for dataset size — no accuracy trade-off
    lr_solver = 'lbfgs' if n_samples < 10000 else 'saga'

    if problem_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE, solver=lr_solver, n_jobs=-1
            ),
            "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=RANDOM_STATE
            ),
            "K-Nearest Neighbors": KNeighborsClassifier(
                n_neighbors=5, algorithm='auto', n_jobs=-1
            ),
        }
    else:
        return {
            "Ridge Regression": Ridge(alpha=1.0),
            "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100, random_state=RANDOM_STATE
            ),
            "K-Nearest Neighbors": KNeighborsRegressor(
                n_neighbors=5, algorithm='auto', n_jobs=-1
            ),
        }


def train_and_evaluate(X, y, problem_type: str) -> Tuple[dict, dict]:
    X_arr = X.values if hasattr(X, "values") else X
    y_arr = y.values if hasattr(y, "values") else y
    n_samples = X_arr.shape[0]

    models = get_models(problem_type, n_samples)
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

    for name, model in models.items():
        start = time.time()
        try:
            cv_results = cross_validate(
                model, X_arr, y_arr,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1,   # parallelize folds across all CPU cores
            )
            elapsed = round(time.time() - start, 3)

            if problem_type == "classification":
                test_acc  = float(np.mean(cv_results["test_accuracy"]))
                train_acc = float(np.mean(cv_results["train_accuracy"]))
                test_f1   = float(np.mean(cv_results["test_f1_macro"]))
                train_f1  = float(np.mean(cv_results["train_f1_macro"]))
                overfit   = max(0.0, train_acc - test_acc)

                results[name] = {
                    "model_name": name,
                    "problem_type": "classification",
                    "test_accuracy": round(test_acc, 4),
                    "train_accuracy": round(train_acc, 4),
                    "test_f1_macro": round(test_f1, 4),
                    "train_f1_macro": round(train_f1, 4),
                    "overfit_gap": round(overfit, 4),
                    "training_time_sec": elapsed,
                    "interpretability_score": INTERPRETABILITY.get(name, 5),
                    "cv_folds": CV_FOLDS,
                    "status": "success",
                }
            else:
                test_r2   = float(np.mean(cv_results["test_r2"]))
                train_r2  = float(np.mean(cv_results["train_r2"]))
                test_mae  = -float(np.mean(cv_results["test_neg_mae"]))
                train_mae = -float(np.mean(cv_results["train_neg_mae"]))
                overfit   = max(0.0, train_r2 - test_r2)

                results[name] = {
                    "model_name": name,
                    "problem_type": "regression",
                    "test_r2": round(test_r2, 4),
                    "train_r2": round(train_r2, 4),
                    "test_mae": round(test_mae, 4),
                    "train_mae": round(train_mae, 4),
                    "overfit_gap": round(overfit, 4),
                    "training_time_sec": elapsed,
                    "interpretability_score": INTERPRETABILITY.get(name, 5),
                    "cv_folds": CV_FOLDS,
                    "status": "success",
                }

            # Train final model on full data for SHAP + prediction endpoint
            model.fit(X_arr, y_arr)
            trained_models[name] = model

        except Exception as e:
            results[name] = {"model_name": name, "status": "failed", "error": str(e)}

    return results, trained_models