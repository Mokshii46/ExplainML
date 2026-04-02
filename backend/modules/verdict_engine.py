"""
verdict_engine.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE DETERMINISTIC BRAIN OF ModelIQ.

This is NOT sklearn choosing models. This is a hand-designed,
fully explainable scoring rubric that evaluates every candidate model
across multiple dimensions and produces a traceable verdict.

Scoring Formula (Classification):
  composite = (accuracy × 0.30) + (f1 × 0.30) + (overfit_penalty × 0.25) + (speed × 0.05) + (interpretability × 0.10)

Scoring Formula (Regression):
  composite = (r2 × 0.35) + (mae_score × 0.30) + (overfit_penalty × 0.25) + (speed × 0.05) + (interpretability × 0.05)

Each dimension is scored 0–1 using deterministic rules.
Every verdict is traceable back to its inputs.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from typing import Dict, Any


# ── Rubric Weights ────────────────────────────────────────────────────────────
CLASSIFICATION_WEIGHTS = {
    "accuracy":          0.30,
    "f1_macro":          0.30,
    "overfit_penalty":   0.25,
    "speed":             0.05,
    "interpretability":  0.10,
}

REGRESSION_WEIGHTS = {
    "r2":                0.35,
    "mae_score":         0.30,
    "overfit_penalty":   0.25,
    "speed":             0.05,
    "interpretability":  0.05,
}

# ── Thresholds & Rules ────────────────────────────────────────────────────────
OVERFIT_RULES = [
    {"max_gap": 0.02, "penalty": 0.00, "label": "No overfitting", "description": "Train/test gap ≤ 2% — model generalises well."},
    {"max_gap": 0.05, "penalty": 0.10, "label": "Slight overfitting", "description": "Train/test gap 2–5% — minor memorisation."},
    {"max_gap": 0.10, "penalty": 0.25, "label": "Moderate overfitting", "description": "Train/test gap 5–10% — model partially memorises training data."},
    {"max_gap": 0.20, "penalty": 0.50, "label": "High overfitting", "description": "Train/test gap 10–20% — poor generalisation expected."},
    {"max_gap": 1.00, "penalty": 0.80, "label": "Severe overfitting", "description": "Train/test gap >20% — model is memorising, not learning."},
]

SPEED_RULES = [
    {"max_sec": 1.0,  "score": 1.00, "label": "Very fast"},
    {"max_sec": 3.0,  "score": 0.80, "label": "Fast"},
    {"max_sec": 10.0, "score": 0.55, "label": "Moderate"},
    {"max_sec": 30.0, "score": 0.30, "label": "Slow"},
    {"max_sec": 1e9,  "score": 0.10, "label": "Very slow"},
]

DISQUALIFICATION_RULES = [
    {
        "id": "DQ1",
        "condition": lambda m: m.get("test_accuracy", 1) < 0.50 and m["problem_type"] == "classification",
        "reason": "DQ1 — Accuracy below 50%: worse than random guessing on binary problems.",
    },
    {
        "id": "DQ2",
        "condition": lambda m: m.get("test_r2", 1) < 0.0 and m["problem_type"] == "regression",
        "reason": "DQ2 — Negative R²: model performs worse than a horizontal mean line.",
    },
    {
        "id": "DQ3",
        "condition": lambda m: m.get("overfit_gap", 0) > 0.25,
        "reason": "DQ3 — Overfit gap > 25%: model cannot be trusted on unseen data.",
    },
    {
        "id": "DQ4",
        "condition": lambda m: m.get("status") == "failed",
        "reason": "DQ4 — Model training failed with an error.",
    },
]


def run_verdict_engine(metrics: Dict[str, Any], problem_type: str) -> dict:
    """
    Core scoring engine. Evaluates each model, assigns scores per dimension,
    applies disqualification rules, and selects a winner.

    Returns full verdict dict with:
    - per-model dimension scores
    - composite scores
    - disqualification log
    - winner with human-readable rationale
    """
    verdicts = {}
    weights = CLASSIFICATION_WEIGHTS if problem_type == "classification" else REGRESSION_WEIGHTS

    for model_name, m in metrics.items():
        # ── Check disqualifications first ─────────────────────────────────────
        dq_reasons = []
        for rule in DISQUALIFICATION_RULES:
            try:
                if rule["condition"](m):
                    dq_reasons.append(rule["reason"])
            except Exception:
                pass

        if dq_reasons:
            verdicts[model_name] = {
                "model_name": model_name,
                "status": "disqualified",
                "disqualification_reasons": dq_reasons,
                "composite_score": 0.0,
                "dimension_scores": {},
                "raw_metrics": m,
            }
            continue

        # ── Compute dimension scores ──────────────────────────────────────────
        dim = {}

        if problem_type == "classification":
            dim["accuracy"] = {
                "raw": m["test_accuracy"],
                "score": round(float(m["test_accuracy"]), 4),
                "weight": weights["accuracy"],
                "contribution": round(float(m["test_accuracy"]) * weights["accuracy"], 5),
                "explanation": f"CV accuracy = {m['test_accuracy']:.1%}",
            }
            dim["f1_macro"] = {
                "raw": m["test_f1_macro"],
                "score": round(float(m["test_f1_macro"]), 4),
                "weight": weights["f1_macro"],
                "contribution": round(float(m["test_f1_macro"]) * weights["f1_macro"], 5),
                "explanation": f"Macro F1 = {m['test_f1_macro']:.1%} (handles class imbalance)",
            }
        else:
            r2_score_val = max(0.0, float(m["test_r2"]))  # floor at 0
            dim["r2"] = {
                "raw": m["test_r2"],
                "score": round(r2_score_val, 4),
                "weight": weights["r2"],
                "contribution": round(r2_score_val * weights["r2"], 5),
                "explanation": f"CV R² = {m['test_r2']:.4f} (variance explained by model)",
            }
            # MAE score: lower MAE is better. Normalise relative to other models later.
            dim["mae_score"] = {
                "raw": m["test_mae"],
                "score": None,  # computed after all models are scored
                "weight": weights["mae_score"],
                "contribution": None,
                "explanation": f"Mean Absolute Error = {m['test_mae']:.4f}",
            }

        # Overfit penalty
        gap = float(m.get("overfit_gap", 0))
        overfit_penalty, overfit_label, overfit_desc = _apply_overfit_rule(gap)
        overfit_score = 1.0 - overfit_penalty
        dim["overfit_penalty"] = {
            "raw_gap": round(gap, 4),
            "penalty_applied": round(overfit_penalty, 4),
            "score": round(overfit_score, 4),
            "weight": weights["overfit_penalty"],
            "contribution": round(overfit_score * weights["overfit_penalty"], 5),
            "label": overfit_label,
            "explanation": overfit_desc,
        }

        # Speed score
        t = float(m.get("training_time_sec", 99))
        speed_score, speed_label = _apply_speed_rule(t)
        dim["speed"] = {
            "raw_time_sec": t,
            "score": round(speed_score, 4),
            "weight": weights["speed"],
            "contribution": round(speed_score * weights["speed"], 5),
            "label": speed_label,
            "explanation": f"Training time = {t:.2f}s → {speed_label}",
        }

        # Interpretability score (0–10 → 0–1)
        interp_raw = int(m.get("interpretability_score", 5))
        interp_norm = interp_raw / 10.0
        dim["interpretability"] = {
            "raw": interp_raw,
            "score": round(interp_norm, 4),
            "weight": weights["interpretability"],
            "contribution": round(interp_norm * weights["interpretability"], 5),
            "explanation": f"Hand-designed interpretability score: {interp_raw}/10",
        }

        verdicts[model_name] = {
            "model_name": model_name,
            "status": "evaluated",
            "dimension_scores": dim,
            "raw_metrics": m,
        }

    # ── Normalize MAE scores cross-model (regression) ─────────────────────────
    if problem_type == "regression":
        _normalize_mae_scores(verdicts)

    # ── Compute composite scores ───────────────────────────────────────────────
    for model_name, v in verdicts.items():
        if v["status"] != "evaluated":
            continue
        dim = v["dimension_scores"]
        composite = sum(
            d["contribution"] for d in dim.values()
            if d.get("contribution") is not None
        )
        v["composite_score"] = round(composite, 5)

    # ── Select winner ─────────────────────────────────────────────────────────
    eligible = {k: v for k, v in verdicts.items() if v["status"] == "evaluated"}
    if not eligible:
        winner_name = None
        runner_up = None
    else:
        ranked = sorted(eligible.items(), key=lambda x: x[1]["composite_score"], reverse=True)
        winner_name = ranked[0][0]
        runner_up = ranked[1][0] if len(ranked) > 1 else None

        # Mark winner
        verdicts[winner_name]["verdict"] = "SELECTED"
        verdicts[winner_name]["verdict_rationale"] = _build_rationale(
            winner_name, verdicts[winner_name], runner_up,
            verdicts.get(runner_up, {}), weights, problem_type
        )

        for name, _ in ranked[1:]:
            verdicts[name]["verdict"] = "REJECTED"
            verdicts[name]["verdict_rationale"] = _build_rejection_rationale(
                name, verdicts[name], winner_name, verdicts[winner_name], weights
            )

    for name, v in verdicts.items():
        if v["status"] == "disqualified":
            v["verdict"] = "DISQUALIFIED"
            v["verdict_rationale"] = "Model failed one or more hard disqualification rules: " + "; ".join(v["disqualification_reasons"])

    # ── Build ranked leaderboard ──────────────────────────────────────────────
    leaderboard = sorted(
        verdicts.values(),
        key=lambda x: x.get("composite_score", -1),
        reverse=True
    )

    return {
        "winner": winner_name,
        "runner_up": runner_up,
        "problem_type": problem_type,
        "weights_used": weights,
        "leaderboard": leaderboard,
        "verdicts": verdicts,
        "engine_rules": {
            "overfit_rules": OVERFIT_RULES,
            "speed_rules": SPEED_RULES,
            "disqualification_rules": [r["id"] + ": " + r["reason"] for r in DISQUALIFICATION_RULES],
        }
    }


def _apply_overfit_rule(gap: float):
    for rule in OVERFIT_RULES:
        if gap <= rule["max_gap"]:
            return rule["penalty"], rule["label"], rule["description"]
    return 0.80, "Severe overfitting", "Train/test gap >20%"


def _apply_speed_rule(seconds: float):
    for rule in SPEED_RULES:
        if seconds <= rule["max_sec"]:
            return rule["score"], rule["label"]
    return 0.10, "Very slow"


def _normalize_mae_scores(verdicts: dict):
    """For regression: normalise MAE so lowest MAE = score 1.0."""
    maes = {
        k: v["dimension_scores"]["mae_score"]["raw"]
        for k, v in verdicts.items()
        if v["status"] == "evaluated" and "mae_score" in v["dimension_scores"]
    }
    if not maes:
        return
    min_mae = min(maes.values())
    max_mae = max(maes.values())
    denom = max_mae - min_mae if max_mae != min_mae else 1.0

    weights = REGRESSION_WEIGHTS

    for k, mae_val in maes.items():
        score = 1.0 - ((mae_val - min_mae) / denom)
        dim = verdicts[k]["dimension_scores"]["mae_score"]
        dim["score"] = round(score, 4)
        dim["contribution"] = round(score * weights["mae_score"], 5)
        dim["explanation"] += f" | Normalised score: {score:.2%} (relative to range [{min_mae:.4f}, {max_mae:.4f}])"


def _build_rationale(winner_name, winner, runner_up_name, runner_up, weights, problem_type) -> str:
    w = winner
    dim = w["dimension_scores"]
    cs = w["composite_score"]
    ru_cs = runner_up.get("composite_score", 0) if runner_up else 0

    lines = [
        f"**{winner_name}** selected with composite score {cs:.4f}.",
        "",
        "Scoring breakdown:",
    ]
    for dim_name, d in dim.items():
        if d.get("score") is not None:
            lines.append(f"  • {dim_name}: {d['score']:.2%} × {d['weight']} weight = {d.get('contribution',0):.4f} pts — {d['explanation']}")
    lines.append("")
    if runner_up_name:
        margin = cs - ru_cs
        lines.append(f"Runner-up: **{runner_up_name}** scored {ru_cs:.4f} (margin: {margin:.4f}).")

    return "\n".join(lines)


def _build_rejection_rationale(name, model, winner_name, winner, weights) -> str:
    cs = model.get("composite_score", 0)
    wcs = winner.get("composite_score", 0)
    margin = wcs - cs
    strongest_dim = max(
        model["dimension_scores"].items(),
        key=lambda x: x[1].get("contribution", 0)
    ) if model["dimension_scores"] else ("N/A", {})
    return (
        f"**{name}** scored {cs:.4f} — {margin:.4f} pts behind winner **{winner_name}** ({wcs:.4f}). "
        f"Strongest dimension: {strongest_dim[0]} ({strongest_dim[1].get('score', 0):.2%}). "
        "Not selected due to lower composite score under the ModelIQ rubric."
    )