"""
main.py — ModelIQ FastAPI Backend
Explainable AutoML Advisor
"""

import io
import os
import traceback
import numpy as np
import pandas as pd

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from modules.problem_detector import detect_problem_type
from modules.preprocessor import preprocess
from modules.feature_intelligence import analyze_features
from modules.model_trainer import train_and_evaluate
from modules.verdict_engine import run_verdict_engine
from modules.explainability import generate_explanations

app = FastAPI(title="ModelIQ - Explainable AutoML Advisor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store for trained models ─────────────────────────────
# Stores the last trained pipeline so /api/predict can reuse it
_session: dict = {}

# Serve frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "service": "ModelIQ"}


@app.post("/api/columns")
async def get_columns(file: UploadFile = File(...)):
    """Returns column names so user can pick the target column."""
    try:
        content = await file.read()
        df = _read_file(content, file.filename)
        return {
            "columns": list(df.columns),
            "shape": list(df.shape),
            "preview": df.head(5).fillna("").to_dict(orient="records"),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    target_col: str = Form(...),
):
    """
    Full AutoML pipeline:
    1. Load data
    2. Detect problem type
    3. Preprocess
    4. Feature intelligence
    5. Train models (5-fold CV, full accuracy)
    6. Verdict Engine (deterministic scoring)
    7. Explanations (SHAP)
    """
    global _session
    try:
        # 1 — Load
        content = await file.read()
        df = _read_file(content, file.filename)

        if target_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{target_col}' not found.")
        if len(df) < 20:
            raise HTTPException(status_code=400, detail="Dataset too small (need ≥ 20 rows).")

        df = df.dropna(subset=[target_col]).reset_index(drop=True)

        # 2 — Detect
        problem_detection = detect_problem_type(df, target_col)
        problem_type = problem_detection["problem_type"]

        # 3 — Preprocess
        X, y, preprocessing_log = preprocess(df, target_col, problem_type)

        # 4 — Feature intelligence
        X_selected, feature_report = analyze_features(X, y, problem_type)
        if X_selected.shape[1] == 0:
            raise HTTPException(status_code=400, detail="No features remaining after preprocessing.")

        # 5 — Train
        model_metrics, trained_models = train_and_evaluate(X_selected, y, problem_type)

        # 6 — Verdict Engine
        verdict = run_verdict_engine(model_metrics, problem_type)

        # 7 — Explainability
        explanations = {}
        winner_name = verdict.get("winner")
        if winner_name and winner_name in trained_models:
            explanations = generate_explanations(
                winner_model=trained_models[winner_name],
                X=X_selected,
                feature_names=list(X_selected.columns),
                problem_type=problem_type,
                winner_name=winner_name,
            )

        # ── Cache pipeline for /api/predict ───────────────────────────────
        # Store the preprocessing pipeline so we can transform new inputs identically
        _, _, preprocessing_log_full = preprocess(df, target_col, problem_type)
        
        # Re-run preprocess to get fitted encoders/scalers - store column metadata
        original_cols = [c for c in df.columns if c != target_col]
        
        _session = {
            "winner_name": winner_name,
            "winner_model": trained_models.get(winner_name),
            "trained_models": trained_models,
            "feature_names": list(X_selected.columns),
            "problem_type": problem_type,
            "target_col": target_col,
            "original_df": df,          # Used to refit transforms for prediction
            "problem_detection": problem_detection,
        }

        return {
            "status": "success",
            "dataset_info": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "target_column": target_col,
                "original_columns": original_cols,
            },
            "problem_detection": problem_detection,
            "preprocessing": preprocessing_log,
            "feature_analysis": feature_report,
            "verdict": verdict,
            "explanations": explanations,
            "predict_ready": True,
            "predict_columns": original_cols,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/predict")
async def predict(request: Request):
    global _session

    if not _session or not _session.get("winner_model"):
        raise HTTPException(status_code=400, detail="No trained model available. Run /api/analyze first.")

    body = await request.json()
    inputs = body.get("inputs", {})

    try:
        winner_model = _session["winner_model"]
        problem_type = _session["problem_type"]
        target_col   = _session["target_col"]
        original_df  = _session["original_df"]
        feature_names = _session["feature_names"]

        # ✅ Build input row
        row = {}
        for col in original_df.columns:
            if col == target_col:
                continue

            val = inputs.get(col, None)

            try:
                if pd.api.types.is_numeric_dtype(original_df[col]):
                    val = float(val) if val not in (None, "", "null") else original_df[col].median()
                else:
                    val = str(val) if val not in (None, "", "null") else original_df[col].mode()[0]
            except:
                val = 0

            row[col] = val

        predict_row = pd.DataFrame([row])

        # ✅ Add dummy target (needed for preprocessing)
        dummy_target = (
            original_df[target_col].mode()[0]
            if problem_type == "classification"
            else float(original_df[target_col].mean())
        )
        predict_row[target_col] = dummy_target

        # ✅ Combine with original data
        combined = pd.concat([original_df, predict_row], ignore_index=True)

        # ✅ ONLY preprocess (NO feature selection)
        X_combined, _, _ = preprocess(combined, target_col, problem_type)

        # ✅ Take last row
        X_new = X_combined.iloc[[-1]]

        # ✅ Align columns with training
        for col in feature_names:
            if col not in X_new.columns:
                X_new[col] = 0

        X_new = X_new[feature_names]

        # ✅ DEBUG (optional)
        print("Expected:", len(feature_names))
        print("Got:", X_new.shape[1])

        # ✅ Predict
        raw_pred = winner_model.predict(X_new.values)[0]

        # ✅ Classification extras
        confidence = None
        class_probabilities = None

        if problem_type == "classification" and hasattr(winner_model, "predict_proba"):
            proba = winner_model.predict_proba(X_new.values)[0]
            confidence = float(np.max(proba))
            class_probabilities = {
                str(c): round(float(p), 4)
                for c, p in zip(winner_model.classes_, proba)
            }

        return {
            "status": "success",
            "prediction": float(raw_pred) if isinstance(raw_pred, (np.integer, np.floating)) else str(raw_pred),
            "problem_type": problem_type,
            "model_used": _session["winner_name"],
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "target_column": target_col,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def _read_file(content: bytes, filename: str) -> pd.DataFrame:
    fname = filename.lower() if filename else ""
    if fname.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    elif fname.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    elif fname.endswith(".json"):
        return pd.read_json(io.BytesIO(content))
    else:
        try:
            return pd.read_csv(io.BytesIO(content))
        except Exception:
            raise ValueError("Unsupported file format. Use CSV, XLSX, or JSON.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)