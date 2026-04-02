"""
main.py — ModelIQ FastAPI Backend
Explainable AutoML Advisor
"""

import io
import os
import traceback

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
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
    2. Detect problem type (deterministic rules)
    3. Preprocess (deterministic rules)
    4. Feature intelligence (deterministic rules)
    5. Train models (cross-validation)
    6. Run Verdict Engine (deterministic scoring rubric)
    7. Generate explanations (SHAP)
    """
    try:
        # 1 — Load
        content = await file.read()
        df = _read_file(content, file.filename)

        if target_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{target_col}' not found.")

        if len(df) < 20:
            raise HTTPException(status_code=400, detail="Dataset too small (need ≥ 20 rows).")

        # Drop rows where target is null
        df = df.dropna(subset=[target_col]).reset_index(drop=True)

        # 2 — Detect problem type
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

        return {
            "status": "success",
            "dataset_info": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "target_column": target_col,
            },
            "problem_detection": problem_detection,
            "preprocessing": preprocessing_log,
            "feature_analysis": feature_report,
            "verdict": verdict,
            "explanations": explanations,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


def _read_file(content: bytes, filename: str) -> pd.DataFrame:
    fname = filename.lower() if filename else ""
    if fname.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    elif fname.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    elif fname.endswith(".json"):
        return pd.read_json(io.BytesIO(content))
    else:
        # Try CSV by default
        try:
            return pd.read_csv(io.BytesIO(content))
        except Exception:
            raise ValueError("Unsupported file format. Use CSV, XLSX, or JSON.")


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)