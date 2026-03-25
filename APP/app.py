
# -*- coding: utf-8 -*-
import os
import io
import yaml
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

with open(os.path.join(BASE_DIR, "registry.yaml"), "r", encoding="utf-8") as f:
    REG = yaml.safe_load(f)

FEATURES = REG.get("features", ["n", "SHR", "Vp", "Is50"])
TARGET = REG.get("target", "UCS")
MODEL_META = REG["models"]

MODEL_CACHE = {}

def get_model(name: str):
    if name not in MODEL_META:
        raise ValueError(f"Unknown model: {name}")
    if name not in MODEL_CACHE:
        path = os.path.join(MODELS_DIR, MODEL_META[name]["file"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        MODEL_CACHE[name] = joblib.load(path)
    return MODEL_CACHE[name]

app = FastAPI(title="UCS Offline API", version="1.0.0")

class PredictReq(BaseModel):
    model: str
    n: float
    SHR: float
    Vp: float
    Is50: float

@app.get("/")
def root():
    return {"status": "UCS API running", "features": FEATURES, "target": TARGET}

@app.get("/models")
def list_models():
    return {
        "features": FEATURES,
        "target": TARGET,
        "models": [
            {"key": k, "file": MODEL_META[k]["file"], "shap": MODEL_META[k].get("shap", "kernel")}
            for k in MODEL_META.keys()
        ]
    }

@app.post("/predict")
def predict(req: PredictReq):
    try:
        m = get_model(req.model)
        X = pd.DataFrame([[req.n, req.SHR, req.Vp, req.Is50]], columns=FEATURES)
        y = float(m.predict(X)[0])
        return {"model": req.model, "UCS_pred": y}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/batch_predict")
async def batch_predict(
    file: UploadFile = File(...),
    model: str = Query(..., description="model key from /models")
):
    try:
        m = get_model(model)
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            return JSONResponse(status_code=400, content={"error": f"Missing columns: {missing}"})

        for c in FEATURES:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        X = df[FEATURES].copy()
        df["UCS_pred"] = m.predict(X)
        df["Model"] = model

        out_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        filename = f"UCS_predictions_{model}.csv".replace(" ", "_")

        return StreamingResponse(
            io.BytesIO(out_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
