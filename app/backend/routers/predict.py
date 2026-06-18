import time

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from app.backend.constants import DISPLAY_NAMES, LABEL_NAMES, MODEL_ORDER
from app.backend.schemas import PredictResponse, ShowdownResponse, TextRequest
from app.backend.services.explain import explain_tokens

router = APIRouter()


def _validate(req: TextRequest, max_len: int) -> str:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text rỗng.")
    if len(text) > max_len:
        raise HTTPException(status_code=400, detail=f"Text quá dài (>{max_len}).")
    return text


@router.post("/predict", response_model=PredictResponse)
def predict(req: TextRequest, request: Request) -> PredictResponse:
    text = _validate(req, request.app.state.settings.max_text_len)
    reg = request.app.state.registry
    phobert = request.app.state.phobert
    if phobert is not None and phobert.available:
        proba = [float(p) for p in phobert.predict_proba(text)]
        model = "PhoBERT-base-v2"
    else:
        proba = [float(p) for p in reg.predict_proba("LogisticRegression", text)]
        model = "Logistic Regression"
    label = int(np.argmax(proba))
    tokens = explain_tokens(reg, text, label)
    return PredictResponse(
        label=label, label_name=LABEL_NAMES[label], proba=proba,
        tokens=tokens, model=model,
    )


@router.post("/showdown", response_model=ShowdownResponse)
def showdown(req: TextRequest, request: Request) -> ShowdownResponse:
    text = _validate(req, request.app.state.settings.max_text_len)
    reg = request.app.state.registry
    phobert = request.app.state.phobert
    results = []
    if phobert is not None and phobert.available:
        t0 = time.perf_counter()
        proba = [float(p) for p in phobert.predict_proba(text)]
        dt = (time.perf_counter() - t0) * 1000
        results.append({
            "name": "PhoBERT", "label": int(np.argmax(proba)),
            "proba": proba, "latency_ms": round(dt, 2),
        })
    results.extend(reg.predict_all(text))
    # order by MODEL_ORDER
    order = {n: i for i, n in enumerate(MODEL_ORDER)}
    results.sort(key=lambda r: order[r["name"]])
    return ShowdownResponse(models=[
        {**r, "display_name": DISPLAY_NAMES[r["name"]]} for r in results
    ])
