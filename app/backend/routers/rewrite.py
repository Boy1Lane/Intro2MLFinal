import numpy as np
from fastapi import APIRouter, HTTPException, Request

from app.backend.constants import LABEL_NAMES
from app.backend.schemas import RewriteResponse, TextRequest, Verdict
from app.backend.services.gemini import rewrite_polite

router = APIRouter()


def _verdict(request: Request, text: str) -> Verdict:
    reg = request.app.state.registry
    phobert = request.app.state.phobert
    proba = phobert.try_proba(text) if phobert is not None else None
    if proba is None:
        proba = [float(p) for p in reg.predict_proba("LogisticRegression", text)]
    label = int(np.argmax(proba))
    return Verdict(label=label, label_name=LABEL_NAMES[label], proba=proba)


@router.post("/rewrite", response_model=RewriteResponse)
def rewrite(req: TextRequest, request: Request) -> RewriteResponse:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text rỗng.")
    before = _verdict(request, text)
    try:
        rewritten = rewrite_polite(text, request.app.state.settings.gemini_api_key)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    after = _verdict(request, rewritten)
    return RewriteResponse(rewritten=rewritten, before=before, after=after)
