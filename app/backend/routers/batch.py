import io

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, UploadFile

from app.backend.config import get_settings
from app.backend.constants import LABEL_NAMES
from app.backend.schemas import BatchResponse, BatchRow

router = APIRouter()


@router.post("/batch", response_model=BatchResponse)
async def batch(request: Request, file: UploadFile) -> BatchResponse:
    settings = get_settings()
    if file.size is not None and file.size > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File quá lớn.")
    raw = await file.read()
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File quá lớn.")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"CSV không đọc được: {e}")
    if "free_text" not in df.columns:
        raise HTTPException(status_code=400, detail="Thiếu cột 'free_text'.")
    max_rows = request.app.state.settings.max_batch_rows
    df = df.head(max_rows)
    reg = request.app.state.registry
    counts = {name: 0 for name in LABEL_NAMES}
    rows = []
    for text in df["free_text"].astype(str).tolist():
        proba = [float(p) for p in reg.predict_proba("LogisticRegression", text)]
        label = int(np.argmax(proba))
        counts[LABEL_NAMES[label]] += 1
        rows.append(BatchRow(text=text, label=label,
                             label_name=LABEL_NAMES[label], proba=proba))
    total = len(rows)
    toxic = counts["OFFENSIVE"] + counts["HATE"]
    return BatchResponse(
        total=total, counts=counts,
        toxic_ratio=(toxic / total if total else 0.0), rows=rows,
    )
