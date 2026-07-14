FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf \
    ARTIFACTS_DIR=/app/artifacts \
    PORT=8080

WORKDIR /app

# Python deps (torch wheels are self-contained)
COPY app/backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code (ml + backend packages) and sklearn artifacts.
# PhoBERT is pulled at runtime from the HF Hub via PHOBERT_REPO, not baked in.
COPY app/__init__.py ./app/__init__.py
COPY app/ml ./app/ml
COPY app/backend ./app/backend
COPY artifacts ./artifacts

# Cloud Run routes traffic to $PORT (default 8080); honor it.
EXPOSE 8080
CMD exec uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT:-8080}
