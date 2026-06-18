FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf \
    ARTIFACTS_DIR=/app/artifacts

WORKDIR /app

# Python deps (torch wheels are self-contained)
COPY app/backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code (ml + backend packages) and artifacts
COPY app/__init__.py ./app/__init__.py
COPY app/ml ./app/ml
COPY app/backend ./app/backend
COPY artifacts ./artifacts

# HF Spaces require listening on 7860
EXPOSE 7860
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
