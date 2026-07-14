---
title: ViHSD Moderation API
emoji: 🛡️
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# ViHSD Moderation API

FastAPI backend serving 7 Vietnamese hate-speech models (6 sklearn + PhoBERT-base-v2),
explainability, Gemini rewrite, batch CSV scoring, and insights.

Deployed on the **Gradio SDK** (free CPU-basic, 16 GB RAM) instead of the Docker
SDK, which became paid in mid-2026. `app.py` mounts the whole FastAPI app and
binds uvicorn on port 7860; the Gradio landing page lives at `/ui`.

## Required Space secrets / variables
- `GEMINI_API_KEY` (secret) — Google AI Studio key (for /rewrite)
- `PHOBERT_REPO` (variable) — HF Hub repo id of the fine-tuned PhoBERT (e.g. `lesliu/vihsd-phobert`)
- `CORS_ORIGINS` (variable) — comma-separated allowed origins (your Vercel URL)
- `HF_HOME` (variable) — `/tmp/hf` (only /tmp is writable on Spaces)
- `MONITOR_STATE_PATH` (variable) — `/tmp/monitor_state.json`

## Space repo layout
Push into the Space repo root: this `README.md`, `app.py`, `requirements.txt`,
the `app/` package (with `app/ml` and `app/backend`), and the `artifacts/` folder
(sklearn `.pkl`s; `model_rf.pkl` is ~179 MB → Git LFS). PhoBERT is **not** committed;
it is pulled at runtime from the Hub via `PHOBERT_REPO`.
