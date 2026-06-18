---
title: ViHSD Moderation API
emoji: 🛡️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# ViHSD Moderation API

FastAPI backend serving 7 Vietnamese hate-speech models (6 sklearn + PhoBERT-base-v2),
explainability, Gemini rewrite, batch CSV scoring, and insights.

## Required Space secrets
- `GEMINI_API_KEY` — Google AI Studio key (for /rewrite)
- `PHOBERT_REPO` — HF Hub model repo id of the fine-tuned PhoBERT
- `CORS_ORIGINS` — comma-separated allowed origins (your Vercel URL)

The Dockerfile (copied to the Space root) builds the image; `artifacts/` must be
committed into the Space repo alongside it.
