# ViHSD Deployment Runbook

## 0. Prerequisites (one-time)
- Populate `artifacts/` from Colab (see `docs/superpowers/plans/2026-06-18-vihsd-backend.md` §7):
  `tfidf_vectorizer.pkl`, `bow_vectorizer.pkl`, `tfidf_svd.pkl`, `models/model_*.pkl`.
- Push fine-tuned PhoBERT:
  `HF_TOKEN=... python scripts/upload_phobert.py output/phobert/phobert_best <user>/vihsd-phobert`
- Have: HF write token, Vercel account, Gemini API key.

## 1. Backend → Hugging Face Space
1. Create a new Space: SDK = **Docker**.
2. Put at the Space repo root: `app/backend/space/README.md` (rename to `README.md`),
   the `Dockerfile` (from `app/backend/Dockerfile`), the `app/` package, and the
   `artifacts/` folder. (Simplest: push the whole repo and keep paths intact.)
3. Set Space **secrets**: `GEMINI_API_KEY`, `PHOBERT_REPO=<user>/vihsd-phobert`,
   `CORS_ORIGINS=https://<your-vercel-app>.vercel.app`.
4. Wait for build; verify `https://<user>-<space>.hf.space/health` returns
   `{"sklearn_loaded": true, ...}`. The first `/showdown` call lazy-loads PhoBERT (slow once).

## 2. Frontend → Vercel
1. Import the GitHub repo in Vercel.
2. Set **Root Directory** = `app/frontend`.
3. Add env var `NEXT_PUBLIC_API_URL=https://<user>-<space>.hf.space`.
4. Deploy. Open the Vercel URL; the Studio should call the Space.

## 3. Wire CORS
- Ensure the Space's `CORS_ORIGINS` exactly matches the Vercel origin (scheme + host,
  no trailing slash). Redeploy the Space if changed.

## 4. Smoke test (production)
- `/` Studio: analyze a sample → verdict + explanation + 7-model showdown + rewrite.
- `/simulate`: upload a small CSV with a `free_text` column → dashboard.
- `/insights`: metrics table loads from `/insights`.
