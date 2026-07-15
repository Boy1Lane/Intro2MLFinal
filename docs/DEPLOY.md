# ViHSD Deployment Runbook

Backend used to run on a Hugging Face Space (Docker); moved to Cloud Run after HF Space
Docker went paid. Cloud Run also means the backend has no long-running background process,
so Monitor only scans on demand (button click), not on a schedule.

## 0. Prerequisites (one-time)
- Populate `artifacts/` from Colab (see `docs/superpowers/plans/2026-06-18-vihsd-backend.md` §7):
  `tfidf_vectorizer.pkl`, `bow_vectorizer.pkl`, `tfidf_svd.pkl`, `models/model_*.pkl`.
- Push fine-tuned PhoBERT:
  `HF_TOKEN=... python scripts/upload_phobert.py output/phobert/phobert_best <user>/vihsd-phobert`
- Have: GCP project with Cloud Run + Artifact Registry enabled, Vercel account, Gemini API key.

## 1. Backend → Google Cloud Run
1. Build and deploy from the repo root (uses the root `Dockerfile`, not `app/backend/Dockerfile`):
   ```bash
   gcloud run deploy vihsd-api \
     --source . \
     --region asia-southeast1 \
     --cpu 2 --memory 4Gi \
     --max-instances 1 \
     --set-env-vars PHOBERT_REPO=<user>/vihsd-phobert,CORS_ORIGINS=https://<your-vercel-app>.vercel.app \
     --set-secrets GEMINI_API_KEY=GEMINI_API_KEY:latest
   ```
   `artifacts/` is baked into the image (`COPY artifacts ./artifacts` in the Dockerfile);
   PhoBERT is **not** baked in — it's pulled at runtime from the HF Hub via `PHOBERT_REPO`.
2. Wait for the deploy; verify `https://<service>-<hash>.<region>.run.app/health` returns
   `{"sklearn_loaded": true, ...}`. The first `/showdown` call lazy-loads PhoBERT (slow once).
3. `--max-instances=1` keeps cost near-zero (scales to 0 when idle) but means no concurrent
   requests scale out — acceptable for a demo, not for production traffic.

## 2. Frontend → Vercel
1. Import the GitHub repo in Vercel.
2. Set **Root Directory** = `app/frontend`.
3. Add env var `NEXT_PUBLIC_API_URL=https://<service>-<hash>.<region>.run.app`.
4. Deploy. Open the Vercel URL; the Studio should call the Cloud Run service.

## 3. Wire CORS
- Ensure the Cloud Run service's `CORS_ORIGINS` exactly matches the Vercel origin (scheme +
  host, no trailing slash). Redeploy the service if changed.

## 4. Smoke test (production)
- `/` Studio: analyze a sample → verdict + explanation + 7-model showdown + rewrite.
- `/simulate`: upload a small CSV with a `free_text` column → dashboard.
- `/insights`: metrics table loads from `/insights`.
- `/monitor`: create a watch on a forum thread URL, hit "scan" → comments + labels appear.
