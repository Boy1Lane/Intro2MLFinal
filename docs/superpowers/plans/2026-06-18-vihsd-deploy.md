# ViHSD Deploy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans for the artifact-creation tasks. The actual deploy/upload steps require the user's credentials and are run by the user (marked **[USER]**).

**Goal:** Ship the ViHSD app: FastAPI backend + PhoBERT on a Hugging Face Space (Docker SDK), Next.js frontend on Vercel, wired together with CORS.

**Architecture:** The backend Docker image bundles the small sklearn artifacts and loads PhoBERT from a HF Hub model repo at runtime. The Space exposes the FastAPI app on port 7860. Vercel builds the Next.js app with `NEXT_PUBLIC_API_URL` pointing at the Space. CORS on the backend allows the Vercel origin.

**Tech Stack:** Docker, Hugging Face Spaces (docker SDK), Vercel, huggingface_hub.

## Global Constraints

- Secrets via platform secret stores only — never commit `GEMINI_API_KEY`, HF tokens, or any key. Backend reads `GEMINI_API_KEY`, `PHOBERT_REPO`, `ARTIFACTS_DIR`, `CORS_ORIGINS` from env.
- HF Space Docker apps must listen on port **7860**.
- The sklearn artifacts (`artifacts/`) are produced by the user on Colab (see backend plan §7) and must be present in the build context before building the backend image.
- PhoBERT fine-tuned weights live in a HF Hub **model** repo (not the Space), referenced by `PHOBERT_REPO`.
- All paths relative to repo root.

## Prerequisites (user-provided, see backend plan §7)

- `artifacts/` populated: `tfidf_vectorizer.pkl`, `bow_vectorizer.pkl`, `tfidf_svd.pkl`, `models/model_*.pkl`.
- A fine-tuned PhoBERT pushed to a HF Hub model repo (the Colab `phobert_best/`).
- Accounts: Hugging Face (with a write token), Vercel, Google AI Studio (Gemini key).

## File Structure

```
app/backend/
  Dockerfile           # build the FastAPI+PhoBERT image
  .dockerignore
  space/
    README.md          # HF Space config header (sdk: docker, app_port: 7860)
app/frontend/
  vercel.json          # build/framework hints (optional but explicit)
docs/
  DEPLOY.md            # human runbook tying it all together
scripts/
  upload_phobert.py    # helper to push Colab phobert_best/ to HF Hub (user runs)
```

---

### Task 1: Backend Dockerfile + dockerignore

**Files:**
- Create: `app/backend/Dockerfile`, `app/backend/.dockerignore`
- Test: local docker build (or, if docker unavailable, a dry validation described below)

**Interfaces:** none (produces a buildable image).

- [ ] **Step 1: Create the Dockerfile**

`app/backend/Dockerfile`:
```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf \
    ARTIFACTS_DIR=/app/artifacts

WORKDIR /app

# System deps kept minimal; torch wheels are self-contained
COPY app/backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code (the ml package + backend package) and artifacts
COPY app/__init__.py ./app/__init__.py
COPY app/ml ./app/ml
COPY app/backend ./app/backend
COPY artifacts ./artifacts

# HF Spaces require listening on 7860
EXPOSE 7860
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

- [ ] **Step 2: Create .dockerignore**

`app/backend/.dockerignore`:
```
**/__pycache__/
**/*.pyc
**/.pytest_cache/
app/frontend/
app/backend/tests/
docs/
notebooks/
reports/
data/
*.ipynb
.git/
```

Note: `.dockerignore` paths are relative to the build context (repo root). The build must be run from repo root so `artifacts/` and `app/` are in context.

- [ ] **Step 3: Validate the build**

If Docker is available:
Run: `docker build -f app/backend/Dockerfile -t vihsd-backend .` (from repo root, with `artifacts/` present)
Expected: image builds; `pip install` succeeds.

If Docker is NOT available in this environment, validate statically instead:
Run: `python -c "import pathlib; [print(p, pathlib.Path(p).exists()) for p in ['app/backend/requirements.txt','app/ml','app/backend','artifacts']]"`
Expected: `app/backend/requirements.txt True`, `app/ml True`, `app/backend True`. (`artifacts` may be False until the user adds Colab output — note this in the report; it is the documented user prerequisite.)

- [ ] **Step 4: Commit**

```bash
git add app/backend/Dockerfile app/backend/.dockerignore
git commit -m "feat(deploy): backend Dockerfile for HF Spaces (port 7860)"
```

---

### Task 2: HF Space config + PhoBERT upload helper

**Files:**
- Create: `app/backend/space/README.md`, `scripts/upload_phobert.py`
- Test: import-check the helper script

**Interfaces:** `upload_phobert.py` is a CLI: `python scripts/upload_phobert.py <local_dir> <repo_id>` — pushes a local PhoBERT folder to a HF Hub model repo. **[USER]** runs it with their token.

- [ ] **Step 1: Create the Space README (config header)**

`app/backend/space/README.md` — this is the file that configures a Hugging Face Space when placed at the Space repo root:
```markdown
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

The Dockerfile (copied to the Space root) builds the image; artifacts/ must be
committed into the Space repo alongside it.
```

- [ ] **Step 2: Write the PhoBERT upload helper**

`scripts/upload_phobert.py`:
```python
"""Push a local fine-tuned PhoBERT folder to a Hugging Face Hub model repo.

Usage:
    HF_TOKEN=hf_xxx python scripts/upload_phobert.py output/phobert/phobert_best my-user/vihsd-phobert
"""
import os
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: upload_phobert.py <local_dir> <repo_id>", file=sys.stderr)
        return 2
    local_dir, repo_id = sys.argv[1], sys.argv[2]
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN env var (a HF write token).", file=sys.stderr)
        return 2
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")
    print(f"Uploaded {local_dir} -> {repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Import-check the helper (no network)**

Run: `python -c "import ast; ast.parse(open('scripts/upload_phobert.py').read()); print('ok')"`
Expected: `ok`. (Full run requires `huggingface_hub` + a token; that is a **[USER]** step.)

- [ ] **Step 4: Commit**

```bash
git add app/backend/space/README.md scripts/upload_phobert.py
git commit -m "feat(deploy): HF Space config header + PhoBERT upload helper"
```

---

### Task 3: Frontend Vercel config + deploy runbook

**Files:**
- Create: `app/frontend/vercel.json`, `docs/DEPLOY.md`
- Test: JSON validity check

**Interfaces:** none.

- [ ] **Step 1: Create vercel.json**

`app/frontend/vercel.json`:
```json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "framework": "nextjs",
  "buildCommand": "next build",
  "installCommand": "npm install"
}
```

Note: set the Vercel project's **Root Directory** to `app/frontend` in the Vercel dashboard, and add `NEXT_PUBLIC_API_URL` as a project env var pointing at the Space URL (`https://<user>-<space>.hf.space`).

- [ ] **Step 2: Validate JSON**

Run: `python -c "import json; json.load(open('app/frontend/vercel.json')); print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Write the deploy runbook**

`docs/DEPLOY.md`:
```markdown
# ViHSD Deployment Runbook

## 0. Prerequisites (one-time)
- Populate `artifacts/` from Colab (see backend plan §7).
- Push fine-tuned PhoBERT: `HF_TOKEN=... python scripts/upload_phobert.py output/phobert/phobert_best <user>/vihsd-phobert`
- Have: HF write token, Vercel account, Gemini API key.

## 1. Backend → Hugging Face Space
1. Create a new Space: SDK = **Docker**.
2. Put at the Space repo root: `app/backend/space/README.md` (rename to `README.md`),
   the `Dockerfile` (from `app/backend/Dockerfile`, adjusted COPY paths if the Space
   only contains the backend — simplest is to push the whole repo and keep paths),
   the `app/` package, and the `artifacts/` folder.
3. Set Space **secrets**: `GEMINI_API_KEY`, `PHOBERT_REPO=<user>/vihsd-phobert`,
   `CORS_ORIGINS=https://<your-vercel-app>.vercel.app`.
4. Wait for build; verify `https://<user>-<space>.hf.space/health` returns
   `{"sklearn_loaded": true, ...}`. First `/showdown` call lazy-loads PhoBERT (slow once).

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
```

- [ ] **Step 4: Commit**

```bash
git add app/frontend/vercel.json docs/DEPLOY.md
git commit -m "feat(deploy): Vercel config + deployment runbook"
```

---

## Self-Review

**Spec coverage (design §3, §8):**
- §3 topology (Vercel + HF Space + Gemini server-side + CORS) → Tasks 1,2,3 + runbook. ✅
- §8 deploy steps (PhoBERT→Hub, backend→Space with secrets, frontend→Vercel, CORS) → Task 2 helper + DEPLOY.md. ✅
- Port 7860, env-only secrets, artifacts-in-context constraints encoded. ✅

**Placeholder scan:** No TBD/TODO; runbook uses `<user>`/`<space>` as explicit user-substitution placeholders (not plan gaps). ✅

**Type/path consistency:** Dockerfile COPY paths (`app/__init__.py`, `app/ml`, `app/backend`, `artifacts`) match the backend plan's file layout; `PHOBERT_REPO`/`GEMINI_API_KEY`/`CORS_ORIGINS` env names match `config.Settings`. ✅

**Note:** Most of Task-1 Step 3 (docker build), all of Task 2's actual upload, and the entire runbook execution depend on user credentials/accounts and are **[USER]** steps. The agentic-buildable deliverables are the Dockerfile, .dockerignore, Space README, upload helper, vercel.json, and DEPLOY.md.

---

## Execution Handoff

This is **Plan 3 of 3**. After the buildable artifacts land, the user executes the credentialed deploy steps in `docs/DEPLOY.md`.
