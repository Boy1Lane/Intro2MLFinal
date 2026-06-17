# ViHSD Comment Moderation Studio — Design Spec

**Ngày:** 2026-06-18
**Giai đoạn:** GĐ3 — Deploy web app hoàn chỉnh
**Dự án:** Vietnamese Hate Speech Detection (ViHSD), phân loại 3 lớp: 0=CLEAN, 1=OFFENSIVE, 2=HATE.

## 1. Mục tiêu & vấn đề

Web app phân loại "text vào → 1 nhãn ra" quá tĩnh, nhàm. Mục tiêu: biến demo thành **Comment Moderation Studio** — một pipeline tương tác gộp 4 năng lực, tận dụng toàn bộ model đã train, và deploy public hoàn chỉnh.

4 năng lực:
1. **Explainability** — highlight token đẩy/kéo prediction, gauge confidence, proba 3 lớp.
2. **Model showdown** — 7 model cùng dự đoán 1 câu, so nhãn/confidence/latency, đánh dấu đồng thuận.
3. **Mô phỏng thực tế** — upload CSV → dashboard thống kê độc hại.
4. **Generative rewrite** — phát hiện độc → Gemini viết lại lịch sự → re-classify chứng minh chuyển CLEAN.

## 2. Ràng buộc đã chốt

| Hạng mục | Quyết định |
|---|---|
| Features | Cả 4 (explain + showdown + simulate + rewrite) |
| Stack | FastAPI (backend) + Next.js App Router/TS/Tailwind/shadcn (frontend) |
| LLM rewrite | Gemini API (Google AI Studio), key ở backend secret |
| Deploy | Next.js → Vercel; FastAPI + PhoBERT → Hugging Face Spaces (Docker SDK) |
| UI language | Tiếng Việt |
| Artifact regen | **User tự chạy Colab** (xem §7) |

## 3. Kiến trúc & deploy topology

Monorepo:
```
app/
  backend/    FastAPI — model serving + Gemini proxy
  frontend/   Next.js (App Router, TS, Tailwind, shadcn/ui, framer-motion)
  ml/         shared: preprocessing.py, teencode dict, model loader
artifacts/    *.pkl vectorizers + models/  (sklearn, vài MB, bundle vào Space)
```

Topology:
- **Vercel** ← `frontend/`. Env `NEXT_PUBLIC_API_URL` trỏ Space.
- **HF Spaces (Docker SDK)** ← `backend/`. sklearn pkl bundle trong Space repo. PhoBERT load từ **HF Hub model repo** riêng (lazy-load, cache in-memory).
- **Gemini API** gọi từ backend; key ở Space secret, KHÔNG lộ ra frontend.
- CORS: backend allow Vercel domain.

Lý do Gemini sau backend: giấu key, kiểm soát prompt, re-classify rewrite.

## 4. Backend (FastAPI)

### 4.1 Shared preprocessing
`ml/preprocessing.py` — port `preprocess_text` + teencode dict từ `notebooks/02_text_preprocessing.ipynb` và `06_model_training.ipynb` (`normalize_teencode`). Một nguồn sự thật, dùng chung mọi endpoint + batch.

### 4.2 Model registry
- sklearn (load lúc startup, nhẹ): TF-IDF vectorizer + SVD + 6 model `.pkl` (LR, NB, SVM-calibrated, RF, SGD, Voting). Voting = soft-vote [LR, calibrated-SVM, SGD].
- PhoBERT: lazy-load lần gọi đầu (torch + transformers, ~500MB), cache.

### 4.3 Endpoints
| Method | Route | In → Out |
|---|---|---|
| POST | `/predict` | `{text}` → `{label, proba[3], tokens[]}` — verdict (label/proba) từ PhoBERT; `tokens[]` (explanation) tính từ LogisticRegression (xem §4.4) |
| POST | `/showdown` | `{text}` → `{models:[{name,label,proba[3],latency_ms}]}` — cả 7 |
| POST | `/rewrite` | `{text}` → `{rewritten, before:{label,proba}, after:{label,proba}}` |
| POST | `/batch` | CSV upload (col `free_text`) → `{summary, rows[]}` |
| GET | `/insights` | metrics JSON tĩnh (đọc từ report figures/metrics) |
| GET | `/health` | trạng thái load model |

### 4.4 Explainability
Token highlight bằng linear model exact: `contribution(token) = tfidf[token] × coef[token, pred_class]`. Dùng **LogisticRegression** (có proba, interpretable). Tức thì, không thêm dependency. Lưu ý: verdict ở `/predict` lấy từ PhoBERT (model mạnh nhất) nhưng explanation tokens lấy từ LR vì PhoBERT không có token-attribution rẻ; UI nêu rõ "giải thích dựa trên model tuyến tính". PhoBERT trong showdown chỉ trả label + proba (không attention — tránh nặng).

### 4.5 Gemini rewrite
Prompt = câu gốc + chỉ thị viết lại lịch sự, giữ nghĩa, tiếng Việt. Sau rewrite → chạy lại classifier (`/predict` nội bộ) để show before→after. Loop "độc → sạch" là điểm nhấn demo.

### 4.6 Error handling
- Text rỗng / quá dài → 400 + giới hạn ký tự.
- CSV thiếu col `free_text` / quá lớn → 400, cap số rows.
- Gemini fail → trả message graceful, classifier vẫn chạy.
- PhoBERT load fail → showdown vẫn trả 6 model sklearn.

## 5. Frontend (Next.js)

Routes:
- `/` — **Studio** (single-page pipeline, feature chính).
- `/simulate` — batch CSV → dashboard.
- `/insights` — metrics tĩnh nhúng từ `reports/report_2/figures/`.

### 5.1 `/` Studio — luồng dọc, 1 input nuôi hết
1. **Input bar** — textarea + nút Phân tích + chips ví dụ (sample CLEAN/OFFENSIVE/HATE).
2. **Verdict card** — nhãn lớn (xanh CLEAN / cam OFFENSIVE / đỏ HATE) + gauge confidence + bar proba 3 lớp.
3. **Explain panel** — render câu, highlight token theo contribution (đỏ = đẩy về độc, xanh = kéo về sạch), hover xem điểm.
4. **Showdown table** — 7 model: nhãn, proba bar, latency; đánh dấu đồng thuận/bất đồng.
5. **Rewrite card** — nút "Viết lại lịch sự" → before→after + nhãn đổi đỏ→xanh + animation.

Section 2–5 lazy: bấm Phân tích gọi `/predict` + `/showdown` (skeleton loading); rewrite gọi riêng khi bấm.

### 5.2 `/simulate`
Drag-drop CSV → `/batch` → dashboard: donut phân bố nhãn, % độc hại, top câu độc nhất, bảng có filter.

### 5.3 `/insights`
Grid hình tĩnh từ report figures + bảng metrics. Gắn web app với báo cáo.

State: React Query gọi backend. Loading skeleton + error toast mọi call. Polish thẩm mỹ ("đẹp nhất") xử lý ở bước implement bằng frontend-design skill.

## 6. Testing
- Backend pytest: preprocessing (teencode/url/emoji), `/predict` proba sum≈1, `/showdown` trả 7, `/batch` parse CSV, `/rewrite` với Gemini mock.
- Smoke: 3 câu mẫu (mỗi nhãn) qua mọi endpoint khớp kỳ vọng.
- Frontend: render + 1 happy-path e2e (optional).

## 7. Prerequisite — artifact regeneration (user tự chạy Colab)

Backend không chạy nếu thiếu artifact. **User chịu trách nhiệm chạy:**
1. Rerun `notebooks/06_model_training.ipynb` → xuất `tfidf_vectorizer.pkl`, `tfidf_svd.pkl`, `models/model_*.pkl` vào `artifacts/` (CPU).
2. Rerun `notebooks/06_model_training_DL.ipynb` trên Colab GPU → `phobert_best/` → push **HF Hub model repo**.
3. Verify: load lại từng artifact, predict 1 câu mẫu, khớp nhãn notebook.

Phía ta: cung cấp export cell/script gọn nếu notebook chưa dump đúng đường dẫn `artifacts/`, và loader đọc đúng format.

## 8. Deploy steps
1. PhoBERT → HF Hub model repo.
2. `backend/` → HF Space (Docker SDK), secret `GEMINI_API_KEY`, env `PHOBERT_REPO`. Verify `/health`.
3. `frontend/` → Vercel, env `NEXT_PUBLIC_API_URL`. Verify Studio gọi được Space.
4. CORS allow Vercel domain.

## 9. Out of scope (YAGNI)
Auth, DB, lưu lịch sử user, train lại trong app, multi-language UI, live-feed streaming (nice-to-have, cắt nếu thiếu giờ).
