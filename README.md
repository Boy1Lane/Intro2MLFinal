# Vietnamese Hate Speech Detection (ViHSD)

Hệ thống tự động phát hiện ngôn từ thù ghét trên mạng xã hội tiếng Việt, phân loại bình luận thành 3 lớp: **CLEAN** (Sạch), **OFFENSIVE** (Xúc phạm) và **HATE** (Thù ghét).

Đồ án cuối kỳ môn Nhập môn Máy học (Intro to ML) — Trường ĐH Khoa học Tự nhiên, ĐHQG-HCM.
Repo gồm cả pipeline huấn luyện (notebooks) lẫn một ứng dụng web demo end-to-end (FastAPI + Next.js).

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Dữ liệu](#dữ-liệu)
- [Pipeline & mô hình](#pipeline--mô-hình)
- [Kết quả](#kết-quả)
- [Cấu trúc repo](#cấu-trúc-repo)
- [Ứng dụng web](#ứng-dụng-web)
- [Chạy thử ở local](#chạy-thử-ở-local)
- [API](#api)
- [Kiểm thử](#kiểm-thử)
- [Triển khai](#triển-khai)

---

## Tổng quan

Bài toán: phân loại văn bản 3 lớp trên bộ **ViHSD** (Vietnamese Hate Speech Detection).
Hướng tiếp cận: so sánh 6 mô hình machine learning cổ điển (đặc trưng TF-IDF / BoW) với một mô hình
deep learning fine-tune từ **PhoBERT-base-v2**, sau đó đóng gói mô hình tốt nhất vào một
"Comment Moderation Studio" có thể tương tác trực tiếp.

Web app cung cấp 4 năng lực:

1. **Studio** — chấm điểm một bình luận: phán quyết + mức tin cậy + giải thích token (mô hình tuyến tính) + so sánh 7 mô hình.
2. **Viết lại lịch sự** — dùng Gemini gợi ý bản viết lại không độc hại.
3. **Mô phỏng** — tải CSV để chấm điểm hàng loạt, xem phân bố nhãn và các bình luận độc hại nhất.
4. **Kết quả mô hình** — bảng metrics của 7 mô hình trên tập test.

## Dữ liệu

Bộ ViHSD chia sẵn 3 tập (cột chính: `free_text`, `label_id`):

| Tập   | Số dòng | File |
|-------|--------:|------|
| Train | 24.774  | `data/train.csv` |
| Dev   |  2.678  | `data/dev.csv`   |
| Test  |  6.690  | `data/test.csv`  |

Mỗi tập có thêm phiên bản đã làm sạch (`*_clean.csv`) và làm sạch theo cấu trúc (`*_structural_clean.csv`)
sinh ra từ notebook tiền xử lý. Nhãn: `0 = CLEAN`, `1 = OFFENSIVE`, `2 = HATE`.

## Pipeline & mô hình

Tiền xử lý (`app/ml/preprocessing.py`, `app/ml/teencode.py`): bỏ URL/email/số điện thoại/HTML/emoji,
chuẩn hóa teencode tiếng Việt, làm sạch khoảng trắng. Đặc trưng: TF-IDF và Bag-of-Words (+ SVD).

Quy trình thực nghiệm nằm trong `notebooks/`, chạy theo thứ tự:

| Notebook | Nội dung |
|----------|----------|
| `01_cleaning.ipynb` | Làm sạch dữ liệu thô |
| `02_text_preprocessing.ipynb` | Tiền xử lý văn bản, teencode |
| `03_label_distribution_check.ipynb` | Kiểm tra phân bố nhãn (mất cân bằng) |
| `04_eda_1.ipynb`, `05_eda_2.ipynb` | Phân tích khám phá (EDA) |
| `06_model_training.ipynb` | Huấn luyện 6 mô hình sklearn |
| `06_model_training_DL.ipynb` | Fine-tune PhoBERT-base-v2 |
| `07_model_figures.ipynb` | Vẽ hình kết quả |

Artifacts huấn luyện (`tfidf_vectorizer.pkl`, `bow_vectorizer.pkl`, `tfidf_svd.pkl`, `models/model_*.pkl`,
và PhoBERT đã fine-tune) **không** được commit (xem `.gitignore`); backend nạp chúng từ thư mục `artifacts/`
ở local hoặc từ HF Hub khi triển khai.

## Kết quả

Trên tập test (đầy đủ trong `app/backend/insights_data.json`):

| Mô hình | Accuracy | F1_w | F1_macro |
|---------|---------:|-----:|---------:|
| **PhoBERT-base-v2** ⭐ | **0.8558** | **0.8637** | **0.6703** |
| Random Forest | 0.8290 | 0.8080 | 0.5101 |
| Multinomial NB | 0.7907 | 0.8068 | 0.5622 |
| Voting Ensemble | 0.7749 | 0.7919 | 0.5455 |
| Logistic Regression | 0.7717 | 0.7899 | 0.5468 |
| Linear SVC | 0.7736 | 0.7889 | 0.5367 |
| SGD Classifier | 0.7446 | 0.7746 | 0.5406 |

PhoBERT-base-v2 đạt cao nhất ở mọi chỉ số; khoảng cách F1_macro lớn cho thấy ưu thế rõ ở
lớp thiểu số HATE — lớp khó nhất do mất cân bằng dữ liệu.

## Cấu trúc repo

```
.
├── data/                  # CSV train/dev/test (+ clean, structural_clean)
├── notebooks/             # 01→07: cleaning, EDA, training (sklearn + DL)
├── app/
│   ├── ml/                # preprocessing, teencode (dùng chung backend ↔ notebook)
│   ├── backend/           # FastAPI: predict / showdown / rewrite / batch / insights
│   │   ├── routers/       # endpoints
│   │   ├── services/      # sklearn_registry, phobert, gemini, explain, metrics
│   │   └── tests/         # pytest
│   └── frontend/          # Next.js 14 + Tailwind (Comment Moderation Studio)
├── scripts/upload_phobert.py   # đẩy PhoBERT đã fine-tune lên HF Hub
├── docs/                  # báo cáo, runbook triển khai (DEPLOY.md)
├── output/eda/            # hình EDA
└── requirements.txt       # deps cho phần notebook/training
```

## Ứng dụng web

- **Backend** — FastAPI phục vụ 7 mô hình (6 sklearn + PhoBERT), giải thích, viết lại bằng Gemini,
  chấm điểm CSV hàng loạt và bảng metrics. PhoBERT được nạp lazy ở lần `/showdown` đầu tiên.
- **Frontend** — Next.js (App Router) + Tailwind + React Query; biểu đồ bằng Recharts.

## Chạy thử ở local

**Yêu cầu:** Python 3.11+, Node 18+. Thư mục `artifacts/` đã có các file `.pkl` (sinh từ `06_model_training.ipynb`).

### Backend

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r app/backend/requirements.txt

# Biến môi trường (đều có default; chỉ /rewrite và PhoBERT cần thiết lập thêm)
export ARTIFACTS_DIR=artifacts
export PHOBERT_REPO=<user>/vihsd-phobert     # tùy chọn — repo HF Hub PhoBERT đã fine-tune
export GEMINI_API_KEY=<key>                  # tùy chọn — bật endpoint /rewrite
export CORS_ORIGINS=http://localhost:3000

uvicorn app.backend.main:app --reload --port 8000
# kiểm tra: http://localhost:8000/health  → {"sklearn_loaded": true, ...}
```

### Frontend

```bash
cd app/frontend
npm install
cp .env.local.example .env.local            # đặt NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev                                  # http://localhost:3000
```

## API

Base URL mặc định local: `http://localhost:8000`.

| Method | Endpoint     | Mô tả |
|--------|--------------|-------|
| POST   | `/predict`   | Chấm điểm 1 bình luận → nhãn, xác suất, token giải thích |
| POST   | `/showdown`  | Chạy cả 7 mô hình trên cùng 1 bình luận (kèm độ trễ) |
| POST   | `/rewrite`   | Viết lại lịch sự bằng Gemini (cần `GEMINI_API_KEY`) |
| POST   | `/batch`     | Upload CSV (cột `free_text`) → phân bố nhãn + top độc hại |
| GET    | `/insights`  | Bảng metrics 7 mô hình trên tập test |
| GET    | `/health`    | Trạng thái nạp sklearn / PhoBERT |

Tài liệu OpenAPI tương tác: `http://localhost:8000/docs`.

## Kiểm thử

```bash
# Backend
pytest                                  # cấu hình ở pytest.ini

# Frontend
cd app/frontend
npm run test                            # vitest (17 test: component + lib)
npm run build                           # kiểm tra build production
```

## Triển khai

Backend → Hugging Face Space (Docker), Frontend → Vercel. Runbook chi tiết: [`docs/DEPLOY.md`](docs/DEPLOY.md).
Đẩy PhoBERT đã fine-tune lên HF Hub bằng `scripts/upload_phobert.py`.
