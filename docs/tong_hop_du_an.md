# 📋 TỔNG HỢP DỰ ÁN: PHÂN LOẠI BÌNH LUẬN TIÊU CỰC TIẾNG VIỆT

## ✅ Trạng thái: HOÀN TẤT 100%

---

## Giai đoạn 1: Phân chia công việc
📄 Xem chi tiết: [giai_doan_1_phan_chia_cong_viec.md](file:///C:/Users/Admin/.gemini/antigravity/brain/f7b1f469-07a1-48db-91ce-9771ae973ec0/artifacts/giai_doan_1_phan_chia_cong_viec.md)

---

## Giai đoạn 2: Thực thi công việc

### 🧑‍💻 Thành viên 1: Data Engineer — Trần Thanh Đạt
**File:** [member1_data_engineer.py](file:///c:/Users/Admin/OneDrive%20-%20VNU-HCMUS/Documents/FinalLABML/member1_data_engineer.py)  
**Kết quả:**
- Pipeline tiền xử lý 10 bước (lowercase, xóa URL/email, emoji, teencode, ký tự đặc biệt...)
- Từ điển teencode 80+ từ tiếng Việt
- Train: 24,048 → 22,510 (-6.4%) | Dev: 2,672 → 2,633 (-1.5%) | Test: 6,680 → 6,527 (-2.3%)
- Output: `train_clean.csv`, `dev_clean.csv`, `test_clean.csv`

---

### 📊 Thành viên 2: Data Analyst — Lê Minh Hải
**File:** [member2_eda.py](file:///c:/Users/Admin/OneDrive%20-%20VNU-HCMUS/Documents/FinalLABML/member2_eda.py)  
**9 biểu đồ đã tạo** trong `eda_output/`:

| Biểu đồ | Mô tả |
|---------|-------|
| `label_distribution.png` | Phân bố nhãn 3 tập |
| `text_length_distribution.png` | Histogram độ dài/số từ |
| `boxplot_text_length.png` | Boxplot theo nhãn |
| `top20_words_by_label.png` | Top-20 từ phổ biến |
| `bigram_analysis.png` | Top bigrams |
| `correlation_heatmap.png` | Ma trận tương quan |
| `train_dev_test_comparison.png` | So sánh phân bố |
| `confusion_matrix.png` | Ma trận nhầm lẫn |
| `model_comparison.png` | So sánh mô hình |

**Insight:** Bình luận thù ghét dài gấp đôi (~83 ký tự vs ~44). Dữ liệu mất cân bằng 12:1:1.6.

---

### 🤖 Thành viên 3: ML Engineer 1 — Phan Hoàng Quốc Huy
**File:** [member3_ml_traditional.py](file:///c:/Users/Admin/OneDrive%20-%20VNU-HCMUS/Documents/FinalLABML/member3_ml_traditional.py)

| Mô hình | DEV Accuracy | DEV F1-weighted | TEST F1-weighted |
|---------|-------------|----------------|-----------------|
| Logistic Regression | 0.8420 | 0.8449 | 0.8535 |
| Multinomial NB | 0.8409 | 0.8232 | 0.8366 |
| **Linear SVM** | **0.8507** | **0.8459** | **0.8585** |

---

### ⚙️ Thành viên 4: ML Engineer 2 — Cao Tiến Thành
**File:** [member4_ml_advanced.py](file:///c:/Users/Admin/OneDrive%20-%20VNU-HCMUS/Documents/FinalLABML/member4_ml_advanced.py)

| Mô hình | DEV F1-weighted | TEST F1-weighted |
|---------|----------------|-----------------|
| Random Forest | 0.8302 | 0.8407 |
| Gradient Boosting | 0.8284 | 0.8341 |
| **Voting Ensemble** | **0.8487** | **0.8581** |

**🏆 Mô hình tốt nhất:** Voting Ensemble (LR+SVM+RF) → **F1=0.8581, Acc=0.8678 trên test**  
**Error Analysis:** 863/6,527 (13.2%) sai. Chủ yếu nhầm Thù ghét↔Không tiêu cực (278+160 mẫu).

---

### 📝 Thành viên 5: QA / Report Writer — Lưu Thượng Hồng
**File:** [bao_cao_du_an.md](file:///C:/Users/Admin/.gemini/antigravity/brain/f7b1f469-07a1-48db-91ce-9771ae973ec0/artifacts/bao_cao_du_an.md)  
Báo cáo khoa học hoàn chỉnh gồm 9 phần, sẵn sàng copy vào file báo cáo cuối cùng.

---

## 📂 Cấu trúc file đã tạo

```
FinalLABML/
├── member1_data_engineer.py      ✅ Code tiền xử lý (đã chạy OK)
├── member2_eda.py                ✅ Code EDA (đã chạy OK)
├── member3_ml_traditional.py     ✅ 3 mô hình truyền thống (đã chạy OK)
├── member4_ml_advanced.py        ✅ 3 mô hình nâng cao (đã chạy OK)
├── train_clean.csv               ✅ 22,510 dòng
├── dev_clean.csv                 ✅ 2,633 dòng
├── test_clean.csv                ✅ 6,527 dòng
├── eda_output/                   ✅ 9 biểu đồ PNG
└── models/                       ✅ 6 mô hình .pkl + 2 vectorizer
```

> [!IMPORTANT]
> Tất cả code đã được chạy thành công và kết quả đã được xác minh. Để chạy lại, dùng: `$env:PYTHONIOENCODING='utf-8'; python <filename>.py`
