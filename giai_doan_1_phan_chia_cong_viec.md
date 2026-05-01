# 📋 GIAI ĐOẠN 1: PHÂN CHIA CÔNG VIỆC

## 📌 Tổng quan dự án

| Thông tin | Chi tiết |
|-----------|----------|
| **Đề tài** | Xây dựng hệ thống phân loại bình luận tiêu cực tiếng Việt |
| **Bài toán** | Phân loại văn bản (Text Classification) - 3 lớp |
| **Dataset** | 3 file CSV: `train.csv` (24,048), `dev.csv` (2,672), `test.csv` (6,680) |
| **Cột dữ liệu** | `free_text` (văn bản bình luận), `label_id` (0, 1, 2) |
| **Nhãn** | 0 = Không tiêu cực (~82%), 1 = Tiêu cực (~6.7%), 2 = Thù ghét/Rất tiêu cực (~10.6%) |
| **Đặc điểm** | Dữ liệu mất cân bằng nghiêm trọng, có null (2 dòng), có trùng lặp (1,356 dòng) |

## 📌 Yêu cầu từ giảng viên (ml-data.pdf)

1. ✅ Giới thiệu bài toán
2. ✅ Phương pháp và nguồn thu thập dữ liệu
3. ✅ Quy trình làm sạch và tiền xử lý (loại nhiễu, trùng lặp, missing, chuẩn hóa)
4. ✅ Đảm bảo chất lượng dữ liệu
5. ✅ Lưu trữ và quản lý dữ liệu
6. ✅ Phân tích khám phá dữ liệu (EDA) - thống kê mô tả, phân bố nhãn, mối quan hệ thuộc tính
7. ✅ EDA cho dữ liệu phi cấu trúc (thống kê độ dài, tần suất từ, tỉ lệ ký tự đặc biệt)
8. ✅ Thử nghiệm ≥ 3 thuật toán ML
9. ✅ Đánh giá mô hình (metrics + confusion matrix)
10. ✅ Đóng gói mô hình & triển khai web app

---

## 👥 PHÂN CHIA NHÂN SỰ VÀ TASK

---

### 🧑‍💻 Thành viên 1: **Data Engineer** — Trần Thanh Đạt (23120030)
> **Trách nhiệm chính:** Tiền xử lý dữ liệu & Pipeline xử lý văn bản tiếng Việt

**To-do list:**
- [ ] **Task 1.1:** Đọc và khám phá sơ bộ 3 file CSV (shape, dtypes, null check)
- [ ] **Task 1.2:** Xử lý dữ liệu thiếu (missing values) - 2 dòng null trong `free_text` ở train
- [ ] **Task 1.3:** Phát hiện và xử lý dữ liệu trùng lặp (1,356 duplicates trong train)
- [ ] **Task 1.4:** Chuẩn hóa văn bản tiếng Việt:
  - Chuyển lowercase
  - Xóa URL, email, số điện thoại
  - Xóa emoji, ký tự đặc biệt thừa
  - Xử lý teencode tiếng Việt (vd: "ko" → "không", "dc" → "được")
  - Xóa stopwords tiếng Việt
  - Xử lý khoảng trắng thừa
- [ ] **Task 1.5:** Kiểm tra tính nhất quán dữ liệu sau tiền xử lý
- [ ] **Task 1.6:** Lưu dataset đã xử lý ra file (train_clean.csv, dev_clean.csv, test_clean.csv)
- [ ] **Task 1.7:** Viết pipeline tiền xử lý dạng hàm có thể tái sử dụng

**Deliverables:** Code Python hoàn chỉnh + 3 file CSV đã làm sạch

---

### 📊 Thành viên 2: **Data Analyst** — Lê Minh Hải (23120041)
> **Trách nhiệm chính:** Phân tích khám phá dữ liệu (EDA) toàn diện

**To-do list:**
- [ ] **Task 2.1:** Thống kê mô tả (descriptive statistics):
  - Số lượng mẫu mỗi tập (train/dev/test)
  - Thống kê cơ bản về độ dài văn bản (mean, median, std, min, max, IQR)
- [ ] **Task 2.2:** Phân tích phân bố nhãn:
  - Biểu đồ bar chart phân bố 3 nhãn cho mỗi tập
  - Tính tỉ lệ % từng lớp, đánh giá mức độ mất cân bằng
  - So sánh phân bố nhãn giữa train/dev/test
- [ ] **Task 2.3:** EDA cho dữ liệu văn bản (phi cấu trúc):
  - Histogram phân bố độ dài câu theo từng nhãn
  - Word Cloud cho từng nhãn (0, 1, 2)
  - Top-20 từ phổ biến nhất theo từng nhãn (bar chart)
  - Thống kê tỉ lệ ký tự đặc biệt, emoji, viết hoa
- [ ] **Task 2.4:** Phân tích chất lượng dữ liệu:
  - Phát hiện outliers (văn bản quá ngắn/quá dài)
  - Thống kê dữ liệu trùng lặp, không hợp lệ
- [ ] **Task 2.5:** Phân tích N-gram (bigram, trigram) cho từng nhãn
- [ ] **Task 2.6:** Tổng hợp nhận xét & insight từ EDA

**Deliverables:** Code Python tạo biểu đồ + Markdown tổng hợp insight

---

### 🤖 Thành viên 3: **Machine Learning Engineer 1** — Phan Hoàng Quốc Huy (23120048)
> **Trách nhiệm chính:** Huấn luyện các mô hình ML truyền thống (Baseline)

**To-do list:**
- [ ] **Task 3.1:** Feature Engineering - Trích xuất đặc trưng văn bản:
  - TF-IDF Vectorizer (tuning ngram_range, max_features)
  - CountVectorizer (Bag of Words)
- [ ] **Task 3.2:** Huấn luyện Mô hình 1: **Logistic Regression**
  - Grid search/Random search hyperparameters (C, penalty, solver)
  - Đánh giá trên dev set
- [ ] **Task 3.3:** Huấn luyện Mô hình 2: **Naive Bayes (MultinomialNB)**
  - Tuning alpha (Laplace smoothing)
  - Đánh giá trên dev set
- [ ] **Task 3.4:** Huấn luyện Mô hình 3: **SVM (LinearSVC / SVC)**
  - Grid search hyperparameters (C, kernel)
  - Đánh giá trên dev set
- [ ] **Task 3.5:** Xử lý mất cân bằng dữ liệu:
  - Thử class_weight='balanced'
  - Thử SMOTE/oversampling
- [ ] **Task 3.6:** So sánh kết quả 3 mô hình trên dev set
- [ ] **Task 3.7:** Chọn mô hình tốt nhất, đánh giá trên test set

**Deliverables:** Code Python huấn luyện 3 mô hình + Bảng kết quả so sánh

---

### ⚙️ Thành viên 4: **Machine Learning Engineer 2** — Cao Tiến Thành (23120088)
> **Trách nhiệm chính:** Huấn luyện mô hình nâng cao & Ensemble + Đánh giá tổng hợp

**To-do list:**
- [ ] **Task 4.1:** Huấn luyện Mô hình 4: **Random Forest**
  - Tuning n_estimators, max_depth, min_samples_split
  - Đánh giá trên dev set
- [ ] **Task 4.2:** Huấn luyện Mô hình 5: **Gradient Boosting / XGBoost** (nếu có thể cài)
  - Tuning learning_rate, n_estimators, max_depth
  - Đánh giá trên dev set
- [ ] **Task 4.3:** Thử nghiệm **Voting Classifier / Stacking**
  - Kết hợp các mô hình tốt nhất từ TV3 và TV4
- [ ] **Task 4.4:** Đánh giá toàn diện mô hình tốt nhất:
  - Confusion Matrix (heatmap)
  - Classification Report (precision, recall, F1 cho từng lớp)
  - Macro/Weighted F1-score
  - ROC-AUC curve (nếu áp dụng được)
- [ ] **Task 4.5:** Phân tích lỗi (Error Analysis):
  - Xem các mẫu bị phân loại sai
  - Nhận xét nguyên nhân và hướng cải thiện
- [ ] **Task 4.6:** Lưu mô hình tốt nhất (pickle/joblib)

**Deliverables:** Code Python + Mô hình đã lưu + Báo cáo đánh giá

---

### 📝 Thành viên 5: **Quality Assurance / Report Writer** — Lưu Thượng Hồng (23122006)
> **Trách nhiệm chính:** Tổng hợp báo cáo khoa học + Kiểm tra chất lượng

**To-do list:**
- [ ] **Task 5.1:** Viết phần **Giới thiệu bài toán** (context Việt Nam, tính cấp thiết)
- [ ] **Task 5.2:** Viết phần **Phương pháp thu thập dữ liệu** (mô tả nguồn, quy trình, định dạng)
- [ ] **Task 5.3:** Viết phần **Quy trình tiền xử lý** (tổng hợp từ TV1, mô tả chi tiết từng bước)
- [ ] **Task 5.4:** Viết phần **EDA & Phân tích** (tổng hợp từ TV2, chèn biểu đồ, nhận xét)
- [ ] **Task 5.5:** Viết phần **Mô hình & Kết quả** (tổng hợp từ TV3 + TV4, bảng so sánh)
- [ ] **Task 5.6:** Viết phần **Đánh giá & Thảo luận** (phân tích kết quả, hạn chế, hướng phát triển)
- [ ] **Task 5.7:** Viết phần **Kết luận**
- [ ] **Task 5.8:** Review toàn bộ code của 4 thành viên (đảm bảo chạy được, comment đầy đủ)
- [ ] **Task 5.9:** Tạo bảng đóng góp (contribution table) cho 5 thành viên

**Deliverables:** File báo cáo Markdown hoàn chỉnh

---

## 📊 Ma trận trách nhiệm (RACI)

| Công việc | TV1 (DE) | TV2 (DA) | TV3 (MLE1) | TV4 (MLE2) | TV5 (QA) |
|-----------|----------|----------|------------|------------|----------|
| Tiền xử lý dữ liệu | **R** | C | I | I | A |
| EDA & Visualization | I | **R** | I | I | A |
| Feature Engineering | C | I | **R** | C | I |
| Mô hình truyền thống | I | I | **R** | C | A |
| Mô hình nâng cao | I | I | C | **R** | A |
| Đánh giá mô hình | I | I | C | **R** | A |
| Viết báo cáo | C | C | C | C | **R** |
| Review & QA | I | I | I | I | **R** |

> **R** = Responsible, **A** = Accountable, **C** = Consulted, **I** = Informed
