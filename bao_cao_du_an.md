# BÁO CÁO DỰ ÁN: XÂY DỰNG HỆ THỐNG PHÂN LOẠI BÌNH LUẬN TIÊU CỰC TIẾNG VIỆT

> **Môn học:** Nhập môn Học máy  
> **Nhóm thực hiện:** Trần Thanh Đạt, Lê Minh Hải, Phan Hoàng Quốc Huy, Cao Tiến Thành, Lưu Thượng Hồng

---

## 1. Giới thiệu bài toán

### 1.1. Bối cảnh và tính cấp thiết

Sự bùng nổ của mạng xã hội và thương mại điện tử tại Việt Nam tạo ra lượng bình luận khổng lồ mỗi ngày. Ngôn ngữ mạng tiếng Việt đặc thù với nhiều **từ lóng, viết tắt (teencode)** và biến thể ngôn ngữ khiến việc kiểm duyệt thủ công trở nên bất khả thi.

Dự án này xây dựng hệ thống **phân loại tự động bình luận tiếng Việt** thành 3 nhóm:
- **Nhãn 0 – Không tiêu cực:** Bình luận trung tính hoặc tích cực
- **Nhãn 1 – Tiêu cực:** Bình luận có nội dung tiêu cực, phàn nàn
- **Nhãn 2 – Thù ghét:** Bình luận có nội dung thù ghét, xúc phạm, độc hại

### 1.2. Mục tiêu

- Thu thập, tiền xử lý và phân tích dữ liệu bình luận tiếng Việt
- Thử nghiệm và so sánh **6 thuật toán học máy** (vượt yêu cầu tối thiểu 3)
- Đánh giá mô hình qua các độ đo và ma trận nhầm lẫn
- Chọn mô hình tối ưu nhất phục vụ triển khai thực tế

---

## 2. Dữ liệu

### 2.1. Nguồn dữ liệu

Dataset dựa trên tập dữ liệu có sẵn về bình luận tiếng Việt trên mạng xã hội, đã được gán nhãn sẵn. Dữ liệu lưu trữ dạng CSV với 2 cột:
- `free_text`: Nội dung bình luận (văn bản tiếng Việt)
- `label_id`: Nhãn phân loại (0, 1, 2)

### 2.2. Thống kê tổng quan

| Tập dữ liệu | Số mẫu | Nhãn 0 (%) | Nhãn 1 (%) | Nhãn 2 (%) | Null | Trùng lặp |
|-------------|---------|------------|------------|------------|------|-----------|
| **Train** | 24,048 | 19,886 (82.7%) | 1,606 (6.7%) | 2,556 (10.6%) | 2 | 1,356 |
| **Dev** | 2,672 | 2,190 (81.9%) | 212 (7.9%) | 270 (10.1%) | 0 | 21 |
| **Test** | 6,680 | 5,548 (83.1%) | 444 (6.6%) | 688 (10.3%) | 0 | 98 |

> **Nhận xét:** Dữ liệu mất cân bằng nghiêm trọng – nhãn 0 chiếm ~82%, trong khi nhãn 1 chỉ ~6.7%.

### 2.3. Tổ chức thư mục

```
FinalLABML/
├── train.csv               # Dữ liệu huấn luyện gốc
├── dev.csv                 # Dữ liệu validation gốc  
├── test.csv                # Dữ liệu kiểm tra gốc
├── train_clean.csv         # Sau tiền xử lý
├── dev_clean.csv           # Sau tiền xử lý
├── test_clean.csv          # Sau tiền xử lý
├── member1_data_engineer.py   # Code tiền xử lý
├── member2_eda.py             # Code EDA
├── member3_ml_traditional.py  # Mô hình truyền thống
├── member4_ml_advanced.py     # Mô hình nâng cao
├── models/                    # Thư mục lưu mô hình
└── eda_output/                # Biểu đồ EDA
```

---

## 3. Quy trình tiền xử lý dữ liệu

### 3.1. Xử lý dữ liệu thiếu và trùng lặp

| Bước | Mô tả | Kết quả |
|------|--------|---------|
| Xóa null | Loại bỏ dòng có `free_text` rỗng | 2 dòng (train) |
| Xóa trùng lặp | Loại bỏ dòng hoàn toàn giống nhau | 1,356 (train), 21 (dev), 98 (test) |

### 3.2. Pipeline chuẩn hóa văn bản

1. **Chuyển lowercase** – Đồng nhất chữ hoa/thường
2. **Xóa HTML tags** – Loại bỏ các thẻ HTML lẫn trong text
3. **Xóa URL** – Loại link web không có ý nghĩa phân loại
4. **Xóa email, số điện thoại** – Thông tin cá nhân
5. **Xóa emoji** – Ký tự Unicode đặc biệt
6. **Chuẩn hóa ký tự lặp** – `"đẹpppppp" → "đẹpp"`, `"hahaha" → "haha"`
7. **Thay thế teencode** – Từ điển 80+ teencode phổ biến (`"ko" → "không"`, `"dc" → "được"`)
8. **Xóa ký tự đặc biệt** – Giữ lại chữ cái tiếng Việt, số, khoảng trắng
9. **Chuẩn hóa khoảng trắng** – Xóa khoảng trắng thừa

### 3.3. Kết quả sau tiền xử lý

| Tập | Trước | Sau | Đã xóa | Tỉ lệ |
|-----|-------|-----|--------|--------|
| Train | 24,048 | 22,510 | 1,538 | -6.4% |
| Dev | 2,672 | 2,633 | 39 | -1.5% |
| Test | 6,680 | 6,527 | 153 | -2.3% |

### 3.4. Kiểm tra chất lượng dữ liệu

- ✅ Không còn giá trị null
- ✅ Tất cả nhãn hợp lệ (0, 1, 2)
- ✅ Không còn text rỗng
- ✅ Kiểm tra bằng cả phương pháp tự động (assert) và kiểm tra thủ công (sampling)

---

## 4. Phân tích khám phá dữ liệu (EDA)

### 4.1. Thống kê mô tả

| Thống kê | Độ dài ký tự | Số từ |
|----------|-------------|-------|
| Mean | 48.15 | 11.70 |
| Std | 84.16 | 19.42 |
| Min | 1 | 1 |
| Q1 (25%) | 19 | 5 |
| Median (50%) | 33 | 8 |
| Q3 (75%) | 56 | 14 |
| Max | 7,084 | 1,550 |

### 4.2. Phân bố nhãn

Dữ liệu mất cân bằng nghiêm trọng với tỉ lệ Nhãn 0 : Nhãn 1 : Nhãn 2 ≈ **12 : 1 : 1.6**. Phân bố tương đồng giữa train/dev/test, chứng tỏ dữ liệu được chia stratified hợp lý.

### 4.3. Đặc điểm theo nhãn

| Nhãn | Số mẫu | Độ dài TB (ký tự) | Số từ TB | Median từ |
|------|---------|-------------------|----------|-----------|
| Không tiêu cực | 18,473 | 43.74 | 10.59 | 8 |
| Tiêu cực | 1,542 | 43.96 | 10.89 | 8 |
| Thù ghét | 2,495 | **83.37** | **20.42** | **15** |

> **Insight quan trọng:** Bình luận thù ghét (nhãn 2) có xu hướng **dài hơn gấp đôi** so với 2 nhãn còn lại. Đây là đặc trưng có ý nghĩa giúp phân loại.

### 4.4. Phân tích chất lượng

- **Outlier:** 6.5% văn bản rất ngắn (≤2 từ), 0.1% rất dài (>100 từ)
- **Tương quan:** Độ dài văn bản có tương quan dương nhẹ với nhãn (bình luận thù ghét thường dài hơn)
- **Đặc trưng phân biệt:** Dấu chấm than (!), viết hoa có tương quan với nội dung tiêu cực/thù ghét

---

## 5. Mô hình và thực nghiệm

### 5.1. Feature Engineering

- **TF-IDF Vectorizer:** `max_features=50,000`, `ngram_range=(1,2)`, `sublinear_tf=True`
- **CountVectorizer (BoW):** Dùng cho Naive Bayes, cùng tham số

### 5.2. Các mô hình đã thử nghiệm

| # | Mô hình | Loại | Hyperparameters tối ưu |
|---|---------|------|----------------------|
| 1 | Logistic Regression | Truyền thống | C=10, class_weight='balanced' |
| 2 | Multinomial Naive Bayes | Truyền thống | alpha=1.0 |
| 3 | Linear SVM | Truyền thống | C=1, class_weight='balanced' |
| 4 | Random Forest | Ensemble | n_estimators=200, max_depth=50, class_weight='balanced' |
| 5 | Gradient Boosting | Ensemble | n_estimators=200, learning_rate=0.1, max_depth=5 |
| 6 | Voting Ensemble | Meta | Soft voting (LR + SVM + RF) |

> Tất cả hyperparameters được tối ưu bằng **GridSearchCV** với 3-fold cross-validation.

### 5.3. Xử lý mất cân bằng dữ liệu

- **class_weight='balanced'**: Áp dụng cho LR, SVM, RF – tự điều chỉnh trọng số nghịch đảo tỉ lệ nhãn
- So sánh kết quả với và không có `class_weight` → **Có `balanced` luôn cho F1 tốt hơn** trên nhãn thiểu số

---

## 6. Kết quả thực nghiệm

### 6.1. Kết quả trên DEV set

| Mô hình | Accuracy | F1-weighted | Ranking |
|---------|----------|-------------|---------|
| **Voting Ensemble** | **0.8602** | **0.8487** | 🥇 |
| Linear SVM | 0.8507 | 0.8459 | 🥈 |
| Logistic Regression | 0.8420 | 0.8449 | 🥉 |
| Random Forest | 0.8386 | 0.8302 | 4 |
| Gradient Boosting | 0.8572 | 0.8284 | 5 |
| Multinomial NB | 0.8409 | 0.8232 | 6 |

### 6.2. Kết quả trên TEST set (mô hình tốt nhất)

**Voting Ensemble – Classification Report (TEST):**

| Lớp | Precision | Recall | F1-Score | Support |
|-----|-----------|--------|----------|---------|
| Không tiêu cực | 0.91 | 0.96 | 0.93 | 5,406 |
| Tiêu cực | 0.53 | 0.32 | 0.40 | 440 |
| Thù ghét | 0.59 | 0.51 | 0.55 | 681 |
| **Weighted Avg** | **0.85** | **0.87** | **0.86** | **6,527** |

- **Accuracy:** 0.8678
- **F1-weighted:** 0.8581
- **Macro F1:** 0.63

### 6.3. So sánh trên TEST set (tất cả mô hình)

| Mô hình | Test Accuracy | Test F1-weighted |
|---------|--------------|-----------------|
| **Voting Ensemble** | **0.8678** | **0.8581** |
| Linear SVM | 0.8630 | 0.8585 |
| Logistic Regression | 0.8506 | 0.8535 |
| Gradient Boosting | 0.8624 | 0.8341 |
| Random Forest | 0.8488 | 0.8407 |
| Multinomial NB | 0.8526 | 0.8366 |

---

## 7. Đánh giá và Thảo luận

### 7.1. Phân tích kết quả

- **Voting Ensemble** đạt hiệu suất tốt nhất nhờ kết hợp ưu điểm của LR, SVM và RF
- **Linear SVM** là mô hình đơn lẻ tốt nhất, cho thấy SVM phù hợp với bài toán phân loại văn bản
- **Multinomial NB** kém nhất do giả định independence không phù hợp với dữ liệu ngôn ngữ

### 7.2. Phân tích lỗi (Error Analysis)

**Tổng mẫu sai:** 863 / 6,527 (13.2%)

| Nhầm lẫn | Số mẫu | Giải thích |
|-----------|--------|-----------|
| Thù ghét → Không tiêu cực | 278 | Bình luận thù ghét dùng ngôn ngữ gián tiếp, châm biếm |
| Tiêu cực → Không tiêu cực | 216 | Bình luận tiêu cực nhẹ, ranh giới mờ |
| Không tiêu cực → Thù ghét | 160 | Văn bản chứa từ nhạy cảm nhưng không mang ý nghĩa tiêu cực |
| Tiêu cực → Thù ghét | 82 | Ranh giới giữa tiêu cực và thù ghét không rõ ràng |

### 7.3. Hạn chế

1. **Mất cân bằng dữ liệu** vẫn là thách thức lớn – F1 cho nhãn 1 (Tiêu cực) chỉ đạt 0.40
2. **Teencode dictionary** chưa bao phủ hết biến thể ngôn ngữ mạng
3. **Feature engineering** mới chỉ dùng TF-IDF, chưa khai thác word embeddings (Word2Vec, PhoBERT)
4. Chưa thử **deep learning** (LSTM, Transformer)

### 7.4. Hướng phát triển

1. Sử dụng **PhoBERT** – mô hình ngôn ngữ tiếng Việt pre-trained
2. Áp dụng **data augmentation** cho nhãn thiểu số
3. Thử **SMOTE** hoặc **focal loss** để xử lý imbalance tốt hơn
4. Triển khai web app demo bằng Flask/Streamlit

---

## 8. Kết luận

Dự án đã hoàn thành các mục tiêu đề ra:

- ✅ Xây dựng **pipeline tiền xử lý** hoàn chỉnh cho văn bản tiếng Việt
- ✅ Thực hiện **EDA toàn diện** với 7+ biểu đồ phân tích
- ✅ Thử nghiệm **6 mô hình** ML (vượt yêu cầu tối thiểu 3)
- ✅ Đánh giá bằng **confusion matrix**, classification report, F1-score
- ✅ Mô hình tốt nhất (**Voting Ensemble**) đạt **F1-weighted = 0.8581** trên test set

---

## 9. Bảng đóng góp

| Thành viên | MSSV | Vai trò | Công việc chính | Đóng góp |
|-----------|------|---------|-----------------|----------|
| Trần Thanh Đạt | 23120030 | Data Engineer | Tiền xử lý, pipeline | 20% |
| Lê Minh Hải | 23120041 | Data Analyst | EDA, visualization | 20% |
| Phan Hoàng Quốc Huy | 23120048 | ML Engineer 1 | LR, NB, SVM | 20% |
| Cao Tiến Thành | 23120088 | ML Engineer 2 | RF, GB, Ensemble, Error Analysis | 20% |
| Lưu Thượng Hồng | 23122006 | QA / Report Writer | Báo cáo, review, QA | 20% |
