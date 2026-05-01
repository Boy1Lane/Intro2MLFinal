# ============================================================================
# 🤖 THÀNH VIÊN 3: ML ENGINEER 1 - Phan Hoàng Quốc Huy (23120048)
# Nhiệm vụ: Huấn luyện mô hình ML truyền thống (Logistic Regression, NB, SVM)
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. ĐỌC DỮ LIỆU
# ============================================================================
print("=" * 60)
print("📂 ĐỌC DỮ LIỆU ĐÃ TIỀN XỬ LÝ")
print("=" * 60)

train = pd.read_csv('train_clean.csv')
dev = pd.read_csv('dev_clean.csv')
test = pd.read_csv('test_clean.csv')

X_train = train['free_text_clean'].astype(str)
y_train = train['label_id']
X_dev = dev['free_text_clean'].astype(str)
y_dev = dev['label_id']
X_test = test['free_text_clean'].astype(str)
y_test = test['label_id']

print(f"Train: {len(X_train)}, Dev: {len(X_dev)}, Test: {len(X_test)}")

LABEL_NAMES = ['Không tiêu cực', 'Tiêu cực', 'Thù ghét']

# ============================================================================
# 2. FEATURE ENGINEERING - TF-IDF
# ============================================================================
print("\n" + "=" * 60)
print("🔧 FEATURE ENGINEERING: TF-IDF")
print("=" * 60)

# TF-IDF với unigram + bigram
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_dev_tfidf = tfidf.transform(X_dev)
X_test_tfidf = tfidf.transform(X_test)

print(f"TF-IDF shape: {X_train_tfidf.shape}")
print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

# Bag of Words cho Naive Bayes
count_vec = CountVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, max_df=0.95)
X_train_bow = count_vec.fit_transform(X_train)
X_dev_bow = count_vec.transform(X_dev)
X_test_bow = count_vec.transform(X_test)
print(f"BoW shape: {X_train_bow.shape}")

# ============================================================================
# 3. MÔ HÌNH 1: LOGISTIC REGRESSION
# ============================================================================
print("\n" + "=" * 60)
print("🤖 MÔ HÌNH 1: LOGISTIC REGRESSION")
print("=" * 60)

# Grid Search trên dev set
print("🔍 Grid Search hyperparameters...")
lr_params = {
    'C': [0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'class_weight': [None, 'balanced'],
    'max_iter': [1000]
}

lr_grid = GridSearchCV(
    LogisticRegression(random_state=42),
    lr_params, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=0
)
lr_grid.fit(X_train_tfidf, y_train)
print(f"Best params: {lr_grid.best_params_}")
print(f"Best CV F1-weighted: {lr_grid.best_score_:.4f}")

lr_best = lr_grid.best_estimator_
y_dev_pred_lr = lr_best.predict(X_dev_tfidf)
print(f"\n📊 Kết quả trên DEV set:")
print(classification_report(y_dev, y_dev_pred_lr, target_names=LABEL_NAMES))

lr_dev_f1 = f1_score(y_dev, y_dev_pred_lr, average='weighted')
lr_dev_acc = accuracy_score(y_dev, y_dev_pred_lr)

# ============================================================================
# 4. MÔ HÌNH 2: NAIVE BAYES
# ============================================================================
print("\n" + "=" * 60)
print("🤖 MÔ HÌNH 2: MULTINOMIAL NAIVE BAYES")
print("=" * 60)

nb_params = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]}
nb_grid = GridSearchCV(
    MultinomialNB(), nb_params, cv=3, scoring='f1_weighted', n_jobs=-1
)
nb_grid.fit(X_train_bow, y_train)
print(f"Best params: {nb_grid.best_params_}")
print(f"Best CV F1-weighted: {nb_grid.best_score_:.4f}")

nb_best = nb_grid.best_estimator_
y_dev_pred_nb = nb_best.predict(X_dev_bow)
print(f"\n📊 Kết quả trên DEV set:")
print(classification_report(y_dev, y_dev_pred_nb, target_names=LABEL_NAMES))

nb_dev_f1 = f1_score(y_dev, y_dev_pred_nb, average='weighted')
nb_dev_acc = accuracy_score(y_dev, y_dev_pred_nb)

# ============================================================================
# 5. MÔ HÌNH 3: SVM (LinearSVC)
# ============================================================================
print("\n" + "=" * 60)
print("🤖 MÔ HÌNH 3: LINEAR SVM")
print("=" * 60)

svm_params = {
    'C': [0.1, 1, 10],
    'class_weight': [None, 'balanced'],
    'max_iter': [5000]
}
svm_grid = GridSearchCV(
    LinearSVC(random_state=42, dual=True),
    svm_params, cv=3, scoring='f1_weighted', n_jobs=-1
)
svm_grid.fit(X_train_tfidf, y_train)
print(f"Best params: {svm_grid.best_params_}")
print(f"Best CV F1-weighted: {svm_grid.best_score_:.4f}")

svm_best = svm_grid.best_estimator_
y_dev_pred_svm = svm_best.predict(X_dev_tfidf)
print(f"\n📊 Kết quả trên DEV set:")
print(classification_report(y_dev, y_dev_pred_svm, target_names=LABEL_NAMES))

svm_dev_f1 = f1_score(y_dev, y_dev_pred_svm, average='weighted')
svm_dev_acc = accuracy_score(y_dev, y_dev_pred_svm)

# ============================================================================
# 6. SO SÁNH KẾT QUẢ 3 MÔ HÌNH
# ============================================================================
print("\n" + "=" * 60)
print("📊 SO SÁNH 3 MÔ HÌNH TRÊN DEV SET")
print("=" * 60)

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Multinomial NB', 'Linear SVM'],
    'Accuracy': [lr_dev_acc, nb_dev_acc, svm_dev_acc],
    'F1-weighted': [lr_dev_f1, nb_dev_f1, svm_dev_f1]
}).round(4)
print(results.to_string(index=False))

# Chọn mô hình tốt nhất
best_idx = results['F1-weighted'].idxmax()
best_model_name = results.loc[best_idx, 'Model']
print(f"\n🏆 Mô hình tốt nhất (DEV): {best_model_name}")

# ============================================================================
# 7. ĐÁNH GIÁ TRÊN TEST SET
# ============================================================================
print("\n" + "=" * 60)
print(f"📊 ĐÁNH GIÁ TRÊN TEST SET - {best_model_name}")
print("=" * 60)

best_models = {
    'Logistic Regression': (lr_best, X_test_tfidf),
    'Multinomial NB': (nb_best, X_test_bow),
    'Linear SVM': (svm_best, X_test_tfidf)
}

for name, (model, X_ts) in best_models.items():
    y_test_pred = model.predict(X_ts)
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    acc = accuracy_score(y_test, y_test_pred)
    print(f"\n--- {name} (TEST) ---")
    print(f"Accuracy: {acc:.4f}, F1-weighted: {f1:.4f}")
    print(classification_report(y_test, y_test_pred, target_names=LABEL_NAMES))

# ============================================================================
# 8. LƯU MÔ HÌNH VÀ VECTORIZER
# ============================================================================
os.makedirs('models', exist_ok=True)
joblib.dump(lr_best, 'models/logistic_regression.pkl')
joblib.dump(nb_best, 'models/naive_bayes.pkl')
joblib.dump(svm_best, 'models/linear_svm.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
joblib.dump(count_vec, 'models/count_vectorizer.pkl')
print("\n✅ Đã lưu tất cả mô hình vào thư mục models/")

# Lưu kết quả so sánh
results.to_csv('models/baseline_results.csv', index=False)
print("✅ Đã lưu bảng so sánh: models/baseline_results.csv")
print("\n🏁 HOÀN TẤT HUẤN LUYỆN MÔ HÌNH TRUYỀN THỐNG!")
