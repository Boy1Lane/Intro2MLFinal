# ============================================================================
# ⚙️ THÀNH VIÊN 4: ML ENGINEER 2 - Cao Tiến Thành (23120088)
# Nhiệm vụ: Mô hình nâng cao (Random Forest, Gradient Boosting, Ensemble)
#            + Đánh giá tổng hợp + Error Analysis
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             accuracy_score, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

LABEL_NAMES = ['Không tiêu cực', 'Tiêu cực', 'Thù ghét']
os.makedirs('models', exist_ok=True)
os.makedirs('eda_output', exist_ok=True)

# ============================================================================
# 1. ĐỌC DỮ LIỆU
# ============================================================================
print("=" * 60)
print("📂 ĐỌC DỮ LIỆU")
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

# TF-IDF (dùng chung với member3)
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_dev_tfidf = tfidf.transform(X_dev)
X_test_tfidf = tfidf.transform(X_test)
print(f"TF-IDF features: {X_train_tfidf.shape[1]}")

# ============================================================================
# 2. MÔ HÌNH 4: RANDOM FOREST
# ============================================================================
print("\n" + "=" * 60)
print("🌲 MÔ HÌNH 4: RANDOM FOREST")
print("=" * 60)

rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [50, 100, None],
    'class_weight': ['balanced'],
    'min_samples_split': [2, 5]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_params, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=0
)
rf_grid.fit(X_train_tfidf, y_train)
print(f"Best params: {rf_grid.best_params_}")
print(f"Best CV F1-weighted: {rf_grid.best_score_:.4f}")

rf_best = rf_grid.best_estimator_
y_dev_pred_rf = rf_best.predict(X_dev_tfidf)
print(f"\n📊 Random Forest - DEV set:")
print(classification_report(y_dev, y_dev_pred_rf, target_names=LABEL_NAMES))
rf_dev_f1 = f1_score(y_dev, y_dev_pred_rf, average='weighted')

# ============================================================================
# 3. MÔ HÌNH 5: GRADIENT BOOSTING
# ============================================================================
print("\n" + "=" * 60)
print("🚀 MÔ HÌNH 5: GRADIENT BOOSTING")
print("=" * 60)

# Dùng subset features cho GB (quá chậm với 50k features)
tfidf_small = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True)
X_train_small = tfidf_small.fit_transform(X_train)
X_dev_small = tfidf_small.transform(X_dev)
X_test_small = tfidf_small.transform(X_test)

gb_model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=5,
    random_state=42, verbose=0
)
gb_model.fit(X_train_small, y_train)

y_dev_pred_gb = gb_model.predict(X_dev_small)
print(f"\n📊 Gradient Boosting - DEV set:")
print(classification_report(y_dev, y_dev_pred_gb, target_names=LABEL_NAMES))
gb_dev_f1 = f1_score(y_dev, y_dev_pred_gb, average='weighted')

# ============================================================================
# 4. MÔ HÌNH 6: VOTING CLASSIFIER (ENSEMBLE)
# ============================================================================
print("\n" + "=" * 60)
print("🗳️ MÔ HÌNH 6: VOTING CLASSIFIER (ENSEMBLE)")
print("=" * 60)

# Load mô hình tốt nhất từ member3
lr_model = LogisticRegression(C=10, class_weight='balanced', max_iter=1000, random_state=42)
svm_model = CalibratedClassifierCV(LinearSVC(C=1, class_weight='balanced', max_iter=5000, random_state=42, dual=True))

voting = VotingClassifier(
    estimators=[
        ('lr', lr_model),
        ('svm', svm_model),
        ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=100, random_state=42, n_jobs=-1))
    ],
    voting='soft',
    n_jobs=-1
)
voting.fit(X_train_tfidf, y_train)

y_dev_pred_voting = voting.predict(X_dev_tfidf)
print(f"\n📊 Voting Classifier - DEV set:")
print(classification_report(y_dev, y_dev_pred_voting, target_names=LABEL_NAMES))
voting_dev_f1 = f1_score(y_dev, y_dev_pred_voting, average='weighted')

# ============================================================================
# 5. BẢNG SO SÁNH TẤT CẢ MÔ HÌNH
# ============================================================================
print("\n" + "=" * 60)
print("📊 BẢNG SO SÁNH TẤT CẢ MÔ HÌNH (DEV SET)")
print("=" * 60)

# Load kết quả baseline từ member3
baseline_results = pd.read_csv('models/baseline_results.csv')

# Thêm kết quả mới
all_results = pd.concat([
    baseline_results,
    pd.DataFrame({
        'Model': ['Random Forest', 'Gradient Boosting', 'Voting Ensemble'],
        'Accuracy': [
            accuracy_score(y_dev, y_dev_pred_rf),
            accuracy_score(y_dev, y_dev_pred_gb),
            accuracy_score(y_dev, y_dev_pred_voting)
        ],
        'F1-weighted': [rf_dev_f1, gb_dev_f1, voting_dev_f1]
    })
], ignore_index=True).round(4)

all_results = all_results.sort_values('F1-weighted', ascending=False)
print(all_results.to_string(index=False))

best_model_name = all_results.iloc[0]['Model']
print(f"\n🏆 Mô hình tốt nhất: {best_model_name} (F1={all_results.iloc[0]['F1-weighted']:.4f})")

# ============================================================================
# 6. ĐÁNH GIÁ TOÀN DIỆN TRÊN TEST SET
# ============================================================================
print("\n" + "=" * 60)
print("📊 ĐÁNH GIÁ TRÊN TEST SET - TẤT CẢ MÔ HÌNH")
print("=" * 60)

test_models = {
    'Random Forest': (rf_best, X_test_tfidf),
    'Gradient Boosting': (gb_model, X_test_small),
    'Voting Ensemble': (voting, X_test_tfidf)
}

test_results = []
for name, (model, X_ts) in test_models.items():
    y_pred = model.predict(X_ts)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    test_results.append({'Model': name, 'Accuracy': round(acc, 4), 'F1-weighted': round(f1, 4)})
    print(f"\n--- {name} (TEST) ---")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

test_results_df = pd.DataFrame(test_results)
print("\n📊 Bảng tổng hợp TEST:")
print(test_results_df.to_string(index=False))

# ============================================================================
# 7. CONFUSION MATRIX (cho mô hình tốt nhất)
# ============================================================================
print("\n📊 Tạo Confusion Matrix...")

# Dùng Voting Ensemble cho confusion matrix (thường là tốt nhất)
y_test_pred_best = voting.predict(X_test_tfidf)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Confusion matrix raw
cm = confusion_matrix(y_test, y_test_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
axes[0].set_title('Confusion Matrix (Voting Ensemble)', fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Normalized confusion matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', ax=axes[1],
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
axes[1].set_title('Normalized Confusion Matrix', fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('eda_output/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: eda_output/confusion_matrix.png")

# ============================================================================
# 8. BIỂU ĐỒ SO SÁNH MÔ HÌNH
# ============================================================================
print("📊 Tạo biểu đồ so sánh mô hình...")

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(all_results))
width = 0.35

bars1 = ax.bar(x - width/2, all_results['Accuracy'], width, label='Accuracy', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, all_results['F1-weighted'], width, label='F1-weighted', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Mô hình')
ax.set_ylabel('Score')
ax.set_title('So sánh hiệu suất tất cả mô hình (DEV set)', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(all_results['Model'], rotation=15, ha='right')
ax.legend()
ax.set_ylim(0, 1.05)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('eda_output/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: eda_output/model_comparison.png")

# ============================================================================
# 9. ERROR ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("🔍 ERROR ANALYSIS")
print("=" * 60)

# Tìm các mẫu bị phân loại sai
test_copy = test.copy()
test_copy['predicted'] = y_test_pred_best
test_copy['correct'] = test_copy['label_id'] == test_copy['predicted']

wrong = test_copy[~test_copy['correct']]
print(f"Tổng mẫu sai: {len(wrong)} / {len(test_copy)} ({len(wrong)/len(test_copy)*100:.1f}%)")

print("\n--- Ma trận nhầm lẫn chi tiết ---")
for true_label in [0, 1, 2]:
    for pred_label in [0, 1, 2]:
        if true_label != pred_label:
            count = wrong[(wrong['label_id']==true_label) & (wrong['predicted']==pred_label)].shape[0]
            if count > 0:
                print(f"  {LABEL_NAMES[true_label]} → {LABEL_NAMES[pred_label]}: {count} mẫu")

print("\n--- Ví dụ mẫu bị phân loại sai ---")
for true_label in [1, 2]:
    wrong_subset = wrong[wrong['label_id'] == true_label].head(3)
    print(f"\n  Nhãn thực: {LABEL_NAMES[true_label]}")
    for _, row in wrong_subset.iterrows():
        text = str(row['free_text_clean'])[:80]
        print(f"    → Dự đoán: {LABEL_NAMES[row['predicted']]} | Text: {text}...")

# ============================================================================
# 10. LƯU MÔ HÌNH
# ============================================================================
joblib.dump(rf_best, 'models/random_forest.pkl')
joblib.dump(gb_model, 'models/gradient_boosting.pkl')
joblib.dump(voting, 'models/voting_ensemble.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
all_results.to_csv('models/all_model_results.csv', index=False)

print("\n✅ Đã lưu tất cả mô hình và kết quả!")
print("🏁 HOÀN TẤT MÔ HÌNH NÂNG CAO & ĐÁNH GIÁ!")
