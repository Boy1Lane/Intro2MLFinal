# ============================================================================
# 📊 THÀNH VIÊN 2: DATA ANALYST - Lê Minh Hải (23120041)
# Nhiệm vụ: Phân tích khám phá dữ liệu (EDA)
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os
import warnings
warnings.filterwarnings('ignore')

# Cấu hình đồ thị
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')

# Tên nhãn
LABEL_NAMES = {0: 'Không tiêu cực', 1: 'Tiêu cực', 2: 'Thù ghét'}
COLORS = ['#2ecc71', '#e67e22', '#e74c3c']

# Thư mục output
OUTPUT_DIR = 'eda_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 1. ĐỌC DỮ LIỆU ĐÃ TIỀN XỬ LÝ
# ============================================================================
print("=" * 60)
print("📂 ĐỌC DỮ LIỆU")
print("=" * 60)

train = pd.read_csv('train_clean.csv')
dev = pd.read_csv('dev_clean.csv')
test = pd.read_csv('test_clean.csv')

for name, df in [('Train', train), ('Dev', dev), ('Test', test)]:
    print(f"  {name}: {df.shape[0]} samples, {df.shape[1]} columns")

# ============================================================================
# 2. THỐNG KÊ MÔ TẢ
# ============================================================================
print("\n" + "=" * 60)
print("📊 THỐNG KÊ MÔ TẢ")
print("=" * 60)

# Thêm cột độ dài và số từ
for df in [train, dev, test]:
    df['text_length'] = df['free_text_clean'].astype(str).str.len()
    df['word_count'] = df['free_text_clean'].astype(str).str.split().str.len()

print("\n--- Thống kê độ dài văn bản (train) ---")
print(train[['text_length', 'word_count']].describe().round(2))

# ============================================================================
# 3. BIỂU ĐỒ PHÂN BỐ NHÃN
# ============================================================================
print("\n📊 Tạo biểu đồ phân bố nhãn...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (name, df) in enumerate([('Train', train), ('Dev', dev), ('Test', test)]):
    counts = df['label_id'].value_counts().sort_index()
    bars = axes[idx].bar([LABEL_NAMES[i] for i in counts.index], counts.values, color=COLORS)
    axes[idx].set_title(f'Phân bố nhãn - {name} ({df.shape[0]} mẫu)', fontsize=13, fontweight='bold')
    axes[idx].set_ylabel('Số lượng')
    for bar, val in zip(bars, counts.values):
        pct = val / df.shape[0] * 100
        axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                      f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/label_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: label_distribution.png")

# ============================================================================
# 4. PHÂN BỐ ĐỘ DÀI VĂN BẢN THEO NHÃN
# ============================================================================
print("📊 Tạo biểu đồ phân bố độ dài văn bản...")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Histogram độ dài ký tự
for label in [0, 1, 2]:
    subset = train[train['label_id'] == label]['text_length']
    # Lọc outlier để biểu đồ đẹp hơn
    subset = subset[subset <= subset.quantile(0.95)]
    axes[0].hist(subset, bins=50, alpha=0.6, label=LABEL_NAMES[label], color=COLORS[label])
axes[0].set_title('Phân bố độ dài văn bản (ký tự) - Train', fontweight='bold')
axes[0].set_xlabel('Số ký tự')
axes[0].set_ylabel('Tần suất')
axes[0].legend()

# Histogram số từ
for label in [0, 1, 2]:
    subset = train[train['label_id'] == label]['word_count']
    subset = subset[subset <= subset.quantile(0.95)]
    axes[1].hist(subset, bins=50, alpha=0.6, label=LABEL_NAMES[label], color=COLORS[label])
axes[1].set_title('Phân bố số từ - Train', fontweight='bold')
axes[1].set_xlabel('Số từ')
axes[1].set_ylabel('Tần suất')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/text_length_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: text_length_distribution.png")

# ============================================================================
# 5. BOXPLOT ĐỘ DÀI VĂN BẢN
# ============================================================================
print("📊 Tạo boxplot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
train_filtered = train[train['text_length'] <= train['text_length'].quantile(0.95)]

sns.boxplot(data=train_filtered, x='label_id', y='text_length', palette=COLORS, ax=axes[0])
axes[0].set_xticklabels([LABEL_NAMES[i] for i in range(3)])
axes[0].set_title('Boxplot độ dài văn bản theo nhãn', fontweight='bold')
axes[0].set_xlabel('Nhãn')
axes[0].set_ylabel('Số ký tự')

sns.boxplot(data=train_filtered, x='label_id', y='word_count', palette=COLORS, ax=axes[1])
axes[1].set_xticklabels([LABEL_NAMES[i] for i in range(3)])
axes[1].set_title('Boxplot số từ theo nhãn', fontweight='bold')
axes[1].set_xlabel('Nhãn')
axes[1].set_ylabel('Số từ')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/boxplot_text_length.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: boxplot_text_length.png")

# ============================================================================
# 6. TOP-20 TỪ PHỔ BIẾN THEO NHÃN
# ============================================================================
print("📊 Tạo biểu đồ top-20 từ phổ biến...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for label in [0, 1, 2]:
    texts = train[train['label_id'] == label]['free_text_clean'].astype(str)
    all_words = ' '.join(texts).split()
    word_freq = Counter(all_words).most_common(20)
    words, counts = zip(*word_freq)
    
    axes[label].barh(range(len(words)), counts, color=COLORS[label], alpha=0.8)
    axes[label].set_yticks(range(len(words)))
    axes[label].set_yticklabels(words)
    axes[label].invert_yaxis()
    axes[label].set_title(f'Top-20 từ - {LABEL_NAMES[label]}', fontweight='bold')
    axes[label].set_xlabel('Tần suất')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/top20_words_by_label.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: top20_words_by_label.png")

# ============================================================================
# 7. PHÂN TÍCH N-GRAM (BIGRAM)
# ============================================================================
print("📊 Tạo biểu đồ bigram...")

def get_ngrams(texts, n=2, top_k=15):
    """Lấy top-k n-gram phổ biến nhất"""
    all_ngrams = []
    for text in texts:
        words = str(text).split()
        if len(words) >= n:
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            all_ngrams.extend(ngrams)
    return Counter(all_ngrams).most_common(top_k)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for label in [0, 1, 2]:
    texts = train[train['label_id'] == label]['free_text_clean']
    bigrams = get_ngrams(texts, n=2, top_k=15)
    if bigrams:
        words, counts = zip(*bigrams)
        axes[label].barh(range(len(words)), counts, color=COLORS[label], alpha=0.8)
        axes[label].set_yticks(range(len(words)))
        axes[label].set_yticklabels(words, fontsize=9)
        axes[label].invert_yaxis()
    axes[label].set_title(f'Top Bigrams - {LABEL_NAMES[label]}', fontweight='bold')
    axes[label].set_xlabel('Tần suất')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/bigram_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: bigram_analysis.png")

# ============================================================================
# 8. THỐNG KÊ KÝ TỰ ĐẶC BIỆT VÀ OUTLIER
# ============================================================================
print("\n📊 PHÂN TÍCH CHẤT LƯỢNG DỮ LIỆU")
print("-" * 40)

# Thống kê outlier (văn bản quá ngắn/dài)
very_short = train[train['word_count'] <= 2].shape[0]
very_long = train[train['word_count'] > 100].shape[0]
print(f"  Văn bản rất ngắn (<=2 từ): {very_short} ({very_short/len(train)*100:.1f}%)")
print(f"  Văn bản rất dài (>100 từ): {very_long} ({very_long/len(train)*100:.1f}%)")

# Thống kê mean text length theo nhãn
print("\n--- Thống kê trung bình theo nhãn ---")
stats_by_label = train.groupby('label_id').agg(
    count=('label_id', 'size'),
    avg_length=('text_length', 'mean'),
    avg_words=('word_count', 'mean'),
    median_words=('word_count', 'median')
).round(2)
stats_by_label.index = [LABEL_NAMES[i] for i in stats_by_label.index]
print(stats_by_label)

# ============================================================================
# 9. CORRELATION HEATMAP (text features)
# ============================================================================
print("\n📊 Tạo correlation heatmap...")

train['has_uppercase'] = train['free_text'].astype(str).apply(lambda x: int(any(c.isupper() for c in x)))
train['exclamation_count'] = train['free_text'].astype(str).str.count('!')
train['question_count'] = train['free_text'].astype(str).str.count(r'\?')

corr_cols = ['text_length', 'word_count', 'has_uppercase', 'exclamation_count', 'question_count', 'label_id']
corr_matrix = train[corr_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, fmt='.3f', ax=ax,
            xticklabels=['Độ dài', 'Số từ', 'Viết hoa', 'Dấu !', 'Dấu ?', 'Nhãn'],
            yticklabels=['Độ dài', 'Số từ', 'Viết hoa', 'Dấu !', 'Dấu ?', 'Nhãn'])
ax.set_title('Ma trận tương quan giữa các đặc trưng', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: correlation_heatmap.png")

# ============================================================================
# 10. SO SÁNH PHÂN BỐ TRAIN/DEV/TEST
# ============================================================================
print("📊 So sánh phân bố giữa train/dev/test...")

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(3)
width = 0.25

for idx, (name, df) in enumerate([('Train', train), ('Dev', dev), ('Test', test)]):
    pcts = [df[df['label_id']==l].shape[0] / df.shape[0] * 100 for l in [0, 1, 2]]
    bars = ax.bar(x + idx*width, pcts, width, label=name, alpha=0.85)

ax.set_xlabel('Nhãn')
ax.set_ylabel('Tỉ lệ (%)')
ax.set_title('So sánh phân bố nhãn giữa Train/Dev/Test', fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels([LABEL_NAMES[i] for i in range(3)])
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/train_dev_test_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: train_dev_test_comparison.png")

# ============================================================================
# TỔNG KẾT
# ============================================================================
print("\n" + "=" * 60)
print("🎯 TỔNG KẾT EDA")
print("=" * 60)
print(f"📁 Tất cả biểu đồ đã lưu trong thư mục: {OUTPUT_DIR}/")
print(f"   - label_distribution.png")
print(f"   - text_length_distribution.png")
print(f"   - boxplot_text_length.png")
print(f"   - top20_words_by_label.png")
print(f"   - bigram_analysis.png")
print(f"   - correlation_heatmap.png")
print(f"   - train_dev_test_comparison.png")
print(f"\n🔑 INSIGHT CHÍNH:")
print(f"   1. Dữ liệu mất cân bằng nghiêm trọng: label 0 chiếm ~82%")
print(f"   2. Độ dài văn bản trung bình ~48 ký tự, ~12 từ")
print(f"   3. Phân bố nhãn tương đồng giữa train/dev/test")
print(f"   4. Cần xử lý class imbalance khi huấn luyện mô hình")
print(f"\n🏁 HOÀN TẤT EDA!")
