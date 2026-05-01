# ============================================================================
# 🧑‍💻 THÀNH VIÊN 1: DATA ENGINEER - Trần Thanh Đạt (23120030)
# Nhiệm vụ: Tiền xử lý dữ liệu & Pipeline xử lý văn bản tiếng Việt
# ============================================================================

import pandas as pd
import numpy as np
import re
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PHẦN 1: ĐỌC VÀ KHÁM PHÁ SƠ BỘ DỮ LIỆU
# ============================================================================

def load_datasets(data_dir='.'):
    """
    Đọc 3 file CSV: train.csv, dev.csv, test.csv
    Returns: dict chứa 3 DataFrame
    """
    datasets = {}
    for name in ['train', 'dev', 'test']:
        filepath = os.path.join(data_dir, f'{name}.csv')
        df = pd.read_csv(filepath)
        datasets[name] = df
        print(f"📂 {name}.csv: {df.shape[0]} dòng, {df.shape[1]} cột")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Dtypes: {df.dtypes.to_dict()}")
        print(f"   Null values: {df.isnull().sum().to_dict()}")
        print(f"   Duplicates: {df.duplicated().sum()}")
        print(f"   Label distribution: {df['label_id'].value_counts().sort_index().to_dict()}")
        print()
    return datasets


# ============================================================================
# PHẦN 2: TỪĐIỂN TEENCODE TIẾNG VIỆT
# ============================================================================

# Từ điển teencode phổ biến trong tiếng Việt mạng
TEENCODE_DICT = {
    # Viết tắt phổ biến
    'ko': 'không', 'k': 'không', 'kh': 'không', 'khg': 'không',
    'kp': 'không phải', 'kq': 'không quan',
    'dc': 'được', 'đc': 'được', 'dk': 'được', 'đk': 'được',
    'nc': 'nước', 'ng': 'người', 'ns': 'nói',
    'mk': 'mình', 'mn': 'mọi người', 'mng': 'mọi người',
    'bn': 'bạn', 'b': 'bạn', 'bro': 'bạn',
    'ib': 'nhắn tin', 'rep': 'trả lời',
    'vs': 'với', 'v': 'với', 'voi': 'với',
    'r': 'rồi', 'rui': 'rồi', 'rii': 'rồi',
    'ntn': 'như thế nào', 'ntn': 'như thế nào',
    'j': 'gì', 'ji': 'gì', 'z': 'gì', 'gi': 'gì',
    'a': 'anh', 'e': 'em', 'c': 'chị',
    'đi': 'đi', 'qua': 'qua',
    'nc': 'nước', 'trc': 'trước', 'tg': 'thời gian',
    'bt': 'bình thường', 'bth': 'bình thường',
    'vl': 'vãi', 'vkl': 'vãi',
    'nch': 'nói chuyện', 'nt': 'nhắn tin',
    'hk': 'không', 'hem': 'không',
    'bi': 'bị', 'bik': 'biết',
    'ck': 'chồng', 'vk': 'vợ',
    'tks': 'thanks', 'thanks': 'cảm ơn', 'thks': 'cảm ơn',
    'ok': 'được', 'okie': 'được', 'oke': 'được',
    'plz': 'làm ơn', 'pls': 'làm ơn',
    'sr': 'xin lỗi', 'sorry': 'xin lỗi',
    'lun': 'luôn', 'lm': 'làm',
    'đag': 'đang', 'dg': 'đang',
    'trg': 'trong', 'trog': 'trong',
    'cx': 'cũng', 'cg': 'cũng',
    'đb': 'đặc biệt', 'cb': 'chuẩn bị',
    'h': 'giờ', 'hm': 'hôm',
    'dt': 'điện thoại', 'sdt': 'số điện thoại',
    'fb': 'facebook', 'yt': 'youtube',
    'ad': 'admin', 'mod': 'moderator',
    'nx': 'nhận xét', 'đt': 'điện thoại',
    'gato': 'ghen ăn tức ở',
    'wtf': 'what the f',
    'dm': 'đ mẹ', 'vcl': 'vãi',
    'clgt': 'chắc luôn',
    'oy': 'rồi', 'ùi': 'rồi',
    'biet': 'biết', 'hiu': 'hiểu',
    'thik': 'thích', 'hjhj': 'hihi',
    'tui': 'tôi', 'mik': 'mình',
    'ngta': 'người ta', 'nyc': 'người yêu cũ',
    'ny': 'người yêu', 'gf': 'bạn gái', 'bf': 'bạn trai',
    'sg': 'sài gòn', 'hn': 'hà nội',
    'vn': 'việt nam',
    'nhma': 'nhưng mà', 'nma': 'nhưng mà',
    'tl': 'trả lời', 'cmn': 'con mẹ nó',
}


# ============================================================================
# PHẦN 3: PIPELINE TIỀN XỬ LÝ VĂN BẢN TIẾNG VIỆT
# ============================================================================

def remove_urls(text):
    """Xóa URL khỏi văn bản"""
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    return text

def remove_emails(text):
    """Xóa email khỏi văn bản"""
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)
    return text

def remove_phone_numbers(text):
    """Xóa số điện thoại khỏi văn bản"""
    text = re.sub(r'(\+84|0)\d{9,10}', ' ', text)
    return text

def remove_html_tags(text):
    """Xóa HTML tags"""
    text = re.sub(r'<[^>]+>', ' ', text)
    return text

def remove_emojis(text):
    """Xóa emoji và các ký tự unicode đặc biệt"""
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"
        u"\u3030"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(' ', text)

def normalize_repeated_chars(text):
    """
    Chuẩn hóa ký tự lặp lại quá nhiều
    Ví dụ: "đẹpppppp" → "đẹpp", "hahahahaha" → "haha"
    """
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text

def replace_teencode(text, teencode_dict=TEENCODE_DICT):
    """
    Thay thế teencode bằng từ chuẩn
    Sử dụng word boundary để tránh thay thế sai
    """
    words = text.split()
    result = []
    for word in words:
        lower_word = word.lower()
        if lower_word in teencode_dict:
            result.append(teencode_dict[lower_word])
        else:
            result.append(word)
    return ' '.join(result)

def remove_special_characters(text):
    """
    Xóa ký tự đặc biệt, giữ lại chữ cái tiếng Việt, số và khoảng trắng
    """
    # Giữ lại chữ cái (bao gồm tiếng Việt), số, khoảng trắng
    text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', ' ', text, flags=re.IGNORECASE)
    return text

def normalize_whitespace(text):
    """Chuẩn hóa khoảng trắng"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_text(text):
    """
    Pipeline tiền xử lý hoàn chỉnh cho 1 đoạn văn bản tiếng Việt.
    Thứ tự xử lý:
    1. Kiểm tra null/NaN
    2. Chuyển lowercase
    3. Xóa HTML tags
    4. Xóa URL
    5. Xóa email
    6. Xóa số điện thoại
    7. Xóa emoji
    8. Chuẩn hóa ký tự lặp
    9. Thay thế teencode
    10. Xóa ký tự đặc biệt
    11. Chuẩn hóa khoảng trắng
    """
    # Bước 0: Kiểm tra null
    if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
        return ''
    
    # Bước 1: Chuyển lowercase
    text = text.lower()
    
    # Bước 2: Xóa HTML tags
    text = remove_html_tags(text)
    
    # Bước 3: Xóa URL
    text = remove_urls(text)
    
    # Bước 4: Xóa email
    text = remove_emails(text)
    
    # Bước 5: Xóa số điện thoại
    text = remove_phone_numbers(text)
    
    # Bước 6: Xóa emoji
    text = remove_emojis(text)
    
    # Bước 7: Chuẩn hóa ký tự lặp
    text = normalize_repeated_chars(text)
    
    # Bước 8: Thay thế teencode
    text = replace_teencode(text)
    
    # Bước 9: Xóa ký tự đặc biệt
    text = remove_special_characters(text)
    
    # Bước 10: Chuẩn hóa khoảng trắng
    text = normalize_whitespace(text)
    
    return text


# ============================================================================
# PHẦN 4: XỬ LÝ DỮ LIỆU THIẾU VÀ TRÙNG LẶP
# ============================================================================

def handle_missing_values(df, name='dataset'):
    """
    Xử lý dữ liệu thiếu:
    - Với free_text null: xóa dòng (vì không thể phân loại văn bản rỗng)
    """
    null_count = df['free_text'].isnull().sum()
    print(f"🔍 [{name}] Số dòng có free_text null: {null_count}")
    
    if null_count > 0:
        df = df.dropna(subset=['free_text'])
        print(f"   ✅ Đã xóa {null_count} dòng null. Còn lại: {df.shape[0]} dòng")
    
    return df


def handle_duplicates(df, name='dataset'):
    """
    Xử lý dữ liệu trùng lặp:
    - Xóa các dòng hoàn toàn giống nhau (cả text lẫn label)
    """
    dup_count = df.duplicated().sum()
    print(f"🔍 [{name}] Số dòng trùng lặp: {dup_count}")
    
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f"   ✅ Đã xóa {dup_count} dòng trùng lặp. Còn lại: {df.shape[0]} dòng")
    
    return df


# ============================================================================
# PHẦN 5: HÀM CHÍNH - XỬ LÝ TOÀN BỘ PIPELINE
# ============================================================================

def process_dataset(df, name='dataset'):
    """
    Xử lý toàn bộ pipeline cho 1 DataFrame:
    1. Xử lý missing values
    2. Xử lý duplicates
    3. Tiền xử lý văn bản
    4. Xóa dòng rỗng sau tiền xử lý
    """
    print(f"\n{'='*60}")
    print(f"🚀 BẮT ĐẦU XỬ LÝ: {name.upper()}")
    print(f"{'='*60}")
    print(f"📊 Kích thước ban đầu: {df.shape}")
    
    # Bước 1: Xử lý missing values
    df = handle_missing_values(df, name)
    
    # Bước 2: Xử lý duplicates
    df = handle_duplicates(df, name)
    
    # Bước 3: Tiền xử lý văn bản
    print(f"\n📝 Đang tiền xử lý văn bản...")
    df = df.copy()
    df['free_text_clean'] = df['free_text'].apply(preprocess_text)
    
    # Bước 4: Xóa dòng rỗng sau tiền xử lý
    empty_after = (df['free_text_clean'] == '').sum()
    print(f"   Số dòng rỗng sau tiền xử lý: {empty_after}")
    if empty_after > 0:
        df = df[df['free_text_clean'] != '']
        print(f"   ✅ Đã xóa {empty_after} dòng rỗng. Còn lại: {df.shape[0]} dòng")
    
    # Thống kê sau xử lý
    print(f"\n📊 KẾT QUẢ SAU TIỀN XỬ LÝ:")
    print(f"   Kích thước: {df.shape}")
    print(f"   Phân bố nhãn: {df['label_id'].value_counts().sort_index().to_dict()}")
    print(f"   Độ dài text trung bình: {df['free_text_clean'].str.len().mean():.1f} ký tự")
    print(f"   Số từ trung bình: {df['free_text_clean'].str.split().str.len().mean():.1f} từ")
    
    return df


def check_data_consistency(datasets_clean):
    """
    Kiểm tra tính nhất quán của dữ liệu sau tiền xử lý
    """
    print(f"\n{'='*60}")
    print(f"🔎 KIỂM TRA TÍNH NHẤT QUÁN DỮ LIỆU")
    print(f"{'='*60}")
    
    for name, df in datasets_clean.items():
        print(f"\n📋 [{name}]:")
        # Kiểm tra null
        null_count = df[['free_text_clean', 'label_id']].isnull().sum()
        print(f"   Null values: {null_count.to_dict()}")
        
        # Kiểm tra label hợp lệ (chỉ 0, 1, 2)
        valid_labels = {0, 1, 2}
        actual_labels = set(df['label_id'].unique())
        if actual_labels.issubset(valid_labels):
            print(f"   ✅ Labels hợp lệ: {sorted(actual_labels)}")
        else:
            invalid = actual_labels - valid_labels
            print(f"   ❌ Labels không hợp lệ: {invalid}")
        
        # Kiểm tra text rỗng
        empty = (df['free_text_clean'].str.strip() == '').sum()
        print(f"   Text rỗng: {empty}")
        
        # Kiểm tra duplicates
        dup = df[['free_text_clean', 'label_id']].duplicated().sum()
        print(f"   Duplicates còn lại: {dup}")
    
    print(f"\n✅ Kiểm tra nhất quán hoàn tất!")


# ============================================================================
# PHẦN 6: MAIN - THỰC THI
# ============================================================================

if __name__ == '__main__':
    # Đường dẫn thư mục chứa dữ liệu
    DATA_DIR = '.'
    OUTPUT_DIR = '.'
    
    # Bước 1: Đọc dữ liệu
    print("=" * 60)
    print("📂 BƯỚC 1: ĐỌC DỮ LIỆU")
    print("=" * 60)
    datasets = load_datasets(DATA_DIR)
    
    # Bước 2: Xử lý từng tập dữ liệu
    datasets_clean = {}
    for name, df in datasets.items():
        df_clean = process_dataset(df, name)
        datasets_clean[name] = df_clean
    
    # Bước 3: Kiểm tra tính nhất quán
    check_data_consistency(datasets_clean)
    
    # Bước 4: Lưu dữ liệu đã xử lý
    print(f"\n{'='*60}")
    print(f"💾 LƯU DỮ LIỆU ĐÃ XỬ LÝ")
    print(f"{'='*60}")
    
    for name, df in datasets_clean.items():
        # Lưu với cả cột gốc và cột đã xử lý
        output_path = os.path.join(OUTPUT_DIR, f'{name}_clean.csv')
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"   ✅ Đã lưu: {output_path} ({df.shape[0]} dòng)")
    
    # Tổng kết
    print(f"\n{'='*60}")
    print(f"🎯 TỔNG KẾT TIỀN XỬ LÝ")
    print(f"{'='*60}")
    for name in ['train', 'dev', 'test']:
        orig = datasets[name].shape[0]
        clean = datasets_clean[name].shape[0]
        removed = orig - clean
        print(f"   {name}: {orig} → {clean} (xóa {removed} dòng, -{removed/orig*100:.1f}%)")
    
    print(f"\n🏁 HOÀN TẤT TIỀN XỬ LÝ DỮ LIỆU!")
