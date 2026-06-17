# TEENCODE_ML: ported from notebooks/02_text_preprocessing.ipynb + 06_model_training.ipynb
TEENCODE_ML = {
    "ko": "không", "k": "không", "kh": "không", "khg": "không", "kp": "không phải",
    "kq": "không quan", "dc": "được", "đc": "được", "dk": "được", "đk": "được",
    "nc": "nước", "ng": "người", "ns": "nói", "mk": "mình", "mn": "mọi người",
    "mng": "mọi người", "bn": "bạn", "b": "bạn", "bro": "bạn", "ib": "nhắn tin",
    "rep": "trả lời", "vs": "với", "v": "với", "voi": "với", "r": "rồi",
    "rui": "rồi", "rii": "rồi", "ntn": "như thế nào", "j": "gì", "ji": "gì",
    "z": "gì", "gi": "gì", "a": "anh", "e": "em", "c": "chị", "đi": "đi",
    "qua": "qua", "trc": "trước", "tg": "thời gian", "bt": "bình thường",
    "bth": "bình thường", "vl": "vãi", "vkl": "vãi", "nch": "nói chuyện",
    "nt": "nhắn tin", "hk": "không", "hem": "không", "bi": "bị", "bik": "biết",
    "ck": "chồng", "vk": "vợ", "tks": "thanks", "thanks": "cảm ơn",
    "thks": "cảm ơn", "ok": "được", "okie": "được", "oke": "được",
    "plz": "làm ơn", "pls": "làm ơn", "sr": "xin lỗi", "sorry": "xin lỗi",
    "lun": "luôn", "lm": "làm", "đag": "đang", "dg": "đang", "trg": "trong",
    "trog": "trong", "cx": "cũng", "cg": "cũng", "đb": "đặc biệt",
    "cb": "chuẩn bị", "h": "giờ", "hm": "hôm", "dt": "điện thoại",
    "sdt": "số điện thoại", "fb": "facebook", "yt": "youtube", "ad": "admin",
    "mod": "moderator", "nx": "nhận xét", "đt": "điện thoại",
    "gato": "ghen ăn tức ở", "wtf": "what the f", "dm": "đ mẹ", "vcl": "vãi",
    "clgt": "chắc luôn", "oy": "rồi", "ùi": "rồi", "biet": "biết", "hiu": "hiểu",
    "thik": "thích", "hjhj": "hihi", "tui": "tôi", "mik": "mình",
    "ngta": "người ta", "nyc": "người yêu cũ", "ny": "người yêu", "gf": "bạn gái",
    "bf": "bạn trai", "sg": "sài gòn", "hn": "hà nội", "vn": "việt nam",
    "nhma": "nhưng mà", "nma": "nhưng mà", "tl": "trả lời", "cmn": "con mẹ nó",
}

# TEENCODE_PHOBERT: ported from notebooks/06_model_training_DL.ipynb (profanity-aware)
TEENCODE_PHOBERT = {
    "dell": "đéo", "del": "đéo", "đell": "đéo", "đel": "đéo", "loz": "lồn",
    "lon": "lồn", "lòn": "lồn", "l": "lồn", "coin card": "củ cặc", "cc": "củ cặc",
    "cức": "cứt", "ms": "mới", "bh": "bây giờ", "kb": "không biết", "kk": "cười",
    "haha": "cười", "đhs": "đéo hiểu sao", "dm": "địt mẹ", "đm": "địt mẹ",
    "dmm": "địt mẹ mày", "vcl": "vãi lồn", "vl": "vãi lồn", "vkl": "vãi lồn",
    "cl": "cái lồn", "clgt": "cái lồn gì thế", "đcm": "địt con mẹ",
    "dcm": "địt con mẹ",
}


def normalize_teencode(text: str, mapping: dict) -> str:
    words = str(text).split()
    return " ".join(mapping.get(w.lower(), w) for w in words)
