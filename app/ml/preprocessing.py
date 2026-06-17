import re


def remove_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_emails(text):
    return re.sub(r"\S+@\S+\.\S+", " ", text)


def remove_phone_numbers(text):
    return re.sub(r"(\+84|0)\d{9,10}", " ", text)


def remove_html_tags(text):
    return re.sub(r"<[^>]+>", " ", text)


def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f1e0-\U0001f1ff"
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "♀-♂"
        "☀-⭕"
        "‍"
        "⏏"
        "⏩"
        "⌚"
        "️"
        "〰"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(" ", text)


def normalize_repeated_chars(text):
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def remove_special_characters(text):
    return re.sub(
        r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]",
        " ",
        text,
        flags=re.IGNORECASE,
    )


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text):
    if text is None or not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_phone_numbers(text)
    text = remove_emojis(text)
    text = normalize_repeated_chars(text)
    text = remove_special_characters(text)
    return normalize_whitespace(text)
