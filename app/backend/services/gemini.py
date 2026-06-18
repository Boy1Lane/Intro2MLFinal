PROMPT = (
    "Bạn là trợ lý kiểm duyệt. Viết lại bình luận tiếng Việt sau cho lịch sự, "
    "tôn trọng, GIỮ NGUYÊN ý chính, bỏ toàn bộ từ ngữ thù ghét/xúc phạm. "
    "Chỉ trả về câu đã viết lại, không giải thích.\n\nBình luận: {text}"
)


def rewrite_polite(text: str, api_key: str | None) -> str:
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY chưa cấu hình.")
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=PROMPT.format(text=text),
        )
        out = (resp.text or "").strip()
        if not out:
            raise RuntimeError("Gemini trả về rỗng.")
        return out
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Gemini lỗi: {e}") from e
