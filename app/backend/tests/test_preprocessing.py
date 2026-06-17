from app.ml.preprocessing import preprocess_text
from app.ml.teencode import normalize_teencode, TEENCODE_ML, TEENCODE_PHOBERT


def test_preprocess_lowercases_and_strips_url_emoji():
    out = preprocess_text("Xem tại https://abc.com nhé 😀😀😀")
    assert "http" not in out
    assert "😀" not in out
    assert out == out.lower()


def test_preprocess_handles_none_and_empty():
    assert preprocess_text(None) == ""
    assert preprocess_text("   ") == ""


def test_preprocess_collapses_repeated_chars():
    assert preprocess_text("đẹppppp") == "đẹpp"


def test_teencode_ml_replaces_known_tokens():
    assert normalize_teencode("ko biet", TEENCODE_ML) == "không biết"


def test_teencode_phobert_has_profanity_mapping():
    assert TEENCODE_PHOBERT["vcl"] == "vãi lồn"
    assert normalize_teencode("vcl", TEENCODE_PHOBERT) == "vãi lồn"


def test_two_dicts_differ():
    assert TEENCODE_ML != TEENCODE_PHOBERT
