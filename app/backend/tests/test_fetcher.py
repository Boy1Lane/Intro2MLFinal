import httpx
import pytest

from app.backend.services import fetcher
from app.backend.services.fetcher import FetchError, fetch_comments, is_blocked_host

HTML = """
<html><body>
  <p>Bình luận đầu tiên rất hay</p>
  <div class="comment">Thằng này nói nhảm quá</div>
  <ul><li>Ý kiến thứ ba ở đây</li></ul>
  <p>x</p>
  <p>Bình luận đầu tiên rất hay</p>
</body></html>
"""


def _mock_transport(status=200, content=HTML, content_type="text/html"):
    def handler(request):
        return httpx.Response(status, headers={"content-type": content_type},
                              text=content)
    return httpx.MockTransport(handler)


def test_fetch_parses_and_dedups():
    out = fetch_comments("https://example.com/post", _transport=_mock_transport())
    assert "Bình luận đầu tiên rất hay" in out
    assert "Thằng này nói nhảm quá" in out
    assert "Ý kiến thứ ba ở đây" in out
    assert out.count("Bình luận đầu tiên rất hay") == 1  # deduped
    assert "x" not in out  # below min_len


FORUM_HTML = """
<html><body>
  <nav><a>Trang chủ</a> <a>Diễn đàn</a></nav>
  <header>Tiêu đề bài báo dài dòng ở đây</header>
  <article><p>Nội dung bài báo chính thức</p></article>
  <div class="comment-body">Bài báo này viết quá tệ</div>
  <div class="cmt-content">Đồng ý với ý kiến bên trên</div>
  <blockquote class="review">Nội dung phản hồi trích dẫn</blockquote>
  <footer>Bản quyền 2026 thuộc về toà soạn</footer>
</body></html>
"""


def test_bs_fallback_strips_noise_and_catches_forum_comments():
    # Exercises the raw BeautifulSoup fallback heuristic directly (used when
    # trafilatura yields nothing, e.g. tiny/odd pages).
    out = fetcher._extract(FORUM_HTML, max_len=5000, min_len=3)
    # forum/news comment containers are captured
    assert "Bài báo này viết quá tệ" in out
    assert "Đồng ý với ý kiến bên trên" in out
    assert "Nội dung phản hồi trích dẫn" in out
    # site chrome (nav/header/footer) is dropped as boilerplate
    assert not any("Trang chủ" in o for o in out)
    assert not any("Diễn đàn" in o for o in out)
    assert not any("Tiêu đề bài báo" in o for o in out)
    assert not any("Bản quyền" in o for o in out)


XENFORO_HTML = """
<html><body>
  <article class="message">
    <span class="message-name">nguoidung1</span><span>Member</span>
    <div class="bbWrapper">Bài viết đầu tiên trong thread này</div>
  </article>
  <article class="message">
    <span class="message-name">nguoidung2</span>
    <div class="bbWrapper">Phản hồi thứ hai của thành viên khác</div>
  </article>
</body></html>
"""


def test_forum_extraction_gets_post_bodies_not_usernames():
    out = fetcher._extract_forum(XENFORO_HTML, max_len=5000, min_len=3)
    assert out == ["Bài viết đầu tiên trong thread này",
                   "Phản hồi thứ hai của thành viên khác"]
    # usernames / badges outside .bbWrapper are excluded
    assert not any("nguoidung" in o for o in out)
    assert "Member" not in out


def test_forum_extraction_is_preferred_over_trafilatura():
    out = fetch_comments("https://forum.example/thread",
                         _transport=_mock_transport(content=XENFORO_HTML))
    assert "Bài viết đầu tiên trong thread này" in out
    assert not any("nguoidung" in o for o in out)


def test_trafilatura_is_primary_bs_is_fallback():
    # trafilatura pulls main content + comments and strips boilerplate on real
    # pages; the fetcher prefers it, falling back to _extract only when empty.
    out = fetch_comments("https://bao.example/bai",
                         _transport=_mock_transport(content=FORUM_HTML))
    assert "Bài báo này viết quá tệ" in out
    assert "Đồng ý với ý kiến bên trên" in out


def test_fetch_rejects_non_html():
    t = _mock_transport(content_type="application/json", content="{}")
    with pytest.raises(FetchError):
        fetch_comments("https://example.com/api", _transport=t)


def test_fetch_rejects_error_status():
    with pytest.raises(FetchError):
        fetch_comments("https://example.com/404", _transport=_mock_transport(status=404))


def test_blocked_hosts():
    assert is_blocked_host("localhost")
    assert is_blocked_host("127.0.0.1")
    assert is_blocked_host("10.0.0.5")
    assert is_blocked_host("192.168.1.1")
    assert is_blocked_host("169.254.1.1")
    assert not is_blocked_host("example.com")


def test_fetch_rejects_bad_scheme():
    with pytest.raises(FetchError):
        fetch_comments("file:///etc/passwd")


def test_fetch_rejects_internal_host():
    with pytest.raises(FetchError):
        fetch_comments("http://localhost:8000/secret")


def test_redirect_to_internal_host_not_followed():
    """Verify that redirects to internal hosts are not transparently followed.

    With follow_redirects=False, the client returns the 302 response itself,
    which lacks HTML content-type and should raise FetchError.
    """
    def redirect_handler(request):
        # Return a 302 redirect to an internal AWS metadata endpoint
        return httpx.Response(
            302,
            headers={
                "location": "http://169.254.169.254/latest/meta-data/",
                "content-type": "text/plain",
            },
            text="Moved",
        )

    transport = httpx.MockTransport(redirect_handler)
    # The redirect response is not HTML, so it should raise FetchError
    with pytest.raises(FetchError, match="không phải HTML"):
        fetch_comments("https://example.com/page", _transport=transport)
