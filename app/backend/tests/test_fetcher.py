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
