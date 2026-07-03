import ipaddress
import socket
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup


class FetchError(Exception):
    """Raised when a watched URL cannot be fetched or parsed."""


def is_blocked_host(host: str) -> bool:
    """True if host is loopback/private/link-local/reserved (SSRF guard)."""
    if not host or host.lower() == "localhost":
        return True
    candidates = [host]
    try:
        infos = socket.getaddrinfo(host, None)
        candidates += [info[4][0] for info in infos]
    except OSError:
        pass
    for cand in candidates:
        try:
            ip = ipaddress.ip_address(cand)
        except ValueError:
            continue
        if (ip.is_loopback or ip.is_private or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return True
    return False


def _guard(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise FetchError("Chỉ hỗ trợ URL http/https.")
    if not parsed.hostname or is_blocked_host(parsed.hostname):
        raise FetchError("Host không hợp lệ hoặc bị chặn (nội bộ).")


# Page regions that almost never hold user comments — dropped to cut boilerplate
# (site chrome, menus, article headers/footers, submit forms).
_NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer",
               "aside", "form"]

# class/id substrings marking comment / forum / review blocks. Covers common
# engines (Disqus, XenForo, vBulletin, WordPress) and VN news/forum markup.
_COMMENT_HINTS = ("comment", "cmt", "reply", "respond", "review", "message",
                  "disqus", "binh-luan", "binhluan", "phan-hoi", "phanhoi",
                  "thao-luan", "thaoluan")


def _hint_match(value) -> bool:
    """True when a class/id attribute value contains a comment-block hint."""
    if not value:
        return False
    text = " ".join(value if isinstance(value, list) else [value]).lower()
    return any(hint in text for hint in _COMMENT_HINTS)


def _extract(html: str, max_len: int, min_len: int) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    # Comment/forum-specific containers carry the strongest signal; generic
    # text blocks are a fallback for plain pages without comment markup.
    nodes = soup.find_all(attrs={"class": _hint_match})
    nodes += soup.find_all(attrs={"id": _hint_match})
    nodes += soup.find_all(["p", "li", "blockquote"])
    seen: set[str] = set()
    out: list[str] = []
    for node in nodes:
        text = node.get_text(" ", strip=True)
        if not text or len(text) < min_len or len(text) > max_len:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def fetch_comments(url: str, *, max_len: int = 5000, min_len: int = 3,
                   timeout: float = 10.0, _transport=None) -> list[str]:
    _guard(url)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False,
                          transport=_transport,
                          headers={"User-Agent": "Mozilla/5.0 (ViHSD-Monitor)"}) as c:
            resp = c.get(url)
    except httpx.HTTPError as e:
        raise FetchError(f"Không tải được URL: {e}")
    if resp.status_code >= 400:
        raise FetchError(f"URL trả về mã lỗi {resp.status_code}.")
    ctype = resp.headers.get("content-type", "")
    if "html" not in ctype.lower():
        raise FetchError("Nội dung không phải HTML.")
    return _extract(resp.text, max_len, min_len)
