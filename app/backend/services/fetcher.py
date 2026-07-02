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


def _extract(html: str, max_len: int, min_len: int) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    nodes = soup.find_all("p") + soup.find_all("li")
    nodes += soup.find_all(attrs={"class": lambda c: c and "comment" in " ".join(
        c if isinstance(c, list) else [c]).lower()})
    nodes += soup.find_all(attrs={"id": lambda i: i and "comment" in i.lower()})
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
        with httpx.Client(timeout=timeout, follow_redirects=True,
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
