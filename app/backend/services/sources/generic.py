from app.backend.services.fetcher import fetch_comments as _generic_fetch


class GenericAdapter:
    """Catch-all: guarded fetch + trafilatura/BeautifulSoup extraction. Handles
    any URL best-effort (server-rendered news, blogs, forums)."""

    name = "Generic"
    display = "Trang bất kỳ (best-effort)"

    def matches(self, url: str) -> bool:
        return True

    def fetch(self, url: str, *, max_len: int, min_len: int, timeout: float,
              _transport=None) -> list[str]:
        return _generic_fetch(url, max_len=max_len, min_len=min_len,
                              timeout=timeout, _transport=_transport)
