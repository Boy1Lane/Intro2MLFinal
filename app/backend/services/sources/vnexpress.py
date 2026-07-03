import re

from app.backend.services.fetcher import http_get
from app.backend.services.sources.base import clean_items

# article id is the trailing number in .../slug-<id>.html
_ID = re.compile(r"-(\d{5,})\.html?$")


class VnExpressAdapter:
    """VnExpress comments live in a public JSON API (usi-saas), never in the
    article HTML. The article id is the trailing number in the URL."""

    name = "VnExpress"
    display = "VnExpress"

    def matches(self, url: str) -> bool:
        base = url.split("?")[0].split("#")[0]
        return "vnexpress.net" in url.lower() and bool(_ID.search(base))

    def fetch(self, url: str, *, max_len: int, min_len: int, timeout: float,
              _transport=None) -> list[str]:
        m = _ID.search(url.split("?")[0].split("#")[0])
        if not m:
            return []
        object_id = m.group(1)
        api = ("https://usi-saas.vnexpress.net/index/get?offset=0&limit=100"
               f"&sort=like&objectid={object_id}&objecttype=1&siteid=1000000")
        resp = http_get(api, timeout=timeout, expect_html=False,
                        _transport=_transport)
        try:
            items = (resp.json().get("data") or {}).get("items") or []
        except ValueError:
            return []
        return clean_items([it.get("content", "") for it in items],
                           max_len, min_len)
