import json
import re

from app.backend.services.fetcher import http_get
from app.backend.services.sources.base import clean_items

# news id is the trailing number in .../slug-<id>.htm
_ID = re.compile(r"-(\d{6,})\.html?$")


class TuoiTreAdapter:
    """Tuổi Trẻ comments come from id.tuoitre.vn's getlist-comment API. The
    response wraps the comment array as a JSON string under the "Data" key."""

    name = "TuoiTre"
    display = "Tuổi Trẻ"

    def matches(self, url: str) -> bool:
        base = url.split("?")[0].split("#")[0]
        return "tuoitre.vn" in url.lower() and bool(_ID.search(base))

    def fetch(self, url: str, *, max_len: int, min_len: int, timeout: float,
              _transport=None) -> list[str]:
        m = _ID.search(url.split("?")[0].split("#")[0])
        if not m:
            return []
        news_id = m.group(1)
        api = ("https://id.tuoitre.vn/api/getlist-comment.api?pageindex=1"
               f"&objId={news_id}&objType=1&sort=2")
        resp = http_get(api, timeout=timeout, expect_html=False,
                        _transport=_transport)
        try:
            payload = resp.json()
            data = payload.get("Data")
            items = json.loads(data) if isinstance(data, str) else (data or [])
        except (ValueError, TypeError):
            return []
        return clean_items([it.get("content", "") for it in items],
                           max_len, min_len)
