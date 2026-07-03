import html as _html
import re
from typing import Protocol, runtime_checkable

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


@runtime_checkable
class SourceAdapter(Protocol):
    """A comment source. Specific adapters (news APIs, forum markup) are tried
    before the catch-all GenericAdapter; the first whose ``matches`` returns
    True handles the URL."""

    name: str      # stable key, e.g. "VnExpress"
    display: str   # UI label

    def matches(self, url: str) -> bool: ...

    def fetch(self, url: str, *, max_len: int, min_len: int, timeout: float,
              _transport=None) -> list[str]: ...


def clean_items(raw_items: list[str], max_len: int, min_len: int) -> list[str]:
    """Strip HTML tags/entities/whitespace from API comment bodies, drop
    too-short/too-long items, and dedup while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in raw_items:
        text = _html.unescape(_TAG.sub(" ", raw or "")).replace("\xa0", " ")
        text = _WS.sub(" ", text).strip()
        if not text or len(text) < min_len or len(text) > max_len or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
