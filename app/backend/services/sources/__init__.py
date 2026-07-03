"""Comment source adapters.

A watch URL is dispatched to the first adapter whose ``matches`` returns True.
Site-specific adapters (news comment APIs) come first for reliable extraction;
GenericAdapter is last and matches everything as a best-effort fallback.
"""
from app.backend.services.fetcher import FetchError  # re-export for callers
from app.backend.services.sources.base import SourceAdapter
from app.backend.services.sources.generic import GenericAdapter
from app.backend.services.sources.tuoitre import TuoiTreAdapter
from app.backend.services.sources.vnexpress import VnExpressAdapter

# order matters: specific first, Generic (matches all) last
ADAPTERS: list[SourceAdapter] = [
    VnExpressAdapter(),
    TuoiTreAdapter(),
    GenericAdapter(),
]

__all__ = ["FetchError", "fetch_comments", "resolve", "list_sources", "ADAPTERS"]


def resolve(url: str) -> SourceAdapter:
    for adapter in ADAPTERS:
        if adapter.matches(url):
            return adapter
    return ADAPTERS[-1]


def fetch_comments(url: str, *, max_len: int = 5000, min_len: int = 3,
                   timeout: float = 10.0, _transport=None) -> list[str]:
    return resolve(url).fetch(url, max_len=max_len, min_len=min_len,
                              timeout=timeout, _transport=_transport)


def list_sources() -> list[dict]:
    """First-class supported sources (excludes the Generic catch-all), for UI."""
    return [{"name": a.name, "display": a.display}
            for a in ADAPTERS if a.name != "Generic"]
