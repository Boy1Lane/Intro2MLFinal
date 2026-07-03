from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request

from app.backend.constants import DISPLAY_NAMES, MODEL_ORDER
from app.backend.schemas import (CreateWatchRequest, ModelOption,
                                  MonitorComment, TokenScore, WatchDetail,
                                  WatchSummary)
from app.backend.services.explain import explain_tokens
from app.backend.services.monitor import VALID_MODELS, Watch

# how many driving tokens to surface per toxic comment
_MAX_EXPLAIN_TOKENS = 8

router = APIRouter(prefix="/monitor")


def _toxic_count(watch: Watch) -> int:
    return sum(1 for c in watch.comments if c.toxic)


def _summary(watch: Watch) -> WatchSummary:
    return WatchSummary(
        id=watch.id, url=watch.url, label=watch.label,
        created_at=watch.created_at, last_scan=watch.last_scan,
        last_error=watch.last_error, alert_count=watch.alert_count,
        total_comments=len(watch.comments), toxic_count=_toxic_count(watch),
        model=watch.model,
    )


def _explain(registry, text: str, label: int) -> list[TokenScore]:
    """Top tokens driving a toxic label, strongest first (sklearn LR proxy)."""
    if registry is None:
        return []
    scored = [t for t in explain_tokens(registry, text, label) if t["score"] != 0.0]
    scored.sort(key=lambda t: abs(t["score"]), reverse=True)
    return [TokenScore(**t) for t in scored[:_MAX_EXPLAIN_TOKENS]]


def _detail(watch: Watch, registry=None) -> WatchDetail:
    return WatchDetail(
        **_summary(watch).model_dump(),
        comments=[MonitorComment(
            text=c.text, label=c.label, label_name=c.label_name,
            proba=c.proba, toxic=c.toxic, model=c.model, seen_at=c.seen_at,
            tokens=_explain(registry, c.text, c.label) if c.toxic else [],
        ) for c in watch.comments],
    )


def _service(request: Request):
    return request.app.state.monitor


def _registry(request: Request):
    return getattr(request.app.state, "registry", None)


@router.post("/watches", response_model=WatchSummary)
def create_watch(req: CreateWatchRequest, request: Request) -> WatchSummary:
    parsed = urlparse(req.url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=400, detail="URL phải là http/https hợp lệ.")
    if req.model not in VALID_MODELS:
        raise HTTPException(status_code=400, detail="Model không hợp lệ.")
    try:
        watch = _service(request).add(req.url, req.label, req.model)
    except ValueError:
        raise HTTPException(status_code=400, detail="Đã đạt giới hạn số URL theo dõi.")
    return _summary(watch)


@router.get("/models", response_model=list[ModelOption])
def list_models(request: Request) -> list[ModelOption]:
    """Models selectable for a watch: PhoBERT (auto) + loaded sklearn models."""
    reg = _registry(request)
    available = set(getattr(reg, "models", {}) or {})
    out = []
    for key in MODEL_ORDER:  # PhoBERT first, then sklearn in report order
        if key == "PhoBERT" or key in available:
            out.append(ModelOption(key=key, name=DISPLAY_NAMES[key]))
    return out


@router.get("/watches", response_model=list[WatchSummary])
def list_watches(request: Request) -> list[WatchSummary]:
    return [_summary(w) for w in _service(request).list()]


@router.get("/watches/{watch_id}", response_model=WatchDetail)
def get_watch(watch_id: str, request: Request) -> WatchDetail:
    watch = _service(request).get(watch_id)
    if watch is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy watch.")
    return _detail(watch, _registry(request))


@router.post("/watches/{watch_id}/scan", response_model=WatchDetail)
def scan_watch(watch_id: str, request: Request) -> WatchDetail:
    watch = _service(request).scan(watch_id)
    if watch is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy watch.")
    return _detail(watch, _registry(request))


@router.post("/watches/{watch_id}/ack", response_model=WatchSummary)
def ack_watch(watch_id: str, request: Request) -> WatchSummary:
    watch = _service(request).ack(watch_id)
    if watch is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy watch.")
    return _summary(watch)


@router.delete("/watches/{watch_id}")
def delete_watch(watch_id: str, request: Request) -> dict:
    if not _service(request).delete(watch_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy watch.")
    return {"deleted": True}
