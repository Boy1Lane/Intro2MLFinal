from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request

from app.backend.schemas import (CreateWatchRequest, MonitorComment,
                                  WatchDetail, WatchSummary)
from app.backend.services.monitor import Watch

router = APIRouter(prefix="/monitor")


def _toxic_count(watch: Watch) -> int:
    return sum(1 for c in watch.comments if c.toxic)


def _summary(watch: Watch) -> WatchSummary:
    return WatchSummary(
        id=watch.id, url=watch.url, label=watch.label,
        created_at=watch.created_at, last_scan=watch.last_scan,
        last_error=watch.last_error, alert_count=watch.alert_count,
        total_comments=len(watch.comments), toxic_count=_toxic_count(watch),
    )


def _detail(watch: Watch) -> WatchDetail:
    return WatchDetail(
        **_summary(watch).model_dump(),
        comments=[MonitorComment(
            text=c.text, label=c.label, label_name=c.label_name,
            proba=c.proba, toxic=c.toxic, model=c.model, seen_at=c.seen_at,
        ) for c in watch.comments],
    )


def _service(request: Request):
    return request.app.state.monitor


@router.post("/watches", response_model=WatchSummary)
def create_watch(req: CreateWatchRequest, request: Request) -> WatchSummary:
    parsed = urlparse(req.url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=400, detail="URL phải là http/https hợp lệ.")
    try:
        watch = _service(request).add(req.url, req.label)
    except ValueError:
        raise HTTPException(status_code=400, detail="Đã đạt giới hạn số URL theo dõi.")
    return _summary(watch)


@router.get("/watches", response_model=list[WatchSummary])
def list_watches(request: Request) -> list[WatchSummary]:
    return [_summary(w) for w in _service(request).list()]


@router.get("/watches/{watch_id}", response_model=WatchDetail)
def get_watch(watch_id: str, request: Request) -> WatchDetail:
    watch = _service(request).get(watch_id)
    if watch is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy watch.")
    return _detail(watch)


@router.post("/watches/{watch_id}/scan", response_model=WatchDetail)
def scan_watch(watch_id: str, request: Request) -> WatchDetail:
    watch = _service(request).scan(watch_id)
    if watch is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy watch.")
    return _detail(watch)


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
