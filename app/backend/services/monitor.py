import hashlib
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

from app.backend.constants import DISPLAY_NAMES, LABEL_NAMES, SKLEARN_ORDER
from app.backend.services.sources import FetchError, fetch_comments

TOXIC = {"OFFENSIVE", "HATE"}
# "PhoBERT" means prefer PhoBERT, fall back to Logistic Regression; the sklearn
# keys pin classification to that specific model.
DEFAULT_MODEL = "PhoBERT"
VALID_MODELS = (DEFAULT_MODEL, *SKLEARN_ORDER)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@dataclass
class Comment:
    text: str
    label: int
    label_name: str
    proba: list[float]
    toxic: bool
    model: str
    seen_at: str
    hash: str


@dataclass
class Watch:
    id: str
    url: str
    label: str | None = None
    created_at: str = field(default_factory=_now)
    last_scan: str | None = None
    last_error: str | None = None
    alert_count: int = 0
    model: str = DEFAULT_MODEL
    comments: list[Comment] = field(default_factory=list)
    seen_hashes: set[str] = field(default_factory=set)


class MonitorService:
    """
    In-memory watch registry with background scanning and JSON persistence.

    NOTE: This service assumes a single process/worker. Running multiple uvicorn
    workers will cause diverging in-memory state and duplicate scans; a shared
    external store (e.g. Redis or a database) would be required for multi-worker
    deployment.
    """

    def __init__(self, registry, phobert, settings, fetch=fetch_comments):
        self.registry = registry
        self.phobert = phobert
        self.settings = settings
        self._fetch = fetch
        self._watches: dict[str, Watch] = {}
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()  # serializes file I/O in save()

    # ---- CRUD ---------------------------------------------------------
    def add(self, url: str, label: str | None = None,
            model: str = DEFAULT_MODEL) -> Watch:
        if model not in VALID_MODELS:
            model = DEFAULT_MODEL
        with self._lock:
            if len(self._watches) >= self.settings.monitor_max_watches:
                raise ValueError("max watches")
            watch = Watch(id=uuid.uuid4().hex, url=url, label=label, model=model)
            self._watches[watch.id] = watch
        self.save()
        return watch

    def list(self) -> list[Watch]:
        return list(self._watches.values())

    def get(self, watch_id: str) -> Watch | None:
        return self._watches.get(watch_id)

    def delete(self, watch_id: str) -> bool:
        with self._lock:
            existed = self._watches.pop(watch_id, None) is not None
        if existed:
            self.save()
        return existed

    def ack(self, watch_id: str) -> Watch | None:
        watch = self._watches.get(watch_id)
        if watch is None:
            return None
        with self._lock:
            watch.alert_count = 0
        self.save()
        return watch

    # ---- classification ----------------------------------------------
    def classify(self, text: str, model_key: str = DEFAULT_MODEL) -> Comment:
        proba = None
        if model_key == DEFAULT_MODEL and self.phobert is not None:
            proba = self.phobert.try_proba(text)
        if proba is not None:
            model = DISPLAY_NAMES["PhoBERT"]  # "PhoBERT-base-v2"
        else:
            # sklearn path: the picked model, or LR when PhoBERT is unavailable
            key = model_key if model_key in SKLEARN_ORDER else "LogisticRegression"
            proba = [float(p) for p in self.registry.predict_proba(key, text)]
            model = DISPLAY_NAMES[key]
        proba = [float(p) for p in proba]
        label = int(np.argmax(proba))
        name = LABEL_NAMES[label]
        return Comment(text=text, label=label, label_name=name, proba=proba,
                       toxic=name in TOXIC, model=model, seen_at=_now(),
                       hash=_hash(text))

    # ---- scanning -----------------------------------------------------
    def scan(self, watch_id: str) -> Watch | None:
        watch = self._watches.get(watch_id)
        if watch is None:
            return None
        # Fetch is slow (network) — keep outside the lock
        try:
            raw = self._fetch(watch.url, max_len=getattr(
                self.settings, "max_text_len", 5000))
        except FetchError as e:
            with self._lock:
                watch.last_error = str(e)
                watch.last_scan = _now()
            self.save()
            return watch
        # Pre-filter using a snapshot of seen_hashes (race-ok; re-checked under lock)
        seen_snapshot = set(watch.seen_hashes)
        cap = self.settings.monitor_max_comments_per_scan
        new_texts = [t for t in raw if _hash(t) not in seen_snapshot][:cap]
        # Classify outside the lock (PhoBERT inference is slow)
        classified = [self.classify(text, watch.model) for text in new_texts]
        # Mutation block under lock
        with self._lock:
            for comment in classified:
                # Re-check: a concurrent scan may have added this hash already
                if comment.hash not in watch.seen_hashes:
                    watch.seen_hashes.add(comment.hash)
                    watch.comments.append(comment)
                    if comment.toxic:
                        watch.alert_count += 1
            # FIFO cap on stored comments
            max_c = self.settings.monitor_max_comments
            if len(watch.comments) > max_c:
                watch.comments = watch.comments[-max_c:]
            watch.last_error = None
            watch.last_scan = _now()
        self.save()
        return watch

    def scan_all(self) -> None:
        for watch_id in list(self._watches.keys()):
            self.scan(watch_id)

    # ---- persistence --------------------------------------------------
    def save(self) -> None:
        # Snapshot under _lock; file I/O is serialized by _io_lock (never nested)
        with self._lock:
            data = []
            for w in self._watches.values():
                d = asdict(w)
                d["seen_hashes"] = list(w.seen_hashes)
                data.append(d)
        path = self.settings.monitor_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with self._io_lock:
            tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp_path, path)

    def load(self) -> None:
        path = self.settings.monitor_state_path
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            watches: dict[str, Watch] = {}
            for d in data:
                comments = [Comment(**c) for c in d.get("comments", [])]
                watches[d["id"]] = Watch(
                    id=d["id"], url=d["url"], label=d.get("label"),
                    created_at=d.get("created_at", _now()),
                    last_scan=d.get("last_scan"), last_error=d.get("last_error"),
                    alert_count=d.get("alert_count", 0),
                    model=d.get("model", DEFAULT_MODEL), comments=comments,
                    seen_hashes=set(d.get("seen_hashes", [])),
                )
            self._watches = watches
        except Exception:
            return
