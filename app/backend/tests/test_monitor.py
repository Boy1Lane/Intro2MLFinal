import json

import numpy as np
import pytest

from app.backend.services.monitor import MonitorService


class FakeRegistry:
    def predict_proba(self, model_key, text):
        # toxic if text contains "ngu"/"diệt", else clean
        if "diệt" in text:
            return np.array([0.1, 0.2, 0.7])   # HATE
        if "ngu" in text:
            return np.array([0.2, 0.7, 0.1])   # OFFENSIVE
        return np.array([0.8, 0.1, 0.1])       # CLEAN


class Settings:
    monitor_max_comments_per_scan = 30
    monitor_max_comments = 200
    monitor_max_watches = 3

    def __init__(self, tmp_path):
        self.monitor_state_path = tmp_path / "state.json"


def make_service(tmp_path, comments, phobert=None):
    calls = {"n": 0}

    def fake_fetch(url, **kw):
        calls["n"] += 1
        return comments

    svc = MonitorService(FakeRegistry(), phobert, Settings(tmp_path),
                         fetch=fake_fetch)
    return svc, calls


def test_scan_flags_toxic_and_dedups(tmp_path):
    svc, calls = make_service(tmp_path, ["xin chào các bạn", "mày ngu quá"])
    w = svc.add("https://ex.com/a")
    svc.scan(w.id)
    w = svc.get(w.id)
    assert len(w.comments) == 2
    assert w.alert_count == 1  # only the toxic one
    # re-scan same comments: no new alerts (dedup by hash)
    svc.scan(w.id)
    w = svc.get(w.id)
    assert w.alert_count == 1
    assert len(w.comments) == 2


def test_ack_resets_alert(tmp_path):
    svc, _ = make_service(tmp_path, ["diệt sạch bọn chúng"])
    w = svc.add("https://ex.com/a")
    svc.scan(w.id)
    assert svc.get(w.id).alert_count == 1
    svc.ack(w.id)
    assert svc.get(w.id).alert_count == 0


def test_per_scan_cap(tmp_path):
    comments = [f"ngu số {i}" for i in range(50)]
    svc, _ = make_service(tmp_path, comments)
    svc.settings.monitor_max_comments_per_scan = 10
    w = svc.add("https://ex.com/a")
    svc.scan(w.id)
    assert len(svc.get(w.id).comments) == 10


def test_max_watches(tmp_path):
    svc, _ = make_service(tmp_path, [])
    svc.add("https://ex.com/1")
    svc.add("https://ex.com/2")
    svc.add("https://ex.com/3")
    with pytest.raises(ValueError):
        svc.add("https://ex.com/4")


def test_fetch_error_sets_last_error(tmp_path):
    from app.backend.services.fetcher import FetchError

    def bad_fetch(url, **kw):
        raise FetchError("boom")

    svc = MonitorService(FakeRegistry(), None, Settings(tmp_path), fetch=bad_fetch)
    w = svc.add("https://ex.com/a")
    svc.scan(w.id)  # must not raise
    w = svc.get(w.id)
    assert w.last_error == "boom"
    assert w.last_scan is not None


def test_classify_prefers_phobert(tmp_path):
    class FakePhoBert:
        available = True
        def try_proba(self, text):
            return [0.05, 0.05, 0.9]

    svc, _ = make_service(tmp_path, [], phobert=FakePhoBert())
    c = svc.classify("bất kỳ")
    assert c.model == "PhoBERT-base-v2"
    assert c.label == 2


def test_classify_falls_back_to_sklearn(tmp_path):
    class NoPhoBert:
        available = False
        def try_proba(self, text):
            return None

    svc, _ = make_service(tmp_path, [], phobert=NoPhoBert())
    c = svc.classify("mày ngu")
    assert c.model == "Logistic Regression"
    assert c.label_name == "OFFENSIVE"


def test_persistence_round_trip(tmp_path):
    svc, _ = make_service(tmp_path, ["mày ngu quá"])
    w = svc.add("https://ex.com/a", label="test")
    svc.scan(w.id)
    svc.save()

    svc2 = MonitorService(FakeRegistry(), None, Settings(tmp_path))
    svc2.load()
    w2 = svc2.get(w.id)
    assert w2 is not None
    assert w2.url == "https://ex.com/a"
    assert w2.label == "test"
    assert len(w2.comments) == 1
    assert w2.alert_count == 1
    assert isinstance(w2.seen_hashes, set)


def test_load_survives_corrupt_file(tmp_path):
    settings = Settings(tmp_path)
    settings.monitor_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.monitor_state_path.write_text("{not json", encoding="utf-8")
    svc = MonitorService(FakeRegistry(), None, settings)
    svc.load()  # must not raise
    assert svc.list() == []


def test_load_survives_bad_shape(tmp_path):
    settings = Settings(tmp_path)
    settings.monitor_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.monitor_state_path.write_text('[{"nope": 1}]', encoding="utf-8")
    svc = MonitorService(FakeRegistry(), None, settings)
    svc.load()  # must not raise
    assert svc.list() == []


def test_save_is_atomic_no_partial_file(tmp_path):
    svc, _ = make_service(tmp_path, [])
    svc.add("https://ex.com/a")
    svc.save()
    path = svc.settings.monitor_state_path
    content = path.read_text(encoding="utf-8")
    json.loads(content)  # must not raise — file must be valid JSON
    tmp_file = path.with_suffix(path.suffix + ".tmp")
    assert not tmp_file.exists()
