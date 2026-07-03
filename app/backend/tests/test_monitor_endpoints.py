import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(artifacts_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.delenv("PHOBERT_REPO", raising=False)
    monkeypatch.setenv("MONITOR_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("MONITOR_INTERVAL_SEC", "3600")  # don't auto-fire in tests
    monkeypatch.setenv("MONITOR_MAX_WATCHES", "2")
    from app.backend.config import get_settings
    get_settings.cache_clear()
    from app.backend.main import create_app
    app = create_app()
    with TestClient(app) as c:
        # Deterministic fetch: no real network.
        app.state.monitor._fetch = lambda url, **kw: [
            "xin chào mọi người", "mày ngu thế không hiểu gì"]
        yield c
    get_settings.cache_clear()


def test_create_and_list_watch(client):
    r = client.post("/monitor/watches", json={"url": "https://ex.com/a",
                                              "label": "trang A"})
    assert r.status_code == 200
    wid = r.json()["id"]
    assert r.json()["url"] == "https://ex.com/a"
    r2 = client.get("/monitor/watches")
    assert any(w["id"] == wid for w in r2.json())


def test_scan_flags_toxic_and_ack(client):
    wid = client.post("/monitor/watches", json={"url": "https://ex.com/a"}).json()["id"]
    r = client.post(f"/monitor/watches/{wid}/scan")
    assert r.status_code == 200
    body = r.json()
    assert body["alert_count"] == 1
    assert len(body["comments"]) == 2
    toxic = [c for c in body["comments"] if c["toxic"]]
    assert toxic
    # toxic comments carry token-level explanation; each token has text + score
    assert toxic[0]["tokens"]
    assert {"token", "score"} <= set(toxic[0]["tokens"][0])
    # clean comments carry no explanation payload
    assert all(c["tokens"] == [] for c in body["comments"] if not c["toxic"])
    ack = client.post(f"/monitor/watches/{wid}/ack")
    assert ack.json()["alert_count"] == 0


def test_get_detail_and_delete(client):
    wid = client.post("/monitor/watches", json={"url": "https://ex.com/a"}).json()["id"]
    d = client.get(f"/monitor/watches/{wid}")
    assert d.status_code == 200
    assert d.json()["url"] == "https://ex.com/a"
    dele = client.delete(f"/monitor/watches/{wid}")
    assert dele.status_code == 200
    assert client.get(f"/monitor/watches/{wid}").status_code == 404


def test_max_watches_enforced(client):
    client.post("/monitor/watches", json={"url": "https://ex.com/1"})
    client.post("/monitor/watches", json={"url": "https://ex.com/2"})
    r = client.post("/monitor/watches", json={"url": "https://ex.com/3"})
    assert r.status_code == 400


def test_rejects_bad_url(client):
    r = client.post("/monitor/watches", json={"url": "not-a-url"})
    assert r.status_code in (400, 422)
