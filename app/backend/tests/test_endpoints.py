import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(artifacts_dir, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    from app.backend.config import get_settings
    get_settings.cache_clear()
    from app.backend.main import create_app
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["sklearn_loaded"] is True


def test_predict_shape(client):
    r = client.post("/predict", json={"text": "mày ngu thế không hiểu gì"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in (0, 1, 2)
    assert body["label_name"] in ("CLEAN", "OFFENSIVE", "HATE")
    assert len(body["proba"]) == 3
    assert len(body["tokens"]) >= 1


def test_predict_rejects_empty(client):
    r = client.post("/predict", json={"text": "   "})
    assert r.status_code == 422 or r.status_code == 400


def test_showdown_returns_six_without_phobert(client):
    r = client.post("/showdown", json={"text": "cảm ơn bạn nhiều nhé"})
    assert r.status_code == 200
    names = [m["name"] for m in r.json()["models"]]
    assert "PhoBERT" not in names
    assert len(names) == 6


def test_showdown_includes_phobert_when_available(artifacts_dir, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    from app.backend.config import get_settings
    get_settings.cache_clear()
    from app.backend.main import create_app
    app = create_app()

    class FakePhoBert:
        available = True
        def predict_proba(self, text):
            return [0.1, 0.2, 0.7]

    with TestClient(app) as c:
        app.state.phobert = FakePhoBert()
        r = c.post("/showdown", json={"text": "diệt sạch bọn chúng đi"})
        names = [m["name"] for m in r.json()["models"]]
        assert names[0] == "PhoBERT"
        assert len(names) == 7
        r2 = c.post("/predict", json={"text": "diệt sạch bọn chúng đi"})
        assert r2.json()["model"] == "PhoBERT-base-v2"
        assert r2.json()["label"] == 2
