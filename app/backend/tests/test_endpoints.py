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


def test_rewrite_loop(client, monkeypatch):
    import app.backend.routers.rewrite as rw
    monkeypatch.setattr(rw, "rewrite_polite",
                        lambda text, api_key: "cảm ơn bạn nhiều nhé")
    r = client.post("/rewrite", json={"text": "mày ngu thế không hiểu gì"})
    assert r.status_code == 200
    body = r.json()
    assert body["rewritten"] == "cảm ơn bạn nhiều nhé"
    assert "label" in body["before"] and "label" in body["after"]


def test_rewrite_503_without_key(client):
    # default test env has no GEMINI_API_KEY -> graceful failure
    r = client.post("/rewrite", json={"text": "đồ ngốc"})
    assert r.status_code == 503


import io


def test_batch_scores_csv(client):
    csv = "free_text\ncảm ơn bạn nhiều nhé\nmày ngu thế không hiểu gì\n"
    files = {"file": ("c.csv", io.BytesIO(csv.encode("utf-8")), "text/csv")}
    r = client.post("/batch", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert set(body["counts"]) == {"CLEAN", "OFFENSIVE", "HATE"}
    assert len(body["rows"]) == 2


def test_batch_rejects_missing_column(client):
    csv = "wrong\nhello\n"
    files = {"file": ("c.csv", io.BytesIO(csv.encode("utf-8")), "text/csv")}
    r = client.post("/batch", files=files)
    assert r.status_code == 400


def test_batch_rejects_oversized_upload(artifacts_dir, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "100")
    from app.backend.config import get_settings
    get_settings.cache_clear()
    from app.backend.main import create_app
    app = create_app()
    big_csv = "free_text\n" + ("x" * 200) + "\n"
    files = {"file": ("big.csv", io.BytesIO(big_csv.encode("utf-8")), "text/csv")}
    with TestClient(app) as c:
        r = c.post("/batch", files=files)
    assert r.status_code == 413
    get_settings.cache_clear()


def test_insights(client):
    r = client.get("/insights")
    assert r.status_code == 200
    body = r.json()
    assert body["best"] == "PhoBERT-base-v2"
    assert len(body["models"]) == 7
    assert body["models"][0]["f1_macro"] == 0.6703
