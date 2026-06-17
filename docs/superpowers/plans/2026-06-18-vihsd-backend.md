# ViHSD Backend (FastAPI) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the FastAPI backend that serves the 7 ViHSD models (6 sklearn + PhoBERT), explainability, Gemini rewrite, batch CSV scoring, and static insights — defining the HTTP contract the Next.js frontend will consume.

**Architecture:** A `ml/` package holds shared preprocessing (ported verbatim from the notebooks). A `services/` layer loads artifacts and runs inference; thin FastAPI routers expose endpoints. sklearn models load at startup; PhoBERT lazy-loads on first use. Gemini is proxied server-side so the key never reaches the browser.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, pydantic v2, scikit-learn, joblib, numpy, torch + transformers (PhoBERT), google-genai (Gemini), pytest.

## Global Constraints

- Python version floor: **3.11**.
- Labels are fixed: **0=CLEAN, 1=OFFENSIVE, 2=HATE** (`LABELS = [0, 1, 2]`, `LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]`).
- The 7 models served MUST be exactly those compared in the report: PhoBERT-base-v2, Random Forest, Multinomial NB, Voting Ensemble, Logistic Regression, Linear SVC (calibrated), SGD Classifier.
- PhoBERT base = `vinai/phobert-base-v2`, tokenizer `max_length=256`, `padding="max_length"`, `truncation=True`.
- Preprocessing parity is mandatory: sklearn input = `normalize_teencode(preprocess_text(raw), TEENCODE_ML)`; PhoBERT input = `normalize_teencode(preprocess_text(raw), TEENCODE_PHOBERT)`. The two teencode dicts differ and are both ported verbatim from the notebooks.
- All response probabilities are length-3 lists ordered `[CLEAN, OFFENSIVE, HATE]` summing to ~1.0.
- Secrets via env only: `GEMINI_API_KEY`, `PHOBERT_REPO` (HF Hub id), `ARTIFACTS_DIR`, `CORS_ORIGINS`. Never hardcode.
- All paths in this plan are relative to repo root `/home/lesliu/Documents/school/25_26_Semester_2/intro2ml/final`.

## Artifacts contract (produced by user on Colab — NOT this plan)

The backend reads, from `$ARTIFACTS_DIR` (default `artifacts/`):
```
artifacts/
  tfidf_vectorizer.pkl     # TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)
  bow_vectorizer.pkl       # CountVectorizer(max_features=10000, ngram_range=(1,2))
  tfidf_svd.pkl            # TruncatedSVD fitted on raw TF-IDF
  models/model_lr.pkl      # LogisticRegression       (features: tfidf)
  models/model_svm.pkl     # CalibratedClassifierCV(LinearSVC) (features: tfidf)
  models/model_sgd.pkl     # SGDClassifier(loss=log_loss)      (features: tfidf)
  models/model_nb.pkl      # MultinomialNB             (features: bow)
  models/model_rf.pkl      # RandomForestClassifier    (features: tfidf -> svd)
  models/model_voting.pkl  # dict {"lr","svm","sgd"}   (features: tfidf, proba=mean)
```
PhoBERT loads from HF Hub repo `$PHOBERT_REPO` (a fine-tuned `vinai/phobert-base-v2`, 3 labels). Tests never touch real artifacts — `conftest.py` builds tiny synthetic ones.

## File Structure

```
app/backend/
  pyproject.toml / requirements.txt   # deps
  config.py                # Settings (env-driven)
  constants.py             # LABELS, LABEL_NAMES, MODEL_ORDER
  schemas.py               # pydantic request/response models
  main.py                  # FastAPI app, CORS, router wiring, startup load
  services/
    sklearn_registry.py    # load vectorizers+models, per-model feature routing, proba
    explain.py             # token contributions via LR
    phobert.py             # lazy-load PhoBERT, proba
    gemini.py              # Gemini rewrite client
    metrics.py             # load static insights json
  routers/
    predict.py             # POST /predict, POST /showdown
    rewrite.py             # POST /rewrite
    batch.py               # POST /batch
    insights.py            # GET /insights, GET /health
  tests/
    conftest.py            # synthetic artifacts + TestClient fixtures
    test_preprocessing.py
    test_sklearn_registry.py
    test_explain.py
    test_endpoints.py
app/ml/
  __init__.py
  teencode.py              # TEENCODE_ML, TEENCODE_PHOBERT, normalize_teencode
  preprocessing.py         # remove_* fns, preprocess_text
```

---

### Task 1: Shared preprocessing package (`app/ml/`)

**Files:**
- Create: `app/ml/__init__.py`
- Create: `app/ml/teencode.py`
- Create: `app/ml/preprocessing.py`
- Test: `app/backend/tests/test_preprocessing.py`

**Interfaces:**
- Produces: `preprocess_text(text: str) -> str`; `normalize_teencode(text: str, mapping: dict[str,str]) -> str`; `TEENCODE_ML: dict[str,str]`; `TEENCODE_PHOBERT: dict[str,str]`.

- [ ] **Step 1: Write the failing test**

Create `app/backend/tests/test_preprocessing.py`:
```python
from app.ml.preprocessing import preprocess_text
from app.ml.teencode import normalize_teencode, TEENCODE_ML, TEENCODE_PHOBERT


def test_preprocess_lowercases_and_strips_url_emoji():
    out = preprocess_text("Xem tại https://abc.com nhé 😀😀😀")
    assert "http" not in out
    assert "😀" not in out
    assert out == out.lower()


def test_preprocess_handles_none_and_empty():
    assert preprocess_text(None) == ""
    assert preprocess_text("   ") == ""


def test_preprocess_collapses_repeated_chars():
    assert preprocess_text("đẹppppp") == "đẹpp"


def test_teencode_ml_replaces_known_tokens():
    assert normalize_teencode("ko biet", TEENCODE_ML) == "không biết"


def test_teencode_phobert_has_profanity_mapping():
    assert TEENCODE_PHOBERT["vcl"] == "vãi lồn"
    assert normalize_teencode("vcl", TEENCODE_PHOBERT) == "vãi lồn"


def test_two_dicts_differ():
    assert TEENCODE_ML != TEENCODE_PHOBERT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lesliu/Documents/school/25_26_Semester_2/intro2ml/final && python -m pytest app/backend/tests/test_preprocessing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ml.preprocessing'`

- [ ] **Step 3: Create `app/ml/__init__.py` (empty) and `app/ml/teencode.py`**

Create `app/ml/__init__.py` empty. Create `app/ml/teencode.py` — port both dicts verbatim:
```python
# TEENCODE_ML: ported from notebooks/02_text_preprocessing.ipynb + 06_model_training.ipynb
TEENCODE_ML = {
    "ko": "không", "k": "không", "kh": "không", "khg": "không", "kp": "không phải",
    "kq": "không quan", "dc": "được", "đc": "được", "dk": "được", "đk": "được",
    "nc": "nước", "ng": "người", "ns": "nói", "mk": "mình", "mn": "mọi người",
    "mng": "mọi người", "bn": "bạn", "b": "bạn", "bro": "bạn", "ib": "nhắn tin",
    "rep": "trả lời", "vs": "với", "v": "với", "voi": "với", "r": "rồi",
    "rui": "rồi", "rii": "rồi", "ntn": "như thế nào", "j": "gì", "ji": "gì",
    "z": "gì", "gi": "gì", "a": "anh", "e": "em", "c": "chị", "đi": "đi",
    "qua": "qua", "trc": "trước", "tg": "thời gian", "bt": "bình thường",
    "bth": "bình thường", "vl": "vãi", "vkl": "vãi", "nch": "nói chuyện",
    "nt": "nhắn tin", "hk": "không", "hem": "không", "bi": "bị", "bik": "biết",
    "ck": "chồng", "vk": "vợ", "tks": "thanks", "thanks": "cảm ơn",
    "thks": "cảm ơn", "ok": "được", "okie": "được", "oke": "được",
    "plz": "làm ơn", "pls": "làm ơn", "sr": "xin lỗi", "sorry": "xin lỗi",
    "lun": "luôn", "lm": "làm", "đag": "đang", "dg": "đang", "trg": "trong",
    "trog": "trong", "cx": "cũng", "cg": "cũng", "đb": "đặc biệt",
    "cb": "chuẩn bị", "h": "giờ", "hm": "hôm", "dt": "điện thoại",
    "sdt": "số điện thoại", "fb": "facebook", "yt": "youtube", "ad": "admin",
    "mod": "moderator", "nx": "nhận xét", "đt": "điện thoại",
    "gato": "ghen ăn tức ở", "wtf": "what the f", "dm": "đ mẹ", "vcl": "vãi",
    "clgt": "chắc luôn", "oy": "rồi", "ùi": "rồi", "biet": "biết", "hiu": "hiểu",
    "thik": "thích", "hjhj": "hihi", "tui": "tôi", "mik": "mình",
    "ngta": "người ta", "nyc": "người yêu cũ", "ny": "người yêu", "gf": "bạn gái",
    "bf": "bạn trai", "sg": "sài gòn", "hn": "hà nội", "vn": "việt nam",
    "nhma": "nhưng mà", "nma": "nhưng mà", "tl": "trả lời", "cmn": "con mẹ nó",
}

# TEENCODE_PHOBERT: ported from notebooks/06_model_training_DL.ipynb (profanity-aware)
TEENCODE_PHOBERT = {
    "dell": "đéo", "del": "đéo", "đell": "đéo", "đel": "đéo", "loz": "lồn",
    "lon": "lồn", "lòn": "lồn", "l": "lồn", "coin card": "củ cặc", "cc": "củ cặc",
    "cức": "cứt", "ms": "mới", "bh": "bây giờ", "kb": "không biết", "kk": "cười",
    "haha": "cười", "đhs": "đéo hiểu sao", "dm": "địt mẹ", "đm": "địt mẹ",
    "dmm": "địt mẹ mày", "vcl": "vãi lồn", "vl": "vãi lồn", "vkl": "vãi lồn",
    "cl": "cái lồn", "clgt": "cái lồn gì thế", "đcm": "địt con mẹ",
    "dcm": "địt con mẹ",
}


def normalize_teencode(text: str, mapping: dict) -> str:
    words = str(text).split()
    return " ".join(mapping.get(w.lower(), w) for w in words)
```

- [ ] **Step 4: Create `app/ml/preprocessing.py`** — port verbatim from `notebooks/02_text_preprocessing.ipynb`:
```python
import re


def remove_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_emails(text):
    return re.sub(r"\S+@\S+\.\S+", " ", text)


def remove_phone_numbers(text):
    return re.sub(r"(\+84|0)\d{9,10}", " ", text)


def remove_html_tags(text):
    return re.sub(r"<[^>]+>", " ", text)


def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f1e0-\U0001f1ff"
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "♀-♂"
        "☀-⭕"
        "‍"
        "⏏"
        "⏩"
        "⌚"
        "️"
        "〰"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(" ", text)


def normalize_repeated_chars(text):
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def remove_special_characters(text):
    return re.sub(
        r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]",
        " ",
        text,
        flags=re.IGNORECASE,
    )


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text):
    if text is None or not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_phone_numbers(text)
    text = remove_emojis(text)
    text = normalize_repeated_chars(text)
    text = remove_special_characters(text)
    return normalize_whitespace(text)
```

Note: notebook's `preprocess_text` ran `replace_teencode` internally, but the saved `free_text_clean` is then re-run through `normalize_teencode` in both training notebooks. We therefore keep teencode OUT of `preprocess_text` and apply it explicitly per-family (see registry/phobert tasks). Net token output is identical because teencode replacement is idempotent on already-normalized words.

- [ ] **Step 5: Add pytest config so `app.` imports resolve**

Create `app/backend/requirements.txt`:
```
fastapi==0.115.*
uvicorn[standard]==0.32.*
pydantic==2.*
scikit-learn==1.5.*
joblib==1.4.*
numpy==1.26.*
pandas==2.2.*
python-multipart==0.0.*
torch==2.4.*
transformers==4.44.*
google-genai==0.3.*
pytest==8.*
httpx==0.27.*
```

Create `pytest.ini` at repo root:
```ini
[pytest]
pythonpath = .
testpaths = app/backend/tests
```
And create `app/__init__.py` (empty) so `app.ml` / `app.backend` are importable packages.

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest app/backend/tests/test_preprocessing.py -v`
Expected: PASS (6 passed)

- [ ] **Step 7: Commit**

```bash
git add app/ml app/__init__.py app/backend/requirements.txt app/backend/tests/test_preprocessing.py pytest.ini
git commit -m "feat(backend): shared preprocessing + teencode dicts"
```

---

### Task 2: Constants, config, sklearn registry

**Files:**
- Create: `app/backend/__init__.py`, `app/backend/constants.py`, `app/backend/config.py`
- Create: `app/backend/services/__init__.py`, `app/backend/services/sklearn_registry.py`
- Create: `app/backend/tests/conftest.py`
- Test: `app/backend/tests/test_sklearn_registry.py`

**Interfaces:**
- Consumes: `preprocess_text`, `normalize_teencode`, `TEENCODE_ML` from Task 1.
- Produces:
  - `constants.LABELS = [0,1,2]`, `constants.LABEL_NAMES = ["CLEAN","OFFENSIVE","HATE"]`, `constants.SKLEARN_ORDER = ["LogisticRegression","LinearSVC","MultinomialNB","RandomForest","SGDClassifier","VotingEnsemble"]`.
  - `config.Settings` with `.artifacts_dir: Path`, `.phobert_repo: str`, `.gemini_api_key: str|None`, `.cors_origins: list[str]`, `.max_text_len: int = 5000`, `.max_batch_rows: int = 5000`; `config.get_settings() -> Settings` (cached).
  - `sklearn_registry.SklearnRegistry` with `.load()`, `.predict_proba(name: str, raw_text: str) -> list[float]` (len 3), `.predict_all(raw_text: str) -> list[dict]` returning `[{"name","label","proba","latency_ms"}]` for the 6 sklearn models in `SKLEARN_ORDER`. Display names map to report names via `constants.DISPLAY_NAMES`.

- [ ] **Step 1: Write `conftest.py` that builds synthetic artifacts**

Create `app/backend/tests/conftest.py`:
```python
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Tiny labelled corpus covering all 3 classes (CLEAN=0, OFFENSIVE=1, HATE=2)
TEXTS = [
    "hôm nay trời đẹp tôi rất vui", "cảm ơn bạn nhiều nhé", "bài viết hay quá",
    "chúc mọi người ngày mới tốt lành", "món ăn này ngon tuyệt vời",
    "mày ngu thế không hiểu gì", "đồ ngốc nói chuyện vô duyên", "im đi đồ phiền phức",
    "thằng này nói nhảm quá", "câm mồm lại đi",
    "bọn mày đáng bị tiêu diệt hết", "lũ súc vật ghê tởm", "diệt sạch bọn chúng đi",
    "đám người đó nên biến mất", "ghét cay ghét đắng lũ khốn",
]
LABELS = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("artifacts")
    (d / "models").mkdir()
    X, y = TEXTS, np.array(LABELS)

    tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    Xt = tfidf.fit_transform(X)
    bow = CountVectorizer(ngram_range=(1, 2))
    Xb = bow.fit_transform(X)
    svd = TruncatedSVD(n_components=5, random_state=0)
    Xs = svd.fit_transform(Xt)

    joblib.dump(tfidf, d / "tfidf_vectorizer.pkl")
    joblib.dump(bow, d / "bow_vectorizer.pkl")
    joblib.dump(svd, d / "tfidf_svd.pkl")

    lr = LogisticRegression(max_iter=1000).fit(Xt, y)
    svm = CalibratedClassifierCV(LinearSVC(), cv=3).fit(Xt, y)
    sgd = SGDClassifier(loss="log_loss", random_state=0).fit(Xt, y)
    nb = MultinomialNB().fit(Xb, y)
    rf = RandomForestClassifier(random_state=0).fit(Xs, y)

    joblib.dump(lr, d / "models" / "model_lr.pkl")
    joblib.dump(svm, d / "models" / "model_svm.pkl")
    joblib.dump(sgd, d / "models" / "model_sgd.pkl")
    joblib.dump(nb, d / "models" / "model_nb.pkl")
    joblib.dump(rf, d / "models" / "model_rf.pkl")
    joblib.dump({"lr": lr, "svm": svm, "sgd": sgd}, d / "models" / "model_voting.pkl")
    return d
```

- [ ] **Step 2: Write the failing test**

Create `app/backend/tests/test_sklearn_registry.py`:
```python
from app.backend.services.sklearn_registry import SklearnRegistry
from app.backend.constants import SKLEARN_ORDER


def test_registry_loads_and_predicts_all(artifacts_dir):
    reg = SklearnRegistry(artifacts_dir)
    reg.load()
    results = reg.predict_all("mày ngu thế không hiểu gì")
    assert [r["name"] for r in results] == SKLEARN_ORDER
    for r in results:
        assert len(r["proba"]) == 3
        assert abs(sum(r["proba"]) - 1.0) < 1e-3
        assert r["label"] in (0, 1, 2)
        assert r["latency_ms"] >= 0


def test_voting_proba_is_mean_of_members(artifacts_dir):
    reg = SklearnRegistry(artifacts_dir)
    reg.load()
    text = "cảm ơn bạn nhiều nhé"
    lr = reg.predict_proba("LogisticRegression", text)
    svm = reg.predict_proba("LinearSVC", text)
    sgd = reg.predict_proba("SGDClassifier", text)
    voting = reg.predict_proba("VotingEnsemble", text)
    expected = [(a + b + c) / 3 for a, b, c in zip(lr, svm, sgd)]
    assert max(abs(v - e) for v, e in zip(voting, expected)) < 1e-6
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest app/backend/tests/test_sklearn_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.backend.services.sklearn_registry'`

- [ ] **Step 4: Create constants + config**

Create `app/backend/__init__.py` (empty). Create `app/backend/constants.py`:
```python
LABELS = [0, 1, 2]
LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]

# internal sklearn keys, ordered for showdown
SKLEARN_ORDER = [
    "LogisticRegression", "LinearSVC", "MultinomialNB",
    "RandomForest", "SGDClassifier", "VotingEnsemble",
]
# full showdown order incl. PhoBERT (best first, matching report table)
MODEL_ORDER = ["PhoBERT"] + SKLEARN_ORDER

# human-facing names matching the report metric table
DISPLAY_NAMES = {
    "PhoBERT": "PhoBERT-base-v2",
    "LogisticRegression": "Logistic Regression",
    "LinearSVC": "Linear SVC",
    "MultinomialNB": "Multinomial NB",
    "RandomForest": "Random Forest",
    "SGDClassifier": "SGD Classifier",
    "VotingEnsemble": "Voting Ensemble",
}
```

Create `app/backend/config.py`:
```python
import os
from functools import lru_cache
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        self.artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
        self.phobert_repo = os.getenv("PHOBERT_REPO", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or None
        origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
        self.cors_origins = [o.strip() for o in origins.split(",") if o.strip()]
        self.max_text_len = int(os.getenv("MAX_TEXT_LEN", "5000"))
        self.max_batch_rows = int(os.getenv("MAX_BATCH_ROWS", "5000"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 5: Create the registry**

Create `app/backend/services/__init__.py` (empty). Create `app/backend/services/sklearn_registry.py`:
```python
import time
from pathlib import Path

import joblib
import numpy as np

from app.backend.constants import SKLEARN_ORDER
from app.ml.preprocessing import preprocess_text
from app.ml.teencode import TEENCODE_ML, normalize_teencode

# feature routing per model key
_FEATURES = {
    "LogisticRegression": "tfidf",
    "LinearSVC": "tfidf",
    "SGDClassifier": "tfidf",
    "VotingEnsemble": "tfidf",
    "MultinomialNB": "bow",
    "RandomForest": "svd",
}
_FILES = {
    "LogisticRegression": "model_lr.pkl",
    "LinearSVC": "model_svm.pkl",
    "SGDClassifier": "model_sgd.pkl",
    "MultinomialNB": "model_nb.pkl",
    "RandomForest": "model_rf.pkl",
    "VotingEnsemble": "model_voting.pkl",
}


def clean_for_sklearn(raw: str) -> str:
    return normalize_teencode(preprocess_text(raw), TEENCODE_ML)


class SklearnRegistry:
    def __init__(self, artifacts_dir: Path):
        self.dir = Path(artifacts_dir)
        self.tfidf = None
        self.bow = None
        self.svd = None
        self.models: dict = {}

    def load(self) -> None:
        self.tfidf = joblib.load(self.dir / "tfidf_vectorizer.pkl")
        self.bow = joblib.load(self.dir / "bow_vectorizer.pkl")
        self.svd = joblib.load(self.dir / "tfidf_svd.pkl")
        for key, fname in _FILES.items():
            self.models[key] = joblib.load(self.dir / "models" / fname)

    def _features(self, kind: str, cleaned: str):
        if kind == "tfidf":
            return self.tfidf.transform([cleaned])
        if kind == "bow":
            return self.bow.transform([cleaned])
        if kind == "svd":
            return self.svd.transform(self.tfidf.transform([cleaned]))
        raise ValueError(kind)

    def predict_proba(self, name: str, raw_text: str) -> list[float]:
        cleaned = clean_for_sklearn(raw_text)
        if name == "VotingEnsemble":
            X = self._features("tfidf", cleaned)
            members = self.models["VotingEnsemble"]
            probas = [m.predict_proba(X)[0] for m in members.values()]
            return list(np.mean(probas, axis=0))
        X = self._features(_FEATURES[name], cleaned)
        return list(self.models[name].predict_proba(X)[0])

    def predict_all(self, raw_text: str) -> list[dict]:
        out = []
        for name in SKLEARN_ORDER:
            t0 = time.perf_counter()
            proba = [float(p) for p in self.predict_proba(name, raw_text)]
            dt = (time.perf_counter() - t0) * 1000
            out.append({
                "name": name,
                "label": int(np.argmax(proba)),
                "proba": proba,
                "latency_ms": round(dt, 2),
            })
        return out
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_sklearn_registry.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Commit**

```bash
git add app/backend/__init__.py app/backend/constants.py app/backend/config.py app/backend/services app/backend/tests/conftest.py app/backend/tests/test_sklearn_registry.py
git commit -m "feat(backend): sklearn registry with per-model feature routing"
```

---

### Task 3: Explainability service (LR token contributions)

**Files:**
- Create: `app/backend/services/explain.py`
- Test: `app/backend/tests/test_explain.py`

**Interfaces:**
- Consumes: `SklearnRegistry` (uses `.tfidf` vectorizer and `.models["LogisticRegression"]`), `clean_for_sklearn`.
- Produces: `explain_tokens(reg: SklearnRegistry, raw_text: str, pred_label: int) -> list[dict]` → `[{"token": str, "score": float}]`, one entry per whitespace token of the cleaned text, score = signed contribution toward `pred_label` (positive = pushes toward that label). Unknown tokens get score 0.0.

- [ ] **Step 1: Write the failing test**

Create `app/backend/tests/test_explain.py`:
```python
import numpy as np

from app.backend.services.explain import explain_tokens
from app.backend.services.sklearn_registry import SklearnRegistry, clean_for_sklearn


def test_explain_returns_one_entry_per_cleaned_token(artifacts_dir):
    reg = SklearnRegistry(artifacts_dir)
    reg.load()
    raw = "mày ngu thế không hiểu gì"
    cleaned = clean_for_sklearn(raw)
    proba = reg.predict_proba("LogisticRegression", raw)
    label = int(np.argmax(proba))
    toks = explain_tokens(reg, raw, label)
    assert [t["token"] for t in toks] == cleaned.split()
    assert all(isinstance(t["score"], float) for t in toks)


def test_explain_unknown_token_scores_zero(artifacts_dir):
    reg = SklearnRegistry(artifacts_dir)
    reg.load()
    toks = explain_tokens(reg, "zzzqqq", 0)
    assert toks == [] or toks[0]["score"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest app/backend/tests/test_explain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.backend.services.explain'`

- [ ] **Step 3: Implement**

Create `app/backend/services/explain.py`:
```python
from app.backend.services.sklearn_registry import SklearnRegistry, clean_for_sklearn


def explain_tokens(reg: SklearnRegistry, raw_text: str, pred_label: int) -> list[dict]:
    cleaned = clean_for_sklearn(raw_text)
    tokens = cleaned.split()
    if not tokens:
        return []
    lr = reg.models["LogisticRegression"]
    vocab = reg.tfidf.vocabulary_       # term -> column index
    idf = reg.tfidf.idf_
    # binary multiclass coef shape (3, n_features); pick predicted class row
    coef_row = lr.coef_[pred_label]
    out = []
    for tok in tokens:
        col = vocab.get(tok)
        if col is None:
            out.append({"token": tok, "score": 0.0})
        else:
            # contribution proxy = idf weight * class coefficient for the unigram
            out.append({"token": tok, "score": float(idf[col] * coef_row[col])})
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_explain.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/backend/services/explain.py app/backend/tests/test_explain.py
git commit -m "feat(backend): LR-based token explainability"
```

---

### Task 4: Schemas + `/predict` and `/showdown` (sklearn-only, PhoBERT pluggable)

**Files:**
- Create: `app/backend/schemas.py`, `app/backend/routers/__init__.py`, `app/backend/routers/predict.py`, `app/backend/main.py`
- Test: `app/backend/tests/test_endpoints.py`

**Interfaces:**
- Consumes: `SklearnRegistry.predict_proba/predict_all`, `explain_tokens`. PhoBERT is accessed via `app.state.phobert` which exposes `.predict_proba(raw_text) -> list[float]` and `.available -> bool`; in this task it is `None` (showdown returns 6 models, predict falls back to best sklearn).
- Produces HTTP contract:
  - `POST /predict {text}` → `PredictResponse {label:int, label_name:str, proba:[f,f,f], tokens:[{token,score}], model:str}`
  - `POST /showdown {text}` → `ShowdownResponse {models:[{name, display_name, label, proba, latency_ms}]}`

- [ ] **Step 1: Write the failing test**

Create `app/backend/tests/test_endpoints.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest app/backend/tests/test_endpoints.py -v`
Expected: FAIL with import error for `app.backend.main`

- [ ] **Step 3: Create schemas**

Create `app/backend/schemas.py`:
```python
from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(min_length=1)


class TokenScore(BaseModel):
    token: str
    score: float


class PredictResponse(BaseModel):
    label: int
    label_name: str
    proba: list[float]
    tokens: list[TokenScore]
    model: str


class ModelResult(BaseModel):
    name: str
    display_name: str
    label: int
    proba: list[float]
    latency_ms: float


class ShowdownResponse(BaseModel):
    models: list[ModelResult]
```

- [ ] **Step 4: Create the predict router**

Create `app/backend/routers/__init__.py` (empty). Create `app/backend/routers/predict.py`:
```python
import time

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from app.backend.constants import DISPLAY_NAMES, LABEL_NAMES, MODEL_ORDER
from app.backend.schemas import PredictResponse, ShowdownResponse, TextRequest
from app.backend.services.explain import explain_tokens

router = APIRouter()


def _validate(req: TextRequest, max_len: int) -> str:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text rỗng.")
    if len(text) > max_len:
        raise HTTPException(status_code=400, detail=f"Text quá dài (>{max_len}).")
    return text


@router.post("/predict", response_model=PredictResponse)
def predict(req: TextRequest, request: Request) -> PredictResponse:
    text = _validate(req, request.app.state.settings.max_text_len)
    reg = request.app.state.registry
    phobert = request.app.state.phobert
    if phobert is not None and phobert.available:
        proba = [float(p) for p in phobert.predict_proba(text)]
        model = "PhoBERT-base-v2"
    else:
        proba = [float(p) for p in reg.predict_proba("LogisticRegression", text)]
        model = "Logistic Regression"
    label = int(np.argmax(proba))
    tokens = explain_tokens(reg, text, label)
    return PredictResponse(
        label=label, label_name=LABEL_NAMES[label], proba=proba,
        tokens=tokens, model=model,
    )


@router.post("/showdown", response_model=ShowdownResponse)
def showdown(req: TextRequest, request: Request) -> ShowdownResponse:
    text = _validate(req, request.app.state.settings.max_text_len)
    reg = request.app.state.registry
    phobert = request.app.state.phobert
    results = []
    if phobert is not None and phobert.available:
        t0 = time.perf_counter()
        proba = [float(p) for p in phobert.predict_proba(text)]
        dt = (time.perf_counter() - t0) * 1000
        results.append({
            "name": "PhoBERT", "label": int(np.argmax(proba)),
            "proba": proba, "latency_ms": round(dt, 2),
        })
    results.extend(reg.predict_all(text))
    # order by MODEL_ORDER
    order = {n: i for i, n in enumerate(MODEL_ORDER)}
    results.sort(key=lambda r: order[r["name"]])
    return ShowdownResponse(models=[
        {**r, "display_name": DISPLAY_NAMES[r["name"]]} for r in results
    ])
```

- [ ] **Step 5: Create the app factory**

Create `app/backend/main.py`:
```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.config import get_settings
from app.backend.routers import predict
from app.backend.services.sklearn_registry import SklearnRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.registry = SklearnRegistry(settings.artifacts_dir)
    app.state.registry.load()
    app.state.phobert = None  # wired in Task 6
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="ViHSD Moderation Studio API", lifespan=lifespan)
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(predict.router)

    @app.get("/health")
    def health():
        return {
            "sklearn_loaded": bool(getattr(app.state, "registry", None)
                                   and app.state.registry.models),
            "phobert_available": bool(getattr(app.state, "phobert", None)
                                      and app.state.phobert.available),
        }

    return app


app = create_app()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_endpoints.py -v`
Expected: PASS (4 passed)

- [ ] **Step 7: Commit**

```bash
git add app/backend/schemas.py app/backend/routers app/backend/main.py app/backend/tests/test_endpoints.py
git commit -m "feat(backend): /predict and /showdown endpoints"
```

---

### Task 5: PhoBERT service (lazy-load) + wiring

**Files:**
- Create: `app/backend/services/phobert.py`
- Modify: `app/backend/main.py` (set `app.state.phobert`)
- Modify: `app/backend/tests/test_endpoints.py` (add a stubbed-PhoBERT test)

**Interfaces:**
- Consumes: `preprocess_text`, `normalize_teencode`, `TEENCODE_PHOBERT`, `config.Settings`.
- Produces: `PhoBertService(repo: str)` with `.available: bool`, `.predict_proba(raw_text: str) -> list[float]` (lazy-loads model+tokenizer on first call; `available=False` if `repo` empty or load fails).

- [ ] **Step 1: Write the failing test (stub PhoBERT via a fake object)**

Append to `app/backend/tests/test_endpoints.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest app/backend/tests/test_endpoints.py::test_showdown_includes_phobert_when_available -v`
Expected: FAIL — without `app.state.phobert` wiring the assertion `names[0] == "PhoBERT"` fails (PhoBERT absent). (This test sets the stub directly, so it actually exercises the router branch; it passes only once the router reads `app.state.phobert`, which it already does from Task 4 — so if Task 4 is correct this test passes immediately and the real work of this task is the service + main wiring below.)

- [ ] **Step 3: Implement the PhoBERT service**

Create `app/backend/services/phobert.py`:
```python
import numpy as np

from app.ml.preprocessing import preprocess_text
from app.ml.teencode import TEENCODE_PHOBERT, normalize_teencode

MAX_LEN = 256


def _clean(raw: str) -> str:
    return normalize_teencode(preprocess_text(raw), TEENCODE_PHOBERT)


class PhoBertService:
    def __init__(self, repo: str):
        self.repo = repo
        self._model = None
        self._tokenizer = None
        self._failed = False

    @property
    def available(self) -> bool:
        return bool(self.repo) and not self._failed

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if not self.repo or self._failed:
            return False
        try:
            import torch
            from transformers import (AutoModelForSequenceClassification,
                                      AutoTokenizer)
            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.repo)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.repo)
            self._model.eval()
            return True
        except Exception:
            self._failed = True
            return False

    def predict_proba(self, raw_text: str) -> list[float]:
        if not self._ensure_loaded():
            raise RuntimeError("PhoBERT unavailable")
        torch = self._torch
        enc = self._tokenizer(
            _clean(raw_text), padding="max_length", truncation=True,
            max_length=MAX_LEN, return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**enc).logits[0]
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
        return [float(p) for p in proba]
```

- [ ] **Step 4: Wire into `main.py`**

In `app/backend/main.py`, replace `app.state.phobert = None  # wired in Task 6` with:
```python
    from app.backend.services.phobert import PhoBertService
    app.state.phobert = PhoBertService(settings.phobert_repo)
```
(With no `PHOBERT_REPO` set, `available` is `False`, so existing tests still see 6 models.)

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest app/backend/tests -v`
Expected: PASS (all previous + the new PhoBERT-stub test)

- [ ] **Step 6: Commit**

```bash
git add app/backend/services/phobert.py app/backend/main.py app/backend/tests/test_endpoints.py
git commit -m "feat(backend): lazy-loaded PhoBERT service + showdown wiring"
```

---

### Task 6: Gemini rewrite service + `/rewrite`

**Files:**
- Create: `app/backend/services/gemini.py`, `app/backend/routers/rewrite.py`
- Modify: `app/backend/main.py` (include rewrite router)
- Modify: `app/backend/schemas.py` (add rewrite schemas)
- Test: extend `app/backend/tests/test_endpoints.py`

**Interfaces:**
- Consumes: `config.Settings.gemini_api_key`, the `/predict` proba path (reuse registry/phobert).
- Produces:
  - `gemini.rewrite_polite(text: str, api_key: str) -> str` (raises `RuntimeError` on failure/no key).
  - `POST /rewrite {text}` → `RewriteResponse {rewritten:str, before:{label,label_name,proba}, after:{label,label_name,proba}}`.

- [ ] **Step 1: Write the failing test (monkeypatch the Gemini call)**

Append to `app/backend/tests/test_endpoints.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest app/backend/tests/test_endpoints.py::test_rewrite_loop -v`
Expected: FAIL with 404 (route not registered).

- [ ] **Step 3: Add rewrite schemas**

Append to `app/backend/schemas.py`:
```python
class Verdict(BaseModel):
    label: int
    label_name: str
    proba: list[float]


class RewriteResponse(BaseModel):
    rewritten: str
    before: Verdict
    after: Verdict
```

- [ ] **Step 4: Implement the Gemini client**

Create `app/backend/services/gemini.py`:
```python
PROMPT = (
    "Bạn là trợ lý kiểm duyệt. Viết lại bình luận tiếng Việt sau cho lịch sự, "
    "tôn trọng, GIỮ NGUYÊN ý chính, bỏ toàn bộ từ ngữ thù ghét/xúc phạm. "
    "Chỉ trả về câu đã viết lại, không giải thích.\n\nBình luận: {text}"
)


def rewrite_polite(text: str, api_key: str | None) -> str:
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY chưa cấu hình.")
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=PROMPT.format(text=text),
        )
        out = (resp.text or "").strip()
        if not out:
            raise RuntimeError("Gemini trả về rỗng.")
        return out
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Gemini lỗi: {e}") from e
```

- [ ] **Step 5: Implement the rewrite router**

Create `app/backend/routers/rewrite.py`:
```python
import numpy as np
from fastapi import APIRouter, HTTPException, Request

from app.backend.constants import LABEL_NAMES
from app.backend.schemas import RewriteResponse, TextRequest, Verdict
from app.backend.services.gemini import rewrite_polite

router = APIRouter()


def _verdict(request: Request, text: str) -> Verdict:
    reg = request.app.state.registry
    phobert = request.app.state.phobert
    if phobert is not None and phobert.available:
        proba = [float(p) for p in phobert.predict_proba(text)]
    else:
        proba = [float(p) for p in reg.predict_proba("LogisticRegression", text)]
    label = int(np.argmax(proba))
    return Verdict(label=label, label_name=LABEL_NAMES[label], proba=proba)


@router.post("/rewrite", response_model=RewriteResponse)
def rewrite(req: TextRequest, request: Request) -> RewriteResponse:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text rỗng.")
    before = _verdict(request, text)
    try:
        rewritten = rewrite_polite(text, request.app.state.settings.gemini_api_key)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    after = _verdict(request, rewritten)
    return RewriteResponse(rewritten=rewritten, before=before, after=after)
```

- [ ] **Step 6: Register the router**

In `app/backend/main.py` add import `from app.backend.routers import predict, rewrite` and `app.include_router(rewrite.router)`.

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_endpoints.py -v`
Expected: PASS (incl. `test_rewrite_loop`, `test_rewrite_503_without_key`)

- [ ] **Step 8: Commit**

```bash
git add app/backend/services/gemini.py app/backend/routers/rewrite.py app/backend/main.py app/backend/schemas.py app/backend/tests/test_endpoints.py
git commit -m "feat(backend): Gemini rewrite endpoint with before/after re-classify"
```

---

### Task 7: Batch CSV scoring `/batch`

**Files:**
- Create: `app/backend/routers/batch.py`
- Modify: `app/backend/main.py`, `app/backend/schemas.py`
- Test: extend `app/backend/tests/test_endpoints.py`

**Interfaces:**
- Consumes: `SklearnRegistry.predict_proba` (uses fast LogisticRegression for batch to keep latency low), `config.Settings.max_batch_rows`.
- Produces: `POST /batch` (multipart file upload, field `file`) → `BatchResponse {total:int, counts:{CLEAN,OFFENSIVE,HATE}, toxic_ratio:float, rows:[{text,label,label_name,proba}]}`. CSV must contain a `free_text` column.

- [ ] **Step 1: Write the failing test**

Append to `app/backend/tests/test_endpoints.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest app/backend/tests/test_endpoints.py::test_batch_scores_csv -v`
Expected: FAIL with 404.

- [ ] **Step 3: Add batch schemas**

Append to `app/backend/schemas.py`:
```python
class BatchRow(BaseModel):
    text: str
    label: int
    label_name: str
    proba: list[float]


class BatchResponse(BaseModel):
    total: int
    counts: dict[str, int]
    toxic_ratio: float
    rows: list[BatchRow]
```

- [ ] **Step 4: Implement the batch router**

Create `app/backend/routers/batch.py`:
```python
import io

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, UploadFile

from app.backend.constants import LABEL_NAMES
from app.backend.schemas import BatchResponse, BatchRow

router = APIRouter()


@router.post("/batch", response_model=BatchResponse)
async def batch(request: Request, file: UploadFile) -> BatchResponse:
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"CSV không đọc được: {e}")
    if "free_text" not in df.columns:
        raise HTTPException(status_code=400, detail="Thiếu cột 'free_text'.")
    max_rows = request.app.state.settings.max_batch_rows
    df = df.head(max_rows)
    reg = request.app.state.registry
    counts = {name: 0 for name in LABEL_NAMES}
    rows = []
    for text in df["free_text"].astype(str).tolist():
        proba = [float(p) for p in reg.predict_proba("LogisticRegression", text)]
        label = int(np.argmax(proba))
        counts[LABEL_NAMES[label]] += 1
        rows.append(BatchRow(text=text, label=label,
                             label_name=LABEL_NAMES[label], proba=proba))
    total = len(rows)
    toxic = counts["OFFENSIVE"] + counts["HATE"]
    return BatchResponse(
        total=total, counts=counts,
        toxic_ratio=(toxic / total if total else 0.0), rows=rows,
    )
```

- [ ] **Step 5: Register the router**

In `app/backend/main.py`: `from app.backend.routers import predict, rewrite, batch` and `app.include_router(batch.router)`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_endpoints.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add app/backend/routers/batch.py app/backend/main.py app/backend/schemas.py app/backend/tests/test_endpoints.py
git commit -m "feat(backend): /batch CSV scoring with toxicity summary"
```

---

### Task 8: Static insights `/insights` + final smoke

**Files:**
- Create: `app/backend/services/metrics.py`, `app/backend/routers/insights.py`, `app/backend/insights_data.json`
- Modify: `app/backend/main.py`, `app/backend/schemas.py`
- Test: extend `app/backend/tests/test_endpoints.py`

**Interfaces:**
- Produces: `GET /insights` → `InsightsResponse {models:[{display_name, accuracy, precision_w, recall_w, f1_w, f1_macro}], best:str}`. Data is static JSON copied from the report metric table (§06_results).

- [ ] **Step 1: Create the static metrics JSON**

Create `app/backend/insights_data.json` (values copied verbatim from `reports/report_2/content/06_results.tex`):
```json
{
  "best": "PhoBERT-base-v2",
  "models": [
    {"display_name": "PhoBERT-base-v2", "accuracy": 0.8558, "precision_w": 0.8733, "recall_w": 0.8558, "f1_w": 0.8637, "f1_macro": 0.6703},
    {"display_name": "Random Forest", "accuracy": 0.8290, "precision_w": 0.7975, "recall_w": 0.8290, "f1_w": 0.8080, "f1_macro": 0.5101},
    {"display_name": "Multinomial NB", "accuracy": 0.7907, "precision_w": 0.8317, "recall_w": 0.7907, "f1_w": 0.8068, "f1_macro": 0.5622},
    {"display_name": "Voting Ensemble", "accuracy": 0.7749, "precision_w": 0.8155, "recall_w": 0.7749, "f1_w": 0.7919, "f1_macro": 0.5455},
    {"display_name": "Logistic Regression", "accuracy": 0.7717, "precision_w": 0.8158, "recall_w": 0.7717, "f1_w": 0.7899, "f1_macro": 0.5468},
    {"display_name": "Linear SVC", "accuracy": 0.7736, "precision_w": 0.8093, "recall_w": 0.7736, "f1_w": 0.7889, "f1_macro": 0.5367},
    {"display_name": "SGD Classifier", "accuracy": 0.7446, "precision_w": 0.8252, "recall_w": 0.7446, "f1_w": 0.7746, "f1_macro": 0.5406}
  ]
}
```

- [ ] **Step 2: Write the failing test**

Append to `app/backend/tests/test_endpoints.py`:
```python
def test_insights(client):
    r = client.get("/insights")
    assert r.status_code == 200
    body = r.json()
    assert body["best"] == "PhoBERT-base-v2"
    assert len(body["models"]) == 7
    assert body["models"][0]["f1_macro"] == 0.6703
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest app/backend/tests/test_endpoints.py::test_insights -v`
Expected: FAIL with 404.

- [ ] **Step 4: Implement metrics loader + router + schema**

Append to `app/backend/schemas.py`:
```python
class ModelMetric(BaseModel):
    display_name: str
    accuracy: float
    precision_w: float
    recall_w: float
    f1_w: float
    f1_macro: float


class InsightsResponse(BaseModel):
    best: str
    models: list[ModelMetric]
```

Create `app/backend/services/metrics.py`:
```python
import json
from functools import lru_cache
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "insights_data.json"


@lru_cache
def load_insights() -> dict:
    return json.loads(_DATA.read_text(encoding="utf-8"))
```

Create `app/backend/routers/insights.py`:
```python
from fastapi import APIRouter

from app.backend.schemas import InsightsResponse
from app.backend.services.metrics import load_insights

router = APIRouter()


@router.get("/insights", response_model=InsightsResponse)
def insights() -> InsightsResponse:
    return InsightsResponse(**load_insights())
```

In `app/backend/main.py`: `from app.backend.routers import predict, rewrite, batch, insights` and `app.include_router(insights.router)`.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest app/backend/tests -v`
Expected: PASS (all tests)

- [ ] **Step 6: Smoke-run the server (no real artifacts → expect a clear load error, documenting the artifact dependency)**

Run: `ARTIFACTS_DIR=artifacts python -c "from app.backend.main import create_app; create_app()"`
Expected: If `artifacts/` is absent, a `FileNotFoundError` naming `tfidf_vectorizer.pkl` — this is the documented signal that the user must drop in Colab artifacts. With synthetic/real artifacts present, no error.

- [ ] **Step 7: Commit**

```bash
git add app/backend/insights_data.json app/backend/services/metrics.py app/backend/routers/insights.py app/backend/main.py app/backend/schemas.py app/backend/tests/test_endpoints.py
git commit -m "feat(backend): /insights endpoint from report metrics + full suite"
```

---

## Self-Review

**Spec coverage:**
- §3 backend topology → Tasks 4–8 (FastAPI, CORS, routers). ✅
- §4.1 shared preprocessing parity → Task 1 + registry/phobert cleaners. ✅
- §4.2 7-model registry + feature routing + Voting mean + LinearSVC calibrated → Task 2. ✅
- §4.3 endpoints predict/showdown/rewrite/batch/insights/health → Tasks 4,5,6,7,8. ✅
- §4.4 LR explainability → Task 3. ✅
- §4.5 Gemini rewrite + re-classify loop → Task 6. ✅
- §4.6 error handling (empty/too long, missing column, Gemini fail 503, PhoBERT fail → 6 models) → Tasks 4,5,6,7. ✅
- §6 testing (pytest, proba sum≈1, 7 models, CSV, Gemini mock) → conftest + all test files. ✅
- §7 artifacts are user-provided → encoded as a contract; tests use synthetic. ✅

Frontend (§5) and deploy (§8) are deliberately out of this plan — separate plans (see Execution Handoff).

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✅

**Type consistency:** `predict_proba(name, raw_text)`, `predict_all(raw_text)`, `explain_tokens(reg, raw_text, pred_label)`, `PhoBertService.predict_proba(raw_text)`, `rewrite_polite(text, api_key)` used identically across tasks; `app.state.{settings,registry,phobert}` names consistent; response model field names match between schemas and routers. ✅

---

## Execution Handoff

This is **Plan 1 of 3** (Backend). Plan 2 (Frontend/Next.js) consumes the contract defined here; Plan 3 (Deploy: HF Space + Vercel) wires them together. Both written after backend lands.
