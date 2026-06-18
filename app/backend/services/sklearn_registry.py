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
