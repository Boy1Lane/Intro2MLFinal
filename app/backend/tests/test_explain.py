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
