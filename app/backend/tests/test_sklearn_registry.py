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
