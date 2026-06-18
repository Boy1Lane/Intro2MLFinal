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
