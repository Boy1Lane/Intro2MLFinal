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
