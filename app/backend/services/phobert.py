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

    def try_proba(self, raw_text: str) -> list[float] | None:
        """Predict, degrading to None on any load/inference failure.

        Routers call this so a PhoBERT problem (missing torch, bad repo, OOM)
        degrades to the sklearn models instead of raising a 500 (spec §4.6).
        """
        if not self.available:
            return None
        try:
            return self.predict_proba(raw_text)
        except Exception:
            return None

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
