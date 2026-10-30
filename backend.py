"""Backend: loads the trained ANN + TF-IDF and exposes prediction/explanation functions.

Uses the same NLP pipeline as AI_OEP.ipynb.
"""
import json
import pickle
import re
from pathlib import Path

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for _pkg in ("punkt_tab", "stopwords", "wordnet"):
    nltk.download(_pkg, quiet=True)

ART = Path(__file__).parent / "artifacts"
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

_STOP = set(stopwords.words("english")) - {"not", "no", "never", "nor"}
_LEMM = WordNetLemmatizer()


class EmotionEngine:
    def __init__(self):
        import tensorflow as tf  # imported lazily so the app starts fast

        if not (ART / "model.keras").exists():
            raise FileNotFoundError("Run `python train_save.py` first to create artifacts/.")
        self.model = tf.keras.models.load_model(ART / "model.keras")
        with open(ART / "tfidf.pkl", "rb") as f:
            self.tfidf = pickle.load(f)
        with open(ART / "metrics.json") as f:
            self.metrics = json.load(f)
        self.vocab = np.array(self.tfidf.get_feature_names_out())

    # ---------- NLP pipeline, step by step ----------
    @staticmethod
    def pipeline_steps(text: str) -> dict:
        lowered = text.lower()
        letters = re.sub(r"[^a-zA-Z\s]", "", lowered)
        spaced = re.sub(r"\s+", " ", letters).strip()
        tokens = word_tokenize(spaced)
        removed = [t for t in tokens if t in _STOP]
        kept = [t for t in tokens if t not in _STOP]
        lemmas = [_LEMM.lemmatize(t) for t in kept]
        return {
            "raw": text,
            "lowercase": lowered,
            "letters_only": spaced,
            "tokens": tokens,
            "stopwords_removed": removed,
            "after_stopwords": kept,
            "lemmatized": lemmas,
            "clean_text": " ".join(lemmas),
        }

    @classmethod
    def clean(cls, text: str) -> str:
        return cls.pipeline_steps(text)["clean_text"]

    # ---------- prediction ----------
    def _proba_clean(self, clean_texts):
        X = self.tfidf.transform(clean_texts).toarray().astype("float32")
        return self.model.predict(X, verbose=0)

    def predict(self, text: str) -> dict:
        clean = self.clean(text)
        p = self._proba_clean([clean])[0]
        i = int(np.argmax(p))
        return {
            "emotion": EMOTIONS[i],
            "confidence": float(p[i]),
            "probs": {e: float(v) for e, v in zip(EMOTIONS, p)},
            "clean_text": clean,
            "known_features": int(self.tfidf.transform([clean]).nnz),
        }

    def predict_many(self, texts, batch=2048):
        out = []
        cleaned = [self.clean(str(t)) for t in texts]
        for s in range(0, len(cleaned), batch):
            out.append(self._proba_clean(cleaned[s:s + batch]))
        P = np.vstack(out) if out else np.zeros((0, 6))
        return cleaned, P

    # ---------- explainability ----------
    def word_importance(self, text: str) -> list:
        """Occlusion: drop each cleaned word and measure the fall in the predicted class probability."""
        clean = self.clean(text)
        words = clean.split()
        if not words:
            return []
        base = self._proba_clean([clean])[0]
        cls = int(np.argmax(base))
        variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        P = self._proba_clean(variants)
        return [
            {"word": w, "impact": float(base[cls] - P[i][cls])}
            for i, w in enumerate(words)
        ]

    def top_features(self, text: str, k=10) -> list:
        vec = self.tfidf.transform([self.clean(text)])
        if vec.nnz == 0:
            return []
        idx = vec.indices
        vals = vec.data
        order = np.argsort(-vals)[:k]
        return [{"feature": str(self.vocab[idx[o]]), "tfidf": float(vals[o])} for o in order]
