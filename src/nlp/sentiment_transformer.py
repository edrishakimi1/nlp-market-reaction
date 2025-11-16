"""Sentiment model built on top of the TF-IDF embedder."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from .embedder import NewsEmbedder


@dataclass
class SentimentTransformer:
    embedder_model: NewsEmbedder | None = None
    regularization: float = 1.0

    def __post_init__(self) -> None:
        self.embedder_model = self.embedder_model or NewsEmbedder()
        self.classifier = LogisticRegression(max_iter=500, C=self.regularization, multi_class="auto")

    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> "SentimentTransformer":
        vectors = self.embedder_model.fit_transform(texts)
        self.classifier.fit(vectors, labels)
        return self

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        vectors = self.embedder_model.transform(texts)
        return self.classifier.predict(vectors)

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        vectors = self.embedder_model.transform(texts)
        return self.classifier.predict_proba(vectors)

    def save(self, path: Path | str) -> None:
        payload = {
            "embedder": self.embedder_model,
            "classifier": self.classifier,
            "regularization": self.regularization,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path | str) -> "SentimentTransformer":
        payload = joblib.load(path)
        model = cls(embedder_model=payload["embedder"], regularization=payload["regularization"])
        model.classifier = payload["classifier"]
        return model
