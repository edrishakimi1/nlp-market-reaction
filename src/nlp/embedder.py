"""Lightweight text embedding using TF-IDF."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class NewsEmbedder:
    max_features: int = 1000
    ngram_range: tuple[int, int] = (1, 2)

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)

    def fit(self, corpus: Iterable[str]) -> "NewsEmbedder":
        self.vectorizer.fit(corpus)
        return self

    def transform(self, corpus: Iterable[str]) -> np.ndarray:
        return self.vectorizer.transform(corpus).toarray()

    def fit_transform(self, corpus: Iterable[str]) -> np.ndarray:
        return self.vectorizer.fit_transform(corpus).toarray()

    def vocab(self) -> list[str]:
        return list(self.vectorizer.get_feature_names_out())
