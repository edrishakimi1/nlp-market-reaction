"""Utilities for training the sentiment transformer."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.nlp.embedder import NewsEmbedder
from src.nlp.sentiment_transformer import SentimentTransformer


def train_model(
    texts: Iterable[str],
    labels: Iterable[int],
    test_size: float = 0.2,
    embedder: NewsEmbedder | None = None,
) -> tuple[SentimentTransformer, float]:
    model = SentimentTransformer(embedder_model=embedder)
    X_train, X_test, y_train, y_test = train_test_split(list(texts), list(labels), test_size=test_size, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) if len(X_test) else np.nan
    return model, float(acc)


def save_model(model: SentimentTransformer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
