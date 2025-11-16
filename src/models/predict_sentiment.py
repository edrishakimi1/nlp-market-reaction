"""Utilities for batch predictions from trained sentiment model."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.nlp.sentiment_transformer import SentimentTransformer


def load_model(path: Path | str) -> SentimentTransformer:
    return SentimentTransformer.load(path)


def predict_dataframe(texts: Iterable[str], model_path: Path | str) -> pd.DataFrame:
    model = load_model(model_path)
    preds = model.predict(texts)
    probas = model.predict_proba(texts)
    df = pd.DataFrame(probas, columns=[f"prob_{cls}" for cls in model.classifier.classes_])
    df.insert(0, "prediction", preds)
    return df
