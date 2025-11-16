"""Feature generation utilities for downstream models."""
from __future__ import annotations

import pandas as pd

from .sentiment_transformer import SentimentTransformer


CLASSES = {-1: "negative", 0: "neutral", 1: "positive"}


def _class_to_name(label: int) -> str:
    return CLASSES.get(int(label), f"class_{label}")


def build_features(df: pd.DataFrame, model: SentimentTransformer) -> pd.DataFrame:
    """Return regression ready features with probabilities from the sentiment model."""
    if "clean_text" not in df.columns:
        raise KeyError("clean_text column missing")
    features = df.copy().reset_index(drop=True)
    features["sentiment_pred"] = model.predict(features["clean_text"])

    probas = model.predict_proba(features["clean_text"])
    class_order = model.classifier.classes_
    prob_df = pd.DataFrame(probas, columns=[_class_to_name(int(c)) for c in class_order])
    prob_df = prob_df.add_prefix("sentiment_prob_")
    features = pd.concat([features, prob_df], axis=1)
    return features
