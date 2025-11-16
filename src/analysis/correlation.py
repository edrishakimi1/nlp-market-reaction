"""Correlation helpers between sentiment metrics and market reactions."""
from __future__ import annotations

import pandas as pd


def sentiment_reaction_correlation(df: pd.DataFrame) -> pd.Series:
    """Compute Pearson correlations for available sentiment probability columns."""
    feature_cols = [col for col in df.columns if col.startswith("sentiment_prob_")]
    if not feature_cols or "reaction" not in df.columns:
        raise KeyError("required columns missing")
    corr = df[feature_cols + ["reaction"]].corr()["reaction"].drop("reaction")
    return corr.sort_values(ascending=False)
