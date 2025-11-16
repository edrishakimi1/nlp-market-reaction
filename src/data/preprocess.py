"""Data cleaning utilities for news data."""
from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from src.utils.helpers import ensure_datetime


TOKEN_PATTERN = re.compile(r"[^A-Za-z0-9 ]+")
POSITIVE_TOKENS = {"beat", "growth", "record", "surge", "profit"}
NEGATIVE_TOKENS = {"miss", "loss", "slow", "fraud", "decline"}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = TOKEN_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def clean_news(df: pd.DataFrame, text_column: str = "headline") -> pd.DataFrame:
    """Return cleaned news DataFrame sorted by timestamp."""
    if text_column not in df.columns:
        raise KeyError(f"missing column {text_column}")
    work = ensure_datetime(df, "timestamp")
    work = work.dropna(subset=[text_column])
    work = work.drop_duplicates(subset=[text_column])
    work["clean_text"] = work[text_column].astype(str).map(normalize_text)
    work = work.sort_values("timestamp").reset_index(drop=True)
    return work


def deduplicate_by_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Drop duplicates for the provided columns while keeping the first occurrence."""
    return df.drop_duplicates(subset=list(columns)).reset_index(drop=True)


def add_sentiment_seed(df: pd.DataFrame, text_column: str = "clean_text") -> pd.DataFrame:
    """Add heuristic sentiment labels based on keyword counts."""
    if text_column not in df.columns:
        raise KeyError(f"missing column {text_column}")

    def classify(text: str) -> int:
        tokens = set(text.split())
        pos = len(tokens & POSITIVE_TOKENS)
        neg = len(tokens & NEGATIVE_TOKENS)
        if pos > neg:
            return 1
        if neg > pos:
            return -1
        return 0

    df = df.copy()
    df["sentiment_seed"] = df[text_column].map(classify)
    return df
