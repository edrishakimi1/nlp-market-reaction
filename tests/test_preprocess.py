import pandas as pd

from src.data import preprocess


def test_normalize_text_removes_symbols():
    assert preprocess.normalize_text("Hello, WORLD!!!") == "hello world"


def test_clean_news_creates_clean_text_and_sorts():
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-02", "2024-01-01"],
            "headline": ["Hi", "Hi"],
        }
    )
    cleaned = preprocess.clean_news(df)
    assert list(cleaned["clean_text"]) == ["hi"]
    assert cleaned.iloc[0]["timestamp"].isoformat().startswith("2024-01-01")


def test_add_sentiment_seed_applies_keywords():
    df = pd.DataFrame({"clean_text": ["record profit", "fraud and loss", "flat results" ]})
    labeled = preprocess.add_sentiment_seed(df)
    assert list(labeled["sentiment_seed"]) == [1, -1, 0]
