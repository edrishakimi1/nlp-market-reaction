import pandas as pd
import pytest

from src.data import merge_news_market


def test_merge_aligns_news_with_future_returns():
    news = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 10:00", "2024-01-02 11:00"]),
            "clean_text": ["record profit", "fraud discovered"],
        }
    )
    market = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 09:30", "2024-01-01 16:00", "2024-01-02 16:00"]),
            "close": [100.0, 102.0, 101.0],
            "volume": [1000, 1100, 1200],
        }
    )
    merged = merge_news_market.merge_on_timestamps(news, market, tolerance="2D")
    assert "reaction" in merged.columns
    # first event should align with next close change 102->101 = -0.0098 approx for second event
    assert len(merged) == 2
    assert merged.iloc[0]["reaction"] == pytest.approx((102.0 - 101.0) / 102.0, rel=1e-5)
