"""Functions for linking news events with subsequent market moves."""
from __future__ import annotations

from typing import Literal

import pandas as pd

from src.utils.helpers import ensure_datetime


ReactionMethod = Literal["close_to_close", "close_to_next"]


def prepare_market(market_df: pd.DataFrame) -> pd.DataFrame:
    market_df = ensure_datetime(market_df, "timestamp")
    market_df = market_df.sort_values("timestamp")
    market_df["return"] = market_df["close"].pct_change().fillna(0.0)
    market_df["future_return"] = market_df["close"].pct_change().shift(-1).fillna(0.0)
    return market_df.reset_index(drop=True)


def merge_on_timestamps(
    news_df: pd.DataFrame,
    market_df: pd.DataFrame,
    reaction: ReactionMethod = "close_to_next",
    tolerance: str = "2D",
) -> pd.DataFrame:
    """Match each news item with the closest market observation."""
    if "timestamp" not in news_df.columns or "timestamp" not in market_df.columns:
        raise KeyError("timestamp column missing")
    news_df = ensure_datetime(news_df, "timestamp").sort_values("timestamp")
    market_df = prepare_market(market_df)

    market_df = market_df.set_index("timestamp").sort_index()
    news_df = news_df.set_index("timestamp").sort_index()

    joined = pd.merge_asof(
        news_df.reset_index().sort_values("timestamp"),
        market_df.reset_index().sort_values("timestamp"),
        on="timestamp",
        direction="forward",
        tolerance=pd.Timedelta(tolerance),
    )

    if reaction == "close_to_close":
        joined["reaction"] = joined["return"].fillna(0.0)
    else:
        joined["reaction"] = joined["future_return"].fillna(0.0)

    return joined.dropna(subset=["reaction", "clean_text"]).reset_index(drop=True)
