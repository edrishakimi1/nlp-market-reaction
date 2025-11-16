"""Event study helpers for understanding post-news market moves."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from src.utils.helpers import ensure_datetime


def compute_event_windows(
    events_df: pd.DataFrame,
    market_df: pd.DataFrame,
    window: int = 3,
) -> pd.DataFrame:
    """Return average abnormal and cumulative returns for an event window."""
    if window <= 0:
        raise ValueError("window must be positive")

    events_df = ensure_datetime(events_df, "timestamp")
    market_df = ensure_datetime(market_df, "timestamp").sort_values("timestamp")
    market_df = market_df.set_index("timestamp")
    market_df["return"] = market_df["close"].pct_change().fillna(0.0)

    windows = []
    for event_time in events_df["timestamp"]:
        start = event_time - pd.Timedelta(days=window)
        end = event_time + pd.Timedelta(days=window)
        slice_df = market_df.loc[start:end, "return"]
        if slice_df.empty:
            continue
        rel_index = (slice_df.index - event_time).days
        windows.append(pd.Series(slice_df.values, index=rel_index))

    if not windows:
        return pd.DataFrame(columns=["avg_abnormal_return", "cumulative_abnormal_return"])

    stacked = pd.concat(windows, axis=1)
    avg = stacked.mean(axis=1).sort_index()
    car = avg.cumsum()
    return pd.DataFrame(
        {
            "avg_abnormal_return": avg,
            "cumulative_abnormal_return": car,
        }
    )


def summarize_events(events_df: pd.DataFrame) -> pd.Series:
    """Provide quick counts of sentiment labels within events."""
    return events_df["sentiment_seed"].value_counts()
