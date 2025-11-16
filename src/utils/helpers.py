"""General helper functions used across modules."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ensure_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return a copy of *df* ensuring column is datetime typed."""
    result = df.copy()
    result[column] = pd.to_datetime(result[column], errors="coerce")
    return result.dropna(subset=[column])


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element wise division that avoids division by zero."""
    denom = denominator.replace(0, np.nan)
    return numerator / denom.fillna(1.0)


def rolling_apply(series: pd.Series, window: int, func) -> pd.Series:
    """Helper around pandas rolling that returns aligned series."""
    if window <= 0:
        raise ValueError("window must be positive")
    return series.rolling(window=window, min_periods=1).apply(func, raw=False)
