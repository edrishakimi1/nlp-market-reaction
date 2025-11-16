"""Utilities to load or simulate raw datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_NEWS_COLUMNS = ["timestamp", "headline", "source"]
DEFAULT_MARKET_COLUMNS = ["timestamp", "close", "volume"]


def _simulate_news(rows: int = 10) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="6H")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "headline": [f"Company {i} beats expectations" for i in range(rows)],
            "source": "synthetic",
        }
    )


def _simulate_market(rows: int = 40) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="H")
    base_price = 100 + np.cumsum(np.random.normal(scale=0.5, size=rows))
    volume = np.random.randint(1_000, 3_000, size=rows)
    return pd.DataFrame({"timestamp": timestamps, "close": base_price, "volume": volume})


def load_news(path: Path | str, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load news CSV at *path* or simulate data when the file does not exist."""
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = _simulate_news()
    cols = list(columns) if columns is not None else DEFAULT_NEWS_COLUMNS
    return df[cols]


def load_market(path: Path | str, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load market data CSV or return simulated values."""
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = _simulate_market()
    cols = list(columns) if columns is not None else DEFAULT_MARKET_COLUMNS
    return df[cols]
