"""Plotting utilities for analysis notebooks."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_event_study(results: pd.DataFrame) -> plt.Axes:
    fig, ax = plt.subplots(figsize=(6, 4))
    results[["avg_abnormal_return", "cumulative_abnormal_return"]].plot(ax=ax)
    ax.set_xlabel("Days relative to event")
    ax.set_ylabel("Return")
    ax.set_title("Event Study Results")
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    fig.tight_layout()
    return ax


def plot_sentiment_distribution(df: pd.DataFrame) -> plt.Axes:
    fig, ax = plt.subplots(figsize=(5, 4))
    df["sentiment_seed"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.set_title("Seed Sentiment Distribution")
    fig.tight_layout()
    return ax
