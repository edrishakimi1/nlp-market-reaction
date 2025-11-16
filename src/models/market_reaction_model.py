"""Model relating sentiment features to subsequent market moves."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

TARGET_COLUMN = "reaction"


def prepare_training_data(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [col for col in df.columns if col.startswith("sentiment_prob_") or col == "sentiment_pred"]
    X = df[feature_cols]
    y = df[target_column]
    return X, y


def train_reaction_model(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> LinearRegression:
    X, y = prepare_training_data(df, target_column)
    model = LinearRegression()
    model.fit(X, y)
    return model


def save_model(model: LinearRegression, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path | str) -> LinearRegression:
    return joblib.load(path)


def predict(model: LinearRegression, df: pd.DataFrame) -> pd.Series:
    X, _ = prepare_training_data(df)
    return pd.Series(model.predict(X), index=df.index, name="predicted_reaction")
