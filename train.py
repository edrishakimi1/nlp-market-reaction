"""High level pipeline runner for training the sentiment and market reaction models."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.utils.config import load_config
from src.utils.logger import configure_logging
from src.data import fetch_data, preprocess, merge_news_market
from src.nlp import sentiment_transformer, embedder, feature_engineering
from src.models import train_sentiment, market_reaction_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sentiment and market reaction models")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    configure_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("Loading raw datasets")
    news_df = fetch_data.load_news(Path(config.paths.data_raw))
    market_df = fetch_data.load_market(Path(config.paths.data_market))

    logger.info("Preprocessing news data")
    news_df = preprocess.clean_news(news_df)
    news_df = preprocess.add_sentiment_seed(news_df)

    logger.info("Merging news and market data")
    merged_df = merge_news_market.merge_on_timestamps(news_df, market_df)

    logger.info("Training sentiment transformer")
    embedder_model = embedder.NewsEmbedder(max_features=config.training.sentiment.max_features)
    transformer = sentiment_transformer.SentimentTransformer(embedder_model=embedder_model)
    transformer.fit(merged_df["clean_text"], merged_df["sentiment_seed"])
    train_sentiment.save_model(transformer, Path(config.paths.sentiment_model))

    logger.info("Generating features for market reaction model")
    features = feature_engineering.build_features(merged_df, transformer)

    logger.info("Training market reaction regression")
    regressor = market_reaction_model.train_reaction_model(features)
    market_reaction_model.save_model(regressor, Path(config.paths.market_model))

    logger.info("Pipeline finished successfully")


if __name__ == "__main__":
    sys.exit(main())
