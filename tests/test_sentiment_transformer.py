from pathlib import Path

import numpy as np

from src.nlp.sentiment_transformer import SentimentTransformer


def test_transformer_fits_simple_dataset(tmp_path: Path):
    texts = ["good profit", "bad loss", "record growth", "fraud investigation"]
    labels = [1, -1, 1, -1]
    model = SentimentTransformer()
    model.fit(texts, labels)
    preds = model.predict(["profit beats", "loss widens"])
    assert preds[0] == 1
    assert preds[1] == -1

    model_path = tmp_path / "sentiment.joblib"
    model.save(model_path)
    loaded = SentimentTransformer.load(model_path)
    loaded_preds = loaded.predict(["profit beats", "loss widens"])
    np.testing.assert_array_equal(preds, loaded_preds)
