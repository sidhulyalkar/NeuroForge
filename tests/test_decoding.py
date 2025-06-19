import numpy as np
import pytest
from middleware.decoding.decoding import Decoder, run as decode_run


def test_decoder_fit_predict():
    # simple XOR-like data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    dec = Decoder()
    dec.fit(X, y)
    preds = dec.predict(X)
    # Should perfectly predict on training data
    assert np.array_equal(preds, y)


def test_run_inference():
    # build a trivial feature dict with two features
    features = {"f1": np.array([0, 1, 1, 0]), "f2": np.array([1, 0, 0, 1])}
    spec = {"model": "RandomForestClassifier", "task": "binary_classification"}
    preds = decode_run(features, spec)
    # same shape as inputs
    assert preds.shape == (4,)
    # values are integers 0 or 1
    assert set(np.unique(preds)).issubset({0, 1})
