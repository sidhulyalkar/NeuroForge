# tests/test_realtime.py

import time
import numpy as np
import pytest

from middleware.sdk.sdk import BCIClient


def test_realtime_processing_pipeline(monkeypatch):
    """
    Verify that start_processing_pipeline calls the callback with
    correct shapes and timestamps, using dummy streaming data.
    """
    # 1) Create client in dummy mode
    client = BCIClient(use_dummy=True)

    # 2) Prepare dummy buffer: 2 channels, 50 samples of value=1
    client._buffer = [np.ones((2, 50))]

    # 3) Define simple pipeline functions
    def preprocess_fn(data, fs):
        # data shape: (2, window_samples)
        return data  # pass through

    def feature_fn(clean, fs):
        # produce a dict of one feature per channel
        # here feature = mean of the window
        feats = {"mean": clean.mean(axis=1)}
        return feats

    def decode_fn(feats, spec):
        # just return the feature plus channel index
        arr = feats["mean"] + np.arange(len(feats["mean"]))
        return arr

    # 4) Collect callback results
    calls = []

    def callback(preds, ts):
        calls.append((preds.copy(), ts))

    # 5) Start real-time processing with small window
    fs = 10  # 10 Hz
    client.start_processing_pipeline(
        fs=fs,
        preprocess_fn=preprocess_fn,
        feature_fn=feature_fn,
        decode_fn=decode_fn,
        decoding_spec={},
        window_size_s=0.1,  # 1 sample per window
        step_size_s=0.1,
        callback=callback,
    )

    # Let it run for ~0.3 seconds
    time.sleep(0.35)
    client.stop_processing_pipeline()

    # 6) Assert that callback was invoked multiple times
    assert len(calls) >= 2

    # And each preds array has length == number of channels (2)
    for preds, ts in calls:
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (2,)
        assert ts >= 0


def test_stop_processing_sets_flag():
    client = BCIClient(use_dummy=True)
    # no threading here: just flip the flag
    client._processing = True
    client._processing_thread = None
    client.stop_processing_pipeline()
    assert client._processing is False
