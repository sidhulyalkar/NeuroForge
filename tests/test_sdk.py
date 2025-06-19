# tests/test_sdk.py

import numpy as np
import pytest


# Monkey‐patch BrainFlow dependencies before importing BCIClient
class DummyBoard:
    def __init__(self, board_id, params):
        self._data = []

    def prepare_session(self):
        pass

    def start_stream(self, packet_size):
        pass

    def get_current_board_data(self, n_samples):
        # Return synthetic data: 4 channels × n_samples
        return np.random.randn(4, n_samples)

    def stop_stream(self):
        pass

    def release_session(self):
        pass


import sys
import types

# Inject dummy BoardShim and BrainFlowInputParams
brainflow = types.SimpleNamespace(
    board_shim=types.SimpleNamespace(
        BoardShim=lambda board_id, params: DummyBoard(board_id, params),
        BrainFlowInputParams=lambda: types.SimpleNamespace(),
        BoardIds=types.SimpleNamespace(CYTON_BOARD=types.SimpleNamespace(value=0)),
    )
)
sys.modules["brainflow.board_shim"] = brainflow.board_shim

from middleware.sdk.sdk import BCIClient


def test_connect_and_disconnect():
    client = BCIClient()
    client.connect()
    client.disconnect()
    # no exceptions


def test_stream_and_buffer():
    client = BCIClient()
    client.connect()
    client.start_stream(sampling_rate=100, packet_size=10)
    # let the dummy thread fill a bit
    import time

    time.sleep(0.01)
    client.stop_stream()
    buf = client.get_buffer()
    # Should be a 2D array with 4 rows
    assert buf.ndim == 2
    assert buf.shape[0] == 4


def test_get_features_and_predict(monkeypatch):
    # stub preprocess_fn, feature_fn, decode_fn
    def fake_pre(data, fs):
        return data * 2

    def fake_feat(data, fs):
        return {"a": np.array([1, 2, 3, 4])}

    def fake_decode(feats, spec):
        return np.array([9, 9, 9, 9])

    client = BCIClient()
    client._buffer = [np.ones((4, 10))]
    client._keep_streaming = False
    feats = client.get_features(100, fake_pre, fake_feat)
    assert "a" in feats and feats["a"].shape == (4,)
    preds = client.predict(100, fake_pre, fake_feat, fake_decode, {})
    assert np.all(preds == 9)
