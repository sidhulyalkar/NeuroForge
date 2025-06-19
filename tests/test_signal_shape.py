import pytest
import numpy as np
import pandas as pd
from middleware.io.signal_shape import run as validate_shape


def make_df(channels=4, fs=100, duration=1.0):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    data = np.random.randn(len(t), channels)
    df = pd.DataFrame(data, columns=[f"ch{i}" for i in range(channels)])
    df.insert(0, "time", t)
    return df, fs


def test_validate_numpy_array_ok():
    arr = np.random.randn(8, 500)
    out_arr, times = validate_shape(arr, expected_channels=8, expected_fs=None)
    assert out_arr.shape == (8, 500)
    assert times is None


def test_validate_dataframe_ok():
    df, fs = make_df(channels=4, fs=100, duration=2)
    arr, times = validate_shape(df, expected_channels=4, expected_fs=fs)
    assert arr.shape == (4, int(100 * 2))
    # compute sampling rate from time vector
    dt = np.diff(times)
    fs_est = 1.0 / np.median(dt)
    # allow small tolerance (±0.1%)
    assert pytest.approx(fs, rel=1e-3) == fs_est


def test_wrong_channel_count():
    arr = np.random.randn(5, 100)
    with pytest.raises(ValueError):
        validate_shape(arr, expected_channels=4)


def test_non_dataframe_non_array():
    with pytest.raises(TypeError):
        validate_shape("not data", expected_channels=None, expected_fs=None)
