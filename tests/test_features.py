import numpy as np
import pytest
from middleware.features.features import compute_bandpower, compute_entropy, run as fe_run

def make_sine(freq, fs, length=1.0):
    t = np.linspace(0, length, int(fs*length), endpoint=False)
    return np.sin(2*np.pi*freq*t)

def test_compute_bandpower_peak():
    fs = 200
    # Single-channel 10Hz sine
    sig = make_sine(10, fs)
    data = sig[np.newaxis, :]
    bands = [[8, 12], [20, 30]]
    bp = compute_bandpower(data, fs, bands)
    # 8-12 band should have power > 20-30 band
    assert bp["band_8_12"][0] > bp["band_20_30"][0] * 5

def test_compute_entropy_uniform_vs_const():
    # uniform noise has higher entropy than constant
    const = np.ones((1, 1000))
    noise = np.random.randn(1, 1000)
    ent_const = compute_entropy(const)[0]
    ent_noise = compute_entropy(noise)[0]
    assert ent_noise > ent_const + 1.0

def test_run_features_combination():
    fs = 100
    data = np.vstack([make_sine(5, fs), make_sine(20, fs)])
    feats = [{"bandpower": [[4, 6]]}, {"signal_entropy": True}]
    results = fe_run(data, fs, feats)
    # keys
    assert "band_4_6" in results
    assert "signal_entropy" in results
    # shapes
    assert results["band_4_6"].shape == (2,)
    assert results["signal_entropy"].shape == (2,)
