import numpy as np
import pytest
from middleware.preprocessing.preprocessing import (
    apply_notch_filter,
    apply_bandpass_filter,
    apply_zscore,
    apply_common_average_reference,
    apply_highpass_filter,
)

def make_sine(freq, fs, duration=1.0):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)

def test_notch_removes_60hz():
    fs = 250
    # 60 Hz sine + small noise
    sig = make_sine(60, fs) + 0.1 * np.random.randn(250)
    filtered = apply_notch_filter(sig[np.newaxis, :], fs, notch_freq=60)
    # compute FFT of original and filtered
    orig_fft = np.abs(np.fft.rfft(sig))
    filt_fft = np.abs(np.fft.rfft(filtered[0]))
    # amplitude at 60Hz bin should drop by >50%
    freq_axis = np.fft.rfftfreq(len(sig), 1/fs)
    idx = np.argmin(np.abs(freq_axis - 60))
    assert filt_fft[idx] < orig_fft[idx] * 0.5

def test_bandpass_allows_only_band():
    fs = 200
    # mixed 10Hz and 50Hz
    sig = make_sine(10, fs) + make_sine(50, fs)
    bp = apply_bandpass_filter(sig[np.newaxis, :], fs, 40, 60)[0]
    fft = np.abs(np.fft.rfft(bp))
    freqs = np.fft.rfftfreq(len(sig), 1/fs)
    # 10Hz component should be heavily attenuated
    idx10 = np.argmin(np.abs(freqs - 10))
    idx50 = np.argmin(np.abs(freqs - 50))
    assert fft[idx50] > fft[idx10] * 5

def test_zscore_zero_mean_unit_std():
    data = np.random.randn(3, 1000)
    zs = apply_zscore(data)
    # mean per channel ~0, std ~1
    assert np.allclose(zs.mean(axis=1), 0, atol=1e-6)
    assert np.allclose(zs.std(axis=1), 1, atol=1e-6)

def test_car_removes_common_signal():
    # two channels: one sine both, one sine plus diff
    fs = 100
    t = np.linspace(0,1,fs,endpoint=False)
    base = np.sin(2*np.pi*5*t)
    data = np.vstack([base, base + 0.5*np.random.randn(fs)])
    car = apply_common_average_reference(data)
    # after CAR, the mean across channels = 0
    assert np.allclose(car.mean(axis=0), 0, atol=1e-6)

def test_highpass_blocks_low_freq():
    fs = 200
    sig = make_sine(5, fs)
    hp = apply_highpass_filter(sig[np.newaxis,:], fs, cutoff=20)[0]
    fft = np.abs(np.fft.rfft(hp))
    freqs = np.fft.rfftfreq(len(sig), 1/fs)
    idx5 = np.argmin(np.abs(freqs - 5))
    assert fft[idx5] < 1e-1
