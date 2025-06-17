import numpy as np
import pandas as pd
from scipy.signal import iirnotch, butter, filtfilt

def apply_notch_filter(data, fs, notch_freq=60.0, quality_factor=30):
    """Remove line noise at `notch_freq` Hz using an IIR notch filter."""
    b, a = iirnotch(notch_freq, quality_factor, fs)
    return filtfilt(b, a, data, axis=-1)

def apply_bandpass_filter(data, fs, lowcut, highcut, order=4):
    """Apply a Butterworth bandpass filter between lowcut and highcut (Hz)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def apply_zscore(data):
    """Z-score normalize each channel independently."""
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    return (data - mean) / (std + 1e-8)

def apply_common_average_reference(data):
    """Subtract the mean across channels (CAR) to remove common-mode noise."""
    mean_sig = np.mean(data, axis=0, keepdims=True)
    return data - mean_sig

def apply_highpass_filter(data, fs, cutoff=70.0, order=4):
    """High-pass Butterworth filter to isolate high-gamma (or other) band."""
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='high')
    return filtfilt(b, a, data, axis=-1)

def run(data, fs, steps):
    """
    Apply a sequence of preprocessing steps.

    Parameters
    ----------
    data : numpy.ndarray of shape (channels, timepoints)
    fs : float
        Sampling rate in Hz.
    steps : list of dict
        Each dict specifies one step:
          - {"notch_filter": 60}
          - {"bandpass_filter": [1, 40]}
          - {"zscore_normalization": True}
          - {"car": True}
          - {"highpass_filter": 70}

    Returns
    -------
    processed : numpy.ndarray
        Data after all steps.
    """
    processed = data.copy()
    for step in steps:
        if not isinstance(step, dict):
            print(f"Warning: skipping invalid step {step}")
            continue
        if 'notch_filter' in step:
            processed = apply_notch_filter(processed, fs, notch_freq=step['notch_filter'])
        elif 'bandpass_filter' in step:
            low, high = step['bandpass_filter']
            processed = apply_bandpass_filter(processed, fs, low, high)
        elif 'zscore_normalization' in step and step['zscore_normalization']:
            processed = apply_zscore(processed)
        elif 'car' in step and step['car']:
            processed = apply_common_average_reference(processed)
        elif 'highpass_filter' in step:
            processed = apply_highpass_filter(processed, fs, cutoff=step['highpass_filter'])
        else:
            print(f"Warning: unknown preprocessing step {step}")
    return processed
