import numpy as np
from scipy.signal import welch

BAND_MAP = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
}


def compute_bandpower(data: np.ndarray, fs: float, bands: list) -> dict:
    """
    Compute bandpower for each channel over specified frequency bands.

    Parameters
    ----------
    data : ndarray, shape (channels, timepoints)
    fs : float
        Sampling rate in Hz.
    bands : list of [low, high] pairs
        Frequency bands to integrate.

    Returns
    -------
    dict
        Keys like 'band_1_4' → ndarray of length channels.
    """
    # allow names or numeric
    numeric_bands = []
    for band in bands:
        if isinstance(band, str):
            numeric_bands.append(BAND_MAP[band])
        else:
            numeric_bands.append(band)

    # Estimate PSD with Welch's method
    freqs, psd = welch(data, fs=fs, axis=1, nperseg=min(data.shape[1], fs * 2))
    bp = {}
    for low, high in numeric_bands:
        mask = (freqs >= low) & (freqs <= high)
        # integrate PSD under the curve for each channel
        power = np.trapz(psd[:, mask], freqs[mask], axis=1)
        bp[f"band_{low}_{high}"] = power
    return bp


def compute_entropy(data: np.ndarray, bins: int = 100) -> np.ndarray:
    """
    Compute Shannon entropy for each channel.

    Parameters
    ----------
    data : ndarray, shape (channels, timepoints)
    bins : int
        Number of bins for histogram.

    Returns
    -------
    entropies : ndarray, length channels
    """
    entropies = []
    for ch in data:
        hist, _ = np.histogram(ch, bins=bins, density=True)
        p = hist + 1e-12  # avoid log(0)
        ent = -np.sum(p * np.log2(p))
        entropies.append(ent)
    return np.array(entropies)


def run(data: np.ndarray, fs: float, features: list) -> dict:
    """
    Apply feature extraction steps.

    Parameters
    ----------
    data : ndarray, shape (channels, timepoints)
    fs : float
    features : list of dicts, e.g.:
        [{"bandpower": [[1,4], [4,8]]}, {"signal_entropy": True}]

    Returns
    -------
    results : dict
        feature_name → ndarray
    """
    results = {}
    for feat in features:
        if "bandpower" in feat:
            bp = compute_bandpower(data, fs, feat["bandpower"])
            results.update(bp)
        if feat.get("signal_entropy", False):
            results["signal_entropy"] = compute_entropy(data)
    return results
