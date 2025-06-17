import numpy as np
import pandas as pd

def run(data, expected_channels=None, expected_fs=None):
    """
    Validate the shape (and optionally sampling rate) of the input signal.

    Parameters
    ----------
    data : numpy.ndarray or pandas.DataFrame
        • If ndarray: expected shape (channels, timepoints).
        • If DataFrame: first column 'time', remaining columns are channels.
    expected_channels : int, optional
        If provided, will assert number of channels matches.
    expected_fs : float, optional
        If provided and `data` is DataFrame with 'time' col, will estimate
        sampling rate and assert it’s within 1% of expected_fs.

    Returns
    -------
    arr : numpy.ndarray
        Signal array of shape (channels, timepoints).
    times : numpy.ndarray or None
        Time vector if `data` was a DataFrame with a 'time' column, else None.

    Raises
    ------
    ValueError
        If expected_channels mismatch or estimated fs differs too much.
    TypeError
        If data is not ndarray or DataFrame.
    """
    times = None

    # Extract array and time vector
    if isinstance(data, pd.DataFrame):
        if 'time' in data.columns:
            times = data['time'].values
            arr = data.drop(columns=['time']).values.T
        else:
            arr = data.values.T
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise TypeError("Data must be a NumPy array or pandas DataFrame")

    channels, timepoints = arr.shape

    # Check channel count
    if expected_channels is not None and channels != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels, got {channels}")

    # Check sampling rate if possible
    if expected_fs is not None and times is not None:
        dt = np.diff(times)
        median_dt = np.median(dt)
        if median_dt <= 0:
            raise ValueError("Non-positive time increment detected")
        fs_est = 1.0 / median_dt
        # allow ±1% tolerance
        if abs(fs_est - expected_fs) / expected_fs > 0.01:
            raise ValueError(f"Estimated fs={fs_est:.1f}Hz ≠ expected {expected_fs}Hz")
    return arr, times
