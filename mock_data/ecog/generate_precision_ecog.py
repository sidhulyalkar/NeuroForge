# mock_data/generate_precision_ecog.py

import numpy as np
import pandas as pd
import math

def generate_precision_ecog(
    channels: int = 1024,
    fs: int = 1000,
    duration: float = 10.0,
    low_freqs: list = [10, 20, 40],
    high_gamma_band: tuple = (70, 150),
    noise_level: float = 0.2,
    electrode_spacing_um: float = 400.0,
    diameter_range_um: tuple = (50.0, 380.0),
):
    """
    Simulate a 1024-channel Precision Neuroscience Layer 7 ECoG array:
      - 32×32 grid (sqrt(1024)) with 400µm spacing
      - Random electrode diameters 50–380µm (metadata only)
      - fs = 1000Hz, duration seconds
      - Low-frequency rhythms + high-gamma bursts + Gaussian noise
    Returns:
      df      : DataFrame with columns ['time','chan_0',…,'chan_1023']
      meta    : DataFrame with metadata: x,y (meters) and diameter (meters)
    """
    assert int(math.sqrt(channels))**2 == channels, "channels must be a perfect square"
    grid_size = int(math.sqrt(channels))
    
    # Generate electrode positions (in meters)
    spacing_m = electrode_spacing_um * 1e-6
    xs = np.arange(grid_size) * spacing_m
    ys = np.arange(grid_size) * spacing_m
    xx, yy = np.meshgrid(xs, ys)
    positions = np.vstack([xx.ravel(), yy.ravel()]).T  # shape (1024,2)

    # Random diameters in meters
    diameters = np.random.uniform(diameter_range_um[0]*1e-6,
                                  diameter_range_um[1]*1e-6,
                                  size=channels)

    # Time vector
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    data = np.zeros((channels, len(t)))

    # Low-frequency background rhythms per channel (random phase)
    for ch in range(channels):
        phase = np.random.rand() * 2*np.pi
        for f in low_freqs:
            data[ch] += np.sin(2 * np.pi * f * t + phase)

    # Add high-gamma bursts on random channels
    num_burst_ch = channels // 8  # ~128 channels get bursts
    burst_channels = np.random.choice(channels, num_burst_ch, replace=False)
    for ch in burst_channels:
        bursts = np.random.choice(len(t), size=5, replace=False)
        for b in bursts:
            start = b
            end = min(b + fs//10, len(t))
            gamma = np.sin(2 * np.pi * np.random.uniform(*high_gamma_band) * t[start:end])
            data[ch, start:end] += gamma

    # Add Gaussian noise
    data += np.random.normal(0, noise_level, size=data.shape)

    # Common Average Reference
    mean_sig = data.mean(axis=0, keepdims=True)
    data_car = data - mean_sig

    # Build DataFrame
    df = pd.DataFrame(data_car.T,
                      columns=[f"chan_{i}" for i in range(channels)])
    df.insert(0, "time", t)

    # Metadata DataFrame
    meta = pd.DataFrame({
        "channel": [f"chan_{i}" for i in range(channels)],
        "x_m": positions[:,0],
        "y_m": positions[:,1],
        "diameter_m": diameters
    })

    return df, meta

if __name__ == "__main__":
    df_ecog, meta_ecog = generate_precision_ecog()
    df_ecog.to_csv("mock_data/synthetic_ecog.csv", index=False)
    meta_ecog.to_csv("mock_data/synthetic_ecog_meta.csv", index=False)
    print("Saved synthetic ECoG data (1024 channels) and metadata.")
