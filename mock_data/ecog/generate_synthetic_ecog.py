import numpy as np
import pandas as pd


def generate_synthetic_ecog(
    channels=32,
    fs=1000,
    duration=10,
    low_freqs=[10, 20, 40],
    high_gamma_band=(70, 150),
    noise_level=0.2,
):
    """
    Simulate ECoG data:
      - `channels` channels at `fs` Hz
      - Low-frequency sinusoids + high-gamma bursts + noise
    """
    t = np.linspace(0, duration, fs * duration)
    data = np.zeros((channels, len(t)))

    # Low-frequency background rhythms
    for idx in range(channels):
        for f in low_freqs:
            data[idx] += np.sin(2 * np.pi * f * t)

    # Add high-gamma bursts on random channels
    for idx in np.random.choice(channels, size=channels // 4, replace=False):
        # random burst times
        bursts = np.random.choice(len(t), size=5, replace=False)
        for b in bursts:
            start = b
            end = min(b + fs // 10, len(t))
            gamma = np.sin(
                2 * np.pi * np.random.uniform(*high_gamma_band) * t[start:end]
            )
            data[idx, start:end] += gamma

    # Add Gaussian noise
    data += np.random.normal(0, noise_level, size=data.shape)

    # Common Average Referencing (CAR)
    mean_signal = data.mean(axis=0)
    data_car = data - mean_signal

    # Package into DataFrame
    df = pd.DataFrame(data_car.T, columns=[f"chan_{i+1}" for i in range(channels)])
    df.insert(0, "time", t)
    return df


if __name__ == "__main__":
    df_ecog = generate_synthetic_ecog()
    df_ecog.to_csv("mock_data/ecog/synthetic_ecog.csv", index=False)
    print("Saved synthetic ECoG to mock_data/ecog/synthetic_ecog.csv")

    # TODO: Add real ECoG data loader
    # e.g., from OpenNeuro BIDS importer or iEEG.org API
