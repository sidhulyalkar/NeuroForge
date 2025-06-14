import numpy as np
import pandas as pd

def generate_mock_eeg(freqs=[10, 20], noise_level=0.5, duration=10, fs=250):
    """
    Simulate EEG: sum of sinusoids + Gaussian noise.
    """
    t = np.linspace(0, duration, fs * duration)
    eeg = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    eeg += np.random.normal(0, noise_level, size=t.shape)
    return pd.DataFrame({"eeg": eeg, "time": t})

if __name__ == "__main__":
    df = generate_mock_eeg()
    df.to_csv("mock_data/eeg/synthetic_eeg.csv", index=False)
    print("Saved synthetic EEG to mock_data/eeg/synthetic_eeg.csv")
