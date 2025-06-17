# mock_data/eeg/generate_synthetic_eeg.py

import numpy as np
import pandas as pd

def generate_mock_eeg(channels=8, freqs=[10,20], noise_level=0.5, duration=10, fs=250):
    t = np.linspace(0, duration, fs * duration)
    data = np.zeros((len(t), channels))
    for ch in range(channels):
        # each channel gets a slightly different phase
        phase = np.random.rand() * 2*np.pi
        sig   = sum(np.sin(2*np.pi*f*t + phase) for f in freqs)
        data[:, ch] = sig + np.random.normal(0, noise_level, size=t.shape)
    df = pd.DataFrame(data, columns=[f"ch{c+1}" for c in range(channels)])
    df.insert(0, "time", t)
    return df

if __name__ == "__main__":
    df = generate_mock_eeg()
    df.to_csv("mock_data/synthetic_eeg.csv", index=False)
    print("Saved synthetic EEG with 8 channels.")
