# mock_data/eeg/generate__eeg_parquet.py

import os
import argparse
import numpy as np
import pandas as pd

def generate_eeg(channels=8, freqs=[10,20], noise_level=0.5, duration=10, fs=250):
    t = np.linspace(0, duration, fs * duration)
    data = np.zeros((len(t), channels))
    for ch in range(channels):
        phase = np.random.rand() * 2*np.pi
        sig   = sum(np.sin(2*np.pi*f*t + phase) for f in freqs)
        data[:, ch] = sig + np.random.normal(0, noise_level, size=t.shape)
    df = pd.DataFrame(data, columns=[f"ch{c+1}" for c in range(channels)])
    df.insert(0, "timestamp", pd.date_range("2025-01-01", periods=len(t), freq=f"{1000/fs}ms"))
    return df

def main(out_dir, **gen_kwargs):
    os.makedirs(out_dir, exist_ok=True)
    df = generate_eeg(**gen_kwargs)
    out_path = os.path.join(out_dir, "synthetic_eeg.parquet")
    # Write Parquet
    df.to_parquet(out_path, index=False)
    print(f"✅ Wrote {df.shape[0]}×{df.shape[1]} to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate synthetic EEG and write to Parquet"
    )
    p.add_argument("--out-dir", required=True,
                   help="Where to write .parquet (e.g. $DATA_LAKE_URI/raw/eeg)")
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--duration", type=float, default=10)
    p.add_argument("--fs", type=int, default=250)
    p.add_argument("--noise-level", type=float, default=0.5)
    p.add_argument("--freqs", nargs="+", type=float, default=[10,20])
    args = p.parse_args()
    main(
        args.out_dir,
        channels=args.channels,
        freqs=args.freqs,
        noise_level=args.noise_level,
        duration=args.duration,
        fs=args.fs,
    )
