# mock_data/ecog/generate_ecog_parquet.py

import os
import argparse
import numpy as np
import pandas as pd

def generate_ecog(
    channels=32,
    fs=1000,
    duration=10,
    low_freqs=[10, 20, 40],
    high_gamma_band=(70, 150),
    noise_level=0.2,
):
    t = np.linspace(0, duration, int(fs * duration))
    data = np.zeros((channels, len(t)))

    # Low-frequency rhythms
    for idx in range(channels):
        for f in low_freqs:
            data[idx] += np.sin(2 * np.pi * f * t)

    # High-gamma bursts
    for idx in np.random.choice(channels, size=channels // 4, replace=False):
        bursts = np.random.choice(len(t), size=5, replace=False)
        for b in bursts:
            start = b
            end = min(b + fs // 10, len(t))
            gamma = np.sin(
                2 * np.pi * np.random.uniform(*high_gamma_band) * t[start:end]
            )
            data[idx, start:end] += gamma

    # Add noise & CAR
    data += np.random.normal(0, noise_level, size=data.shape)
    data_car = data - data.mean(axis=0)

    # Build DataFrame
    df = pd.DataFrame(data_car.T, columns=[f"chan_{i+1}" for i in range(channels)])
    df.insert(0, "timestamp", pd.date_range("2025-01-01", periods=len(t), freq=f"{1000/fs}ms"))
    return df

def main(out_dir, **gen_kwargs):
    os.makedirs(out_dir, exist_ok=True)
    df = generate_ecog(**gen_kwargs)
    out_path = os.path.join(out_dir, "synthetic_ecog.parquet")
    df.to_parquet(out_path, index=False)
    print(f"✅ Wrote {df.shape[0]}×{df.shape[1]} to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate synthetic ECoG and write to Parquet"
    )
    p.add_argument("--out-dir", required=True,
                   help="Where to write .parquet (e.g. $DATA_LAKE_URI/raw/ecog)")
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--fs", type=int, default=1000)
    p.add_argument("--duration", type=float, default=10)
    p.add_argument("--noise-level", type=float, default=0.2)
    p.add_argument("--low-freqs", nargs="+", type=float, default=[10,20,40])
    p.add_argument("--high-gamma-band", nargs=2, type=float, default=[70,150])
    args = p.parse_args()
    # unpack high_gamma_band tuple
    kwargs = dict(
        channels=args.channels,
        fs=args.fs,
        duration=args.duration,
        low_freqs=args.low_freqs,
        high_gamma_band=tuple(args.high_gamma_band),
        noise_level=args.noise_level,
    )
    main(args.out_dir, **kwargs)
