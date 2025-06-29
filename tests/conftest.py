# tests/conftest.py

import os
import importlib
from pathlib import Path

import pandas as pd
import pytest


def _reload_catalog(tmp_path):
    """Helper: set DATA_LAKE_URI and reload catalog module."""
    os.environ["DATA_LAKE_URI"] = str(tmp_path)
    import middleware.data_layer.neuralake_catalog as C

    importlib.reload(C)
    return C.BCI_CATALOG


@pytest.fixture
def catalog_data_lake(tmp_path):
    """
    400-row EEG + 1600-row ECoG → for test_neuralake_catalog.py
    """
    # EEG: 1s @100Hz × 4ch = 400 rows
    eeg_dir = tmp_path / "raw" / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    t0 = pd.Timestamp("2025-06-01")
    ts = pd.date_range(t0, periods=100, freq="10ms")
    rows = [
        {"timestamp": t, "channel_id": ch, "voltage": 0.0}
        for t in ts
        for ch in range(4)
    ]
    pd.DataFrame(rows).to_parquet(eeg_dir / "data.parquet", index=False)

    # ECoG: 1s @200Hz × 8ch = 1600 rows
    ecog_dir = tmp_path / "raw" / "ecog"
    ecog_dir.mkdir(parents=True, exist_ok=True)
    ts2 = pd.date_range(t0, periods=200, freq="5ms")
    rows2 = [
        {"timestamp": t, "channel_id": ch, "voltage": 0.0}
        for t in ts2
        for ch in range(8)
    ]
    pd.DataFrame(rows2).to_parquet(ecog_dir / "data.parquet", index=False)

    return _reload_catalog(tmp_path)


@pytest.fixture
def agent_data_lake(tmp_path):
    """
    5-row EEG → for test_neuralake_agent.py
    """
    eeg_dir = tmp_path / "raw" / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-06-01", periods=5, freq="S"),
            "channel_id": [0, 1, 0, 1, 0],
            "voltage": [0.1, 0.2, -0.1, 0.3, 0.0],
        }
    )
    df.to_parquet(eeg_dir / "data.parquet", index=False)

    return _reload_catalog(tmp_path)


@pytest.fixture
def integration_data_lake(tmp_path):
    """
    10-row EEG → for test_neuralake_integration.py
    """
    eeg_dir = tmp_path / "raw" / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-06-01", periods=10, freq="100ms"),
            "channel_id": [i % 2 for i in range(10)],
            "voltage": 0.0,
        }
    )
    df.to_parquet(eeg_dir / "data.parquet", index=False)

    return _reload_catalog(tmp_path)
