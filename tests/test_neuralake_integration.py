# tests/test_neuralake_integration.py
import os
import pytest
from middleware.data_layer.neuralake_catalog import BCI_CATALOG

@pytest.fixture(autouse=True)
def set_env(tmp_path, monkeypatch):
    # Point to a temp directory as data lake
    monkeypatch.setenv("DATA_LAKE_URI", str(tmp_path))
    # Create dummy parquet files for raw_eeg
    import pandas as pd
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=10, freq="S"),
        "channel_id": [0]*10,
        "voltage": [0.1]*10,
    })
    (tmp_path/"raw"/"eeg").mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp_path/"raw"/"eeg"/"data.parquet")

def test_raw_table_queryable():
    # Table has at least 10 rows
    df = BCI_CATALOG.db("bci").table("raw_eeg").collect()
    assert df.shape[0] == 10

def test_cleaned_table_runs():
    # Should run the preprocessing function
    df_clean = BCI_CATALOG.db("bci").table("cleaned_eeg").collect()
    assert "voltage" in df_clean.columns
