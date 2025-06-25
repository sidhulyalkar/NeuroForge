# tests/test_neuralake_catalog.py
import pytest
import polars as pl

def test_raw_eeg_table(mock_data_lake):
    # RAW_EEG is now pointing at tmp_path/raw/eeg/
    from middleware.data_layer.neuralake_catalog import RAW_EEG
    df = RAW_EEG.read().collect()  # or RAW_EEG.to_lazy().collect()
    # We generated 1s × 100Hz × 4 channels = 400 rows
    assert df.shape[0] == 100 * 1 * 4
    assert set(df.columns) >= {"timestamp", "ch1", "ch2", "ch3", "ch4"}

def test_raw_ecog_table(mock_data_lake):
    from middleware.data_layer.neuralake_catalog import ECOG_TABLE  # define analogously
    df = ECOG_TABLE.read().collect()
    # should match 1s × 200Hz × 8 channels = 1600 rows
    assert df.shape[0] == 200 * 1 * 8
    assert "chan_1" in df.columns
