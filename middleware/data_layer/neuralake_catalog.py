# neuralake_catalog.py
import os
import pyarrow as pa
import polars as pl
from neuralake.core import (
    Catalog,
    ModuleDatabase,
    ParquetTable,
    DeltalakeTable,
    table,
    Filter,
)

# 1) Define raw signal table (stored as Parquet or Delta files)
DATA_LAKE_URI = os.environ.get("DATA_LAKE_URI", "./mock_data")
RAW_EEG = ParquetTable(
    name="raw_eeg",
    uri=f"{DATA_LAKE_URI}/raw/eeg/",
    schema=pa.schema([
        ("timestamp", pa.timestamp("ns")),
        ("channel_id", pa.int32()),
        ("voltage", pa.float32()),
    ]),
    description="Raw EEG time-series data",
    unique_columns=["timestamp", "channel_id"],
)  # :contentReference[oaicite:4]{index=4}

# 2) Define cleaned/segmented signals via a table decorator
@table(
    description="Notch + bandpass + CAR cleaned signals",
    data_input="Processed from raw_eeg via NeuroForge preprocessing",
)
def cleaned_eeg() -> pl.LazyFrame:
    df = RAW_EEG.lazy()
    # Reuse NeuroForge preprocessing functions
    from middleware.preprocessing.preprocessing import preprocess
    return preprocess(df, fs=int(os.getenv("SAMPLING_RATE", 500)), steps=None)

# 3) Define feature extraction table
@table(
    description="Bandpower & entropy features per window",
    data_input="Computed from cleaned_eeg using neuro-DL-Classifier preprocessing",
)
def features() -> pl.LazyFrame:
    cln = cleaned_eeg()
    from middleware.features.features import extract_features
    return extract_features(cln, fs=int(os.getenv("SAMPLING_RATE", 500)), specs=None)

# 4) Assemble catalog
dbs = {"bci": ModuleDatabase(module=__import__(__name__))}
BCI_CATALOG = Catalog(dbs)
