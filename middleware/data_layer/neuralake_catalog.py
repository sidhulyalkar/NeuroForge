# middleware/data_layer/neuralake_catalog.py
import os
import sys
import pyarrow as pa
import polars as pl
from neuralake.core import Catalog, ModuleDatabase, ParquetTable, table

DATA_LAKE_URI = os.environ.get("DATA_LAKE_URI", "./mock_data")

# 1) Define raw signal table (stored as Parquet files, no partitioning)
raw_eeg = ParquetTable(
    name="raw_eeg",
    uri=f"{DATA_LAKE_URI}/raw/eeg/",
    partitioning=[],                          
    description="Raw EEG time-series data",
    docs_columns=["timestamp", "channel_id", "voltage"],
)
raw_ecog = ParquetTable(
    name="raw_ecog",
    uri=f"{DATA_LAKE_URI}/raw/ecog/",
    partitioning=[],                           
    description="Raw ECoG time-series data",
    docs_columns=["timestamp"] + [f"chan_{i}" for i in range(1,33)],
)

# 2) Define cleaned/segmented signals via a table decorator
@table(
    description="Notch + bandpass + CAR cleaned signals",
    data_input="Processed from raw_eeg via NeuroForge preprocessing",
)
def cleaned_eeg() -> pl.LazyFrame:
    # read all the mock Parquet files, return a LazyFrame for downstream processing
    import polars as pl
    df = pl.read_parquet(f"{DATA_LAKE_URI}/raw/eeg/*.parquet")
    return df.lazy()

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
# pass the module object itself as the sole positional argument
dbs = {"bci": ModuleDatabase(sys.modules[__name__])}  # :contentReference[oaicite:0]{index=0}
 
BCI_CATALOG = Catalog(dbs)
