import os
import pytest
import pandas as pd
import importlib
from agents.neuralake_agent import NeuralakeAgent


@pytest.fixture(autouse=True)
def temp_data_lake(tmp_path, monkeypatch):
    from middleware.data_layer.neuralake_catalog import BCI_CATALOG

    # 1) point your catalog to a fresh temp dir
    monkeypatch.setenv("DATA_LAKE_URI", str(tmp_path))
    # 2) create a dummy raw_eeg Parquet table
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-06-01", periods=5, freq="S"),
            "channel_id": [0, 1, 0, 1, 0],
            "voltage": [0.1, 0.2, -0.1, 0.3, 0.0],
        }
    )
    raw_dir = tmp_path / "raw" / "eeg"
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_dir / "data.parquet")

    # 3) reload the catalog so it sees the new DATA_LAKE_URI
    import middleware.data_layer.neuralake_catalog as catmod

    importlib.reload(catmod)
    return catmod.BCI_CATALOG


def test_list_and_schema(agent_data_lake):
    agent = NeuralakeAgent(catalog=agent_data_lake)
    tables = agent.list_tables()
    assert "raw_eeg" in tables
    schema = agent.get_schema("raw_eeg")
    assert "voltage" in schema.names


def test_materialize_and_query(agent_data_lake, tmp_path):
    agent = NeuralakeAgent(catalog=agent_data_lake)
    out_uri = str(tmp_path / "materialized")
    agent.materialize("raw_eeg", out_uri, mode="overwrite")
    # read back via Polars to verify
    import polars as pl

    df = pl.read_parquet(out_uri)
    assert df.shape[0] == 5

    # now test a simple query
    result = agent.query(
        "raw_eeg", "SELECT channel_id, COUNT(*) AS cnt FROM raw_eeg GROUP BY channel_id"
    )
    # result should be a DataFrame-like with two rows (channel 0 & 1)
    assert hasattr(result, "to_pandas")
    pdf = result.to_pandas()
    assert set(pdf["channel_id"]) == {0, 1}


def test_export_and_roapi(agent_data_lake, tmp_path):
    # Site commented out for now...
    agent = NeuralakeAgent(catalog=agent_data_lake)
    # site_dir = tmp_path / "site"
    # agent.export_site(str(site_dir))
    # assert (site_dir / "index.html").exists()

    roapi_file = tmp_path / "roapi-config.yaml"
    agent.generate_roapi_config(output_file=str(roapi_file))
    assert roapi_file.exists()
