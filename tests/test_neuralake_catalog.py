# /tests/test_neuralake_catalog.py

def test_raw_eeg_table(catalog_data_lake):
    # mock_data_lake now returns BCI_CATALOG pointed at the 400-row lake
    df = catalog_data_lake.db("bci").table("raw_eeg").collect()
    # 1s @100Hz × 4 channels = 400 rows
    assert df.shape[0] == 100 * 1 * 4
    # check a few channel columns exist
    assert set(df.columns) >= {"timestamp","voltage","channel_id"}
    

def test_raw_ecog_table(catalog_data_lake):
    df = catalog_data_lake.db("bci").table("raw_ecog").collect()
    assert df.shape[0] == 200 * 1 * 8
    assert "voltage" in df.columns