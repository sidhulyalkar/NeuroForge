# File: integrated_pipeline.py

import os
from pathlib import Path
from agents import (
    DataIngestAgent,
    PreprocessAgent,
    FeatureAgent,
    TrainAgent,
    InferenceAgent,
    VisualizeAgent,
)
from middleware.data_layer.neuralake_catalog import (
    BCI_CATALOG,
    RAW_EEG,
    cleaned_eeg,
    features,
)

# from neuralake.export.web import export_and_generate_site
from neuralake.export import roapi

# 1. Configuration
DATA_LAKE_URI = os.getenv("DATA_LAKE_URI", "s3://my-bci-lake")
CATALOG_SITE_DIR = os.getenv("CATALOG_SITE_DIR", "./neuralake_site")
os.makedirs(CATALOG_SITE_DIR, exist_ok=True)


# 2. Step 1: Ingest raw BCI data
def ingest_dataset(dataset_path: str):
    ingest = DataIngestAgent(
        catalog=BCI_CATALOG,
        raw_table=RAW_EEG,
        source=dataset_path,
        target_uri=f"{DATA_LAKE_URI}/raw/eeg/",
    )
    return ingest.run()


# 3. Step 2: Preprocessing
def run_preprocessing():
    pp = PreprocessAgent(
        catalog=BCI_CATALOG,
        input_table="raw_eeg",
        output_table="cleaned_eeg",
        parameters={"notch": [50], "bandpass": [1, 100]},
    )
    return pp.run()


# 4. Step 3: Feature extraction
def extract_features_workflow():
    fa = FeatureAgent(
        catalog=BCI_CATALOG,
        input_table="cleaned_eeg",
        output_table="features",
        specs={"bands": ["theta", "beta", "gamma"]},
    )
    return fa.run()


# 5. Step 4: Model training & inference
def train_and_infer():
    trainer = TrainAgent(
        catalog=BCI_CATALOG,
        feature_table="features",
        model_name="neuro_dl_classifier",
        hyperparameters={"epochs": 50, "batch_size": 32},
    )
    model = trainer.run()

    infer = InferenceAgent(
        catalog=BCI_CATALOG,
        model=model,
        input_table="features",
        result_table="predictions",
    )
    return infer.run()


# 6. Step 5: Visualization & report
def visualize_results():
    vz = VisualizeAgent(
        catalog=BCI_CATALOG, tables=["predictions", "features"], output_dir="./reports"
    )
    return vz.run()


# 7. Step 6: Export Neuralake site & ROAPI
def export_api():
    # export_and_generate_site(
    #     catalogs=[("bci", BCI_CATALOG)],
    #     output_dir=CATALOG_SITE_DIR
    # )
    roapi.generate_config(BCI_CATALOG, output_file="roapi-config.yaml")


# 8. Orchestrator
if __name__ == "__main__":
    import sys

    ds_path = sys.argv[1]  # e.g. path to PhysioNet files
    ingest_dataset(ds_path)
    run_preprocessing()
    extract_features_workflow()
    train_and_infer()
    visualize_results()
    export_api()
    print(
        "Pipeline complete—Neuralake catalog, model outputs, and dashboards generated."
    )
