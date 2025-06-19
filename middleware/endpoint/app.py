# middleware/endpoint/app.py

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from middleware.sdk.sdk import BCIClient
from middleware.preprocessing.preprocessing import run as preprocess
from middleware.features.features import run as extract_features
from middleware.decoding.decoding import run as decode

# Shared ML client instance
client = BCIClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Connect to the BCI hardware when the application starts,
    and clean up (disconnect) when it shuts down.
    """
    try:
        client.connect()
        logging.getLogger("uvicorn.access").info("BCIClient connected successfully.")
    except Exception as e:
        logging.getLogger("uvicorn.error").warning(
            f"BCIClient.connect() failed at startup: {e}"
        )
    yield
    # Shutdown / cleanup
    try:
        client.disconnect()
        logging.getLogger("uvicorn.access").info("BCIClient disconnected successfully.")
    except Exception as e:
        logging.getLogger("uvicorn.error").warning(
            f"BCIClient.disconnect() failed at shutdown: {e}"
        )


# Create the FastAPI app with our lifespan manager
app = FastAPI(lifespan=lifespan)


class PredictRequest(BaseModel):
    mode: str  # "EEG" or "ECoG"


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    Run the full pipeline on the latest buffer and return predictions.
    """
    # Choose sampling rate and preprocessing steps based on mode
    if req.mode.upper() == "EEG":
        fs = 250
        steps = [
            {"notch_filter": 60},
            {"bandpass_filter": [1, 40]},
            {"zscore_normalization": True},
        ]
    else:
        fs = 1000
        steps = [
            {"notch_filter": 60},
            {"bandpass_filter": [1, 40]},
            {"zscore_normalization": True},
            {"car": True},
            {"highpass_filter": 70},
        ]

    # 1) Acquire data from the board buffer
    raw = client.get_buffer()  # shape: (channels, samples)
    if raw.size == 0 or raw.shape[1] == 0:
        return {"predictions": []}

    # 2) Preprocess
    clean = preprocess(raw, fs, steps)

    # 3) Feature extraction
    feats = extract_features(
        clean,
        fs,
        [{"bandpower": [[1, 4], [4, 8], [8, 12], [12, 30]]}, {"signal_entropy": True}],
    )

    # 4) Decode (inference only)
    preds = decode(feats, {"model": "RandomForestClassifier"})

    return {"predictions": preds.tolist()}
