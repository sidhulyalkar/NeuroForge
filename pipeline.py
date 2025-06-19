# pipeline.py
import numpy as np
from agents.spec_agent import SpecAgent
from middleware.io.signal_shape import run as validate_shape
from middleware.preprocessing.preprocessing import run as preprocess
from middleware.features.features import run as extract_features
from middleware.decoding.decoding import run as decode
from mock_data.eeg.generate_synthetic_eeg import generate_mock_eeg
from mock_data.ecog.generate_synthetic_ecog import generate_synthetic_ecog
from mock_data.ecog.generate_precision_ecog import generate_precision_ecog


def run_full_pipeline(yaml_path, mode="EEG", labels=None):
    # 1. Load the hardware YAML directly
    spec_agent = SpecAgent(model_name="gpt-4", temperature=0.0)
    hw_spec = spec_agent.load_spec(yaml_path)

    # 2. Also get the LLM‐generated pipeline spec if you need it later
    pipeline_spec = spec_agent.generate_pipeline_spec(hw_spec)

    # 3. Determine data & steps from the hardware spec
    if mode == "EEG":
        df = generate_mock_eeg()
        steps = hw_spec.get("preprocessing", [])
        fs = hw_spec.get("sampling_rate", 250)
        meta = []
    else:
        df, meta = generate_precision_ecog()
        print(f"DEBUG: df shape = {df.shape}")
        # For ECoG add CAR + high‐gamma on top of the YAML steps
        steps = hw_spec.get("preprocessing", []) + [
            {"car": True},
            {"highpass_filter": 70},
        ]
        # Force ECoG sampling rate to 1000 Hz to match Precision Style Electrode Array
        fs = 1000

    # 4. Validate & preprocess
    arr, times = validate_shape(df, expected_channels=df.shape[1] - 1, expected_fs=fs)
    clean = preprocess(arr, fs, steps)

    raw_feats_spec = hw_spec.get("features", [])
    normalized_feats = []
    for f in raw_feats_spec:
        if isinstance(f, str):
            normalized_feats.append({f: True})
        elif isinstance(f, dict):
            normalized_feats.append(f)

    # 5. Features & decoding
    feats = extract_features(clean, fs, normalized_feats)

    # Decode or default if no features
    if feats:
        labels = np.tile([0, 1], clean.shape[1] // 2 + 1)[: clean.shape[1]]
        # decoder = decode(feats, hw_spec.get("decoding", {}), labels=labels)
        preds_or_model = decode(feats, pipeline_spec.get("decoding", {}), labels=labels)
        # preds = decode(feats, hw_spec.get("decoding", {}))
    else:
        preds = np.zeros(clean.shape[1], dtype=int)

    return {
        "spec": pipeline_spec,
        "hw_spec": hw_spec,
        "raw": (arr, times),
        "clean": clean,
        "features": feats,
        "predictions": preds_or_model,
        "metadata": meta,
    }


if __name__ == "__main__":
    out = run_full_pipeline("hardware_profiles/openbci_cyton.yaml", mode="EEG")
    print("Predictions:", out["predictions"])
