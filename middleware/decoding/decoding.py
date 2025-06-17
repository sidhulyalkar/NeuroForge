# middleware/decoding/decoding.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier

class Decoder:
    def __init__(self, model_name: str = "RandomForestClassifier", **kwargs):
        if model_name == "RandomForestClassifier":
            self.model = RandomForestClassifier(**kwargs)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

def run(features_dict: dict, decoding_spec: dict, labels: np.ndarray = None):
    """
    Train or run inference on a decoding model.

    Parameters
    ----------
    features_dict : dict
        feature_name → ndarray (length = N samples)
    decoding_spec : dict
        {"model": "RandomForestClassifier", "task": "binary_classification", ...}
    labels : ndarray, optional
        True labels for training. If provided, returns a trained Decoder instance.
        If None, returns a default inference (zeros) array of shape (N,).

    Returns
    -------
    Decoder or ndarray
    """
    # Build feature matrix: shape (N, num_features)
    feat_names = list(features_dict.keys())
    arrs = [features_dict[name] for name in feat_names]
    X = np.vstack(arrs).T  # shape (N, num_features)

    if labels is not None:
        # Training mode
        decoder = Decoder(model_name=decoding_spec.get("model", "RandomForestClassifier"))
        decoder.fit(X, labels)
        return decoder
    else:
        # Inference without training: return all-zero predictions
        N = X.shape[0]
        return np.zeros(N, dtype=int)
