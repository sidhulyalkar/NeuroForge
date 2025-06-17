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

# def run(features_dict: dict, decoding_spec: dict, labels: np.ndarray = None):
#     """
#     Train or run inference on a decoding model.

#     Parameters
#     ----------
#     features_dict : dict
#         feature_name → ndarray (length = N samples)
#     decoding_spec : dict
#         {"model": "RandomForestClassifier", "task": "binary_classification", ...}
#     labels : ndarray, optional
#         True labels for training. If provided, returns a trained Decoder instance.
#         If None, returns a default inference (zeros) array of shape (N,).

#     Returns
#     -------
#     Decoder or ndarray
#     """
#     # Build feature matrix: shape (N, num_features)
#     feat_names = list(features_dict.keys())
#     arrs = [features_dict[name] for name in feat_names]
#     X = np.vstack(arrs).T  # shape (n_samples, n_features)

#     if labels is not None:
#         if len(labels) != X.shape[0]:
#             raise ValueError(
#                 f"Number of labels ({len(labels)}) must match number of samples ({X.shape[0]})"
#             )
#         decoder = Decoder(model_name=decoding_spec.get("model","RF"))
#         decoder.fit(X, labels)
#         return decoder
#     else:
#         # inference‐only fallback
#         return np.zeros(X.shape[0], dtype=int)

def run(features_dict: dict, decoding_spec: dict, labels=None):
    """
    Train or run inference on a decoding model.

    If labels are provided *and* their length matches the number
    of samples (rows in the feature matrix), the model will
    be trained and returned. Otherwise, returns a zero‐array
    of length = n_samples.
    """
    feat_names = list(features_dict.keys())
    arrs = [features_dict[name] for name in feat_names]
    
    # If no features, immediately fallback
    if len(arrs) == 0:
        return np.zeros(1 if labels is None else len(labels), dtype=int)
    
    X = np.vstack(arrs).T  # shape (n_samples, n_features)
    n_samples = X.shape[0]

    # If valid labels, train
    if labels is not None and len(labels) == n_samples:
        decoder = Decoder(model_name=decoding_spec.get("model", "RandomForestClassifier"))
        decoder.fit(X, labels)
        return decoder

    # Otherwise, fallback to zeros (inference)
    return np.zeros(n_samples, dtype=int)