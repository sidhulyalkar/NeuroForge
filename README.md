# NeuroForge

NeuroForge is a modular framework for Brain–Computer Interface (BCI) development. It combines:

* **Spec Agents**: Generate pipeline specifications from hardware YAML specs.
* **Middleware**: Preprocessing, feature extraction, and decoding modules.
* **SDK Layer**: `BCIClient` wraps BrainFlow (or dummy) for streaming and real‑time processing.
* **Endpoint**: FastAPI server exposing `/predict` for inference.
* **Dashboard**: Streamlit UI to orchestrate, visualize, and interact with every layer.

---

## 📁 Repository Structure

```text
NeuroForge/
├── backend/
│   ├── pipeline.py               # Runs full pipeline (EEG/ECoG) end-to-end
│   └── hardware_profiles/        # YAML specs for hardware (e.g. openbci_cyton.yaml)
├── frontend/
│   └── dashboard.py              # Streamlit application
├── middleware/
│   ├── io/
│   │   └── signal_shape.py       # Validate shapes & sampling rates
│   ├── preprocessing/
│   │   └── preprocessing.py      # Notch, bandpass, CAR, normalization
│   ├── features/
│   │   └── features.py           # Bandpower, entropy, high-gamma extraction
│   ├── decoding/
│   │   └── decoding.py           # Fit/predict with scikit-learn models
│   └── sdk/
│       └── sdk.py                # BCIClient (BrainFlow wrapper + dummy modes)
├── mock_data/
│   ├── generate_synthetic_eeg.py # 8‑channel sinusoid + noise
│   └── generate_synthetic_ecog.py # 1024‑ch grid with bursts + metadata
├── tests/
│   ├── test_signal_shape.py      # PyTest for I/O layer
│   ├── test_preprocessing.py     # Preprocessing tests
│   ├── test_features.py          # Feature extraction tests
│   ├── test_decoding.py          # Decoder tests
│   ├── test_sdk.py               # SDK tests (connect, stream, get_features)
│   ├── test_endpoint.py          # FastAPI endpoint tests
│   └── test_realtime.py          # Real‑time pipeline tests
├── requirements.txt              # Python dependencies
└── README.md                     # This manual
```

---

## 🚀 Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/sidhulyalkar/NeuroForge.git
   cd NeuroForge
   ```

2. **Create & activate a virtualenv**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run tests**

   ```bash
   pytest -q
   ```

---

## 🏗️ Streamlit Dashboard Overview

Start the dashboard:

```bash
streamlit run frontend/dashboard.py
```

The UI is organized into **five tabs**:

1. **Spec**

   * Upload or select a hardware YAML spec.
   * The Code Agent generates a pipeline spec (preprocessing, features, decoding).

2. **Code**

   * View the scaffolded Python modules: `signal_shape.py`, `preprocessing.py`, etc.
   * Copy-paste or export them into your own repository.

3. **Run**

   * Choose **EEG** or **ECoG** mode.
   * For EEG: generates 8‑channel synthetic sine waves.
   * For ECoG: simulates a 32×32 grid with metadata.
   * Displays:

     * **Raw signals**
     * **Cleaned signals** (Notch + bandpass + CAR + z‑score)
     * **Feature plots** (bandpower & entropy)
     * **Decoding output** (predictions per channel)

4. **Hardware (SDK)**

   * `Connect`, `Start Stream`, `Stop Stream`, and **Show Buffer Shape**.
   * Uses `BCIClient(use_dummy=True)` by default. To enable real hardware, pass `use_dummy=False` and configure `serial_port`.

5. **API (/predict)**

   * Select mode, click **Call /predict**.
   * Sends POST to `http://localhost:8000/predict`.
   * Displays the JSON predictions array.

---

## 🛠️ Customizing for Your Pipelines

### Hardware Specs

* Add new `%PROFILE%.yaml` in `hardware_profiles/`:

  ```yaml
  name: my_device_pipeline
  input_channels: 16
  sampling_rate: 500
  preprocessing:
    - notch_filter: 60
    - bandpass_filter: [1, 100]
    - zscore_normalization: true
  features:
    - bandpower: ["delta","theta","alpha","beta"]
    - signal_entropy: true
  decoding:
    model: "LogisticRegression"
    task: "binary_classification"
  sdk:
    language: "python"
    endpoints: ["/connect","/start_stream","/get_features","/predict"]
  ```
* Reload the **Spec** tab and choose your new YAML.

### Custom Preprocessing & Features

* Edit `middleware/preprocessing/preprocessing.py` to add filters or artifact removal.
* Extend `middleware/features/features.py` with your own feature functions (e.g. cross-frequency coupling, instantaneous phase).

### Decoding Models

* In `middleware/decoding/decoding.py`, swap out `RandomForestClassifier` for any `sklearn` model (or wrap a PyTorch/TensorFlow model).
* Ensure `run()` returns either a fitted model or a NumPy array of predictions.

### Real‑Time Processing

* In **Hardware** tab, set `use_dummy=False` and specify your OpenBCI `serial_port` to stream from real hardware.
* In **Run** or a new **Real‑Time** tab, call:

  ```python
  client.start_processing_pipeline(
      fs=500,
      preprocess_fn=lambda raw, fs: preprocess(raw, fs, steps),
      feature_fn=lambda clean, fs: extract_features(clean, fs, features_spec),
      decode_fn=lambda feats, spec: decode(feats, spec),
      decoding_spec=decoding_spec,
      window_size_s=1.0,
      step_size_s=0.5,
      callback=my_callback
  )
  ```
* Implement `my_callback(preds, timestamp)` to visualize or log in real time.

---

## 📊 Visualizations

* **Time‑Series**: Raw & cleaned signals via `st.line_chart`.
* **Feature Bar Plot**: Bandpower & entropy per channel.
* **2D/3D Layout**: Electrode positions & topography (Matplotlib or Plotly).
* **Heatmaps**: RMS or correlation maps on grids.
* **Real-Time Charts**: Sliding‐window prediction traces.

![Pipeline Overview](assets/pipeline_overview.png)

> *Sample screenshot of the NeuroForge dashboard (time‑series, heatmap, and layout).*

---

## 🧪 Testing

Run the full pytest suite:

```bash
pytest -q
```

* **Unit tests** for each middleware layer (I/O, preprocessing, features, decoding).
* **SDK** tests for connect/stream/buffer, get\_features, predict.
* **Endpoint** tests with FastAPI TestClient.
* **Real‑Time** tests verifying sliding‐window callbacks.

---

## 📖 Further Reading

* [BrainFlow Documentation](https://brainflow.readthedocs.io/)
* [Streamlit Cheat Sheet](https://docs.streamlit.io/)
* [FastAPI Lifespan Events](https://fastapi.tiangolo.com/advanced/events/)

---

© 2025 Sidharth Hulyalkar. MIT License.
