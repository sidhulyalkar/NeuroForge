import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
from agents.spec_agent import SpecAgent
from agents.code_agent import CodeAgent
from pipeline import run_full_pipeline

st.set_page_config(page_title="NeuroForge", layout="wide")
st.title("🧠 NeuroForge: BCI Middleware Builder")

tabs = st.tabs(["Spec", "Code", "Run"])

# --- Spec tab ---
with tabs[0]:
    uploaded = st.file_uploader("Upload hardware YAML spec", type=["yaml","yml"])
    if uploaded:
        content = uploaded.read().decode()
        spec = SpecAgent().run_from_content(content)
        st.subheader("Generated Pipeline Spec")
        st.json(spec)
    else:
        st.info("Upload a hardware spec to see the pipeline configuration.")

# --- Code tab ---
with tabs[1]:
    if 'spec' not in locals():
        st.info("First generate the spec in the Spec tab.")
    else:
        if st.button("Generate Code Stubs"):
            with st.spinner("Generating code…"):
                stubs = CodeAgent().generate_code(spec)
            for layer, code in stubs.items():
                st.markdown(f"#### `{layer}.py`")
                st.code(code, language="python")

# --- Run tab ---
with tabs[2]:
    st.subheader("Run Middleware on Synthetic Data")
    mode = st.radio("Select data type", ("EEG", "ECoG"))
    if st.button("Run Pipeline with Synthetic Labels"):
        # 1. Run the pipeline once to get raw data & times
        with st.spinner("Running first pass pipeline…"):
            first_pass = run_full_pipeline(
                "hardware_profiles/openbci_cyton.yaml",
                mode=mode,
                labels=None  # no training yet
            )
        arr, times = first_pass["raw"]  # extract raw data and times
        
        # 2. Build alternating labels: 0 for [0–1)s, 1 for [1–2)s, etc.
        labels = ((times // 1).astype(int) % 2)
        
        # 3. Rerun the pipeline with labels → returns a Decoder instance
        with st.spinner("Running synthetic label pipeline…"):
            result = run_full_pipeline(
                "hardware_profiles/openbci_cyton.yaml",
                mode=mode,
                labels=labels
            )
        decoder = result["predictions"]
        
        # 4. Predict on the same features
        feat_arr = np.vstack(list(result["features"].values())).T
        preds    = decoder.predict(feat_arr)
        
        # 5. Display “True vs. Predicted” over time
        df_cmp = pd.DataFrame({
            "True Label":      labels,
            "Predicted Label": preds
        }, index=times)
        st.subheader("True vs. Predicted Labels")
        st.line_chart(df_cmp)

    if st.button("Run Pipeline"):
        with st.spinner("Running pipeline…"):
            result = run_full_pipeline(
                "hardware_profiles/openbci_cyton.yaml",
                mode=mode
            )
        arr, times = result["raw"]
        # Suppose `times` is the Time vector from raw data:
        arr, times = result["raw"]
        n = len(times)
        # Create a label for every sample: 0 for first half, 1 for second half
        # or alternate every second:
        state = (times // 1).astype(int) % 2  # 0/1 alternating each second

        clean = result["clean"]

        # Build DataFrames indexed by time
        n_plot = min(3, arr.shape[0])
        raw_dict = {f"ch{i+1} (raw)": arr[i] for i in range(n_plot)}
        clean_dict = {f"ch{i+1} (clean)": clean[i] for i in range(n_plot)}
 
        raw_df = pd.DataFrame(raw_dict, index=times)
        clean_df = pd.DataFrame(clean_dict, index=times)

        # Plot first 3 channels raw vs clean
        st.subheader("Raw Signals (first 3 channels)")
        st.line_chart(raw_df)

        st.subheader("Cleaned Signals (first 3 channels)")
        st.line_chart(clean_df)

        # 4. Feature extraction
        with st.expander("Feature Extraction"):
            feat_dict = result["features"]
            # Map numeric band names back to standard labels
            band_label_map = {
                "band_1_4":   "delta (1-4Hz)",
                "band_4_8":   "theta (4-8Hz)",
                "band_8_12":  "alpha (8-12Hz)",
                "band_12_30": "beta (12-30Hz)",
                "band_70_150":"high-gamma (70-150Hz)",
            }
            # Build a new dict with renamed keys
            readable_bp = {
                band_label_map.get(k, k): v
                for k, v in feat_dict.items()
                if k.startswith("band_")
            }

            if readable_bp:
                means = {label: float(v.mean()) for label, v in readable_bp.items()}
                st.bar_chart(means)

            # Display first few feature values as table
            st.dataframe({k: v[:5] for k, v in feat_dict.items()})

            # If bandpower features exist, bar-chart their means
            bp_feats = {k: v for k, v in feat_dict.items() if k.startswith("band_")}
            if bp_feats:
                means = {k: float(v.mean()) for k, v in bp_feats.items()}
                st.bar_chart(means)

        # 5. Decoding
        with st.expander("Decoding Predictions"):
            preds = result["predictions"]
            st.write("Predictions:", preds)

        st.success("Pipeline complete!")
