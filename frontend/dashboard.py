import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
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
    if st.button("Run Pipeline"):
        with st.spinner("Running pipeline…"):
            result = run_full_pipeline(
                "hardware_profiles/openbci_cyton.yaml",
                mode=mode
            )
        arr, times = result["raw"]
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
