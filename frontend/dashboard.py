import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import requests
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px

from agents.spec_agent import SpecAgent
from agents.code_agent import CodeAgent
from pipeline import run_full_pipeline

# Silence brainflow’s “serial port is empty” errors
logging.getLogger("board_logger").setLevel(logging.WARNING)

st.set_page_config(page_title="NeuroForge", layout="wide")
st.title("🧠 NeuroForge: BCI Middleware Builder")

tabs = st.tabs(["Spec", "Code", "Run", "Hardware", "API"])

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

    if st.button("Run Pipeline with Synthetic Channel Labels"):
        # 1. Grab raw to get channel count
        with st.spinner("Running raw pipeline…"):
            out0   = run_full_pipeline("hardware_profiles/openbci_cyton.yaml", mode=mode, labels=None)
        arr, _ = out0["raw"]
        st.write(f"DEBUG: arr.shape = {arr.shape}")

        # 2. Create per‐channel labels
        n_channels = arr.shape[0]
        labels     = np.array([i % 2 for i in range(n_channels)])
        st.write(f"DEBUG: labels.shape = {labels.shape}")

        # 3. Run pipeline *with* labels
        with st.spinner("Running pipeline with labels…"):
            result = run_full_pipeline("hardware_profiles/openbci_cyton.yaml", mode=mode, labels=labels)
        preds_or_model = result["predictions"]
        feats         = result["features"]

        # 4. Build feature matrix: one row per channel
        feat_arr = np.vstack(list(feats.values())).T  # shape (n_channels, n_features)

        # 5. Dispatch on what `predictions` gave us
        if hasattr(preds_or_model, "predict"):
            # it’s a Decoder
            preds = preds_or_model.predict(feat_arr)
        else:
            # it’s already an array of predictions
            preds = preds_or_model

        # 6. Display per‐channel comparison
        df_cmp = pd.DataFrame({
            "True": labels,
            "Pred": preds
        }, index=[f"ch{i+1}" for i in range(n_channels)])
        st.subheader("Per-Channel True vs. Predicted Labels")
        st.dataframe(df_cmp)

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

        if mode == "ECoG":
            # === unpack what pipeline returned ===
            df_raw, times = result["raw"]
            clean         = result["clean"]
            meta          = result["metadata"]

            # # Quick sanity check
            # st.write(f"DEBUG: clean.shape = {clean.shape}, meta.shape = {meta.shape}")
            # st.write("⚙️ DEBUG result keys:", list(result.keys()))
            # st.write("⚙️ DEBUG meta.shape:", None if meta is None else meta.shape)
            # st.write("⚙️ DEBUG meta.head():", None if meta is None else meta.head())


            # 1) 2D Scatter of Electrode Layout
            st.subheader("ECoG Electrode Layout (Matplotlib)")

            x_mm    = meta["x_m"].values * 1e3
            y_mm    = meta["y_m"].values * 1e3
            diam_um = meta["diameter_m"].values * 1e6

            fig1, ax1 = plt.subplots(figsize=(6,6))
            sc = ax1.scatter(
                x_mm, y_mm,
                s=(diam_um / diam_um.max() * 200) + 5,
                c=diam_um,
                cmap="turbo",
                alpha=0.7,
                edgecolors="k",
            )
            fig1.colorbar(sc, ax=ax1, label="Electrode Ø (µm)")
            ax1.set_xlabel("X (mm)")
            ax1.set_ylabel("Y (mm)")
            ax1.set_title("32×32 ECoG Grid (400 µm spacing)")
            ax1.set_aspect("equal", "box")
            st.pyplot(fig1)

            # 2) Heatmap of Mean RMS
            st.subheader("Mean RMS Topography (Matplotlib)")

            n_chan, _ = clean.shape
            window = 100
            rms = (clean.reshape(n_chan, -1, window)**2).mean(axis=2)**0.5
            mean_rms = rms.mean(axis=1)

            grid = int(np.sqrt(n_chan))
            topo = mean_rms.reshape(grid, grid)

            fig2, ax2 = plt.subplots(figsize=(6,5))
            im = ax2.imshow(topo, origin="lower", cmap="viridis",
                            extent=[0, grid*0.4, 0, grid*0.4])
            fig2.colorbar(im, ax=ax2, label="RMS amplitude")
            ax2.set_xlabel("X (mm)")
            ax2.set_ylabel("Y (mm)")
            ax2.set_title("Mean RMS on 32×32 Grid")
            st.pyplot(fig2)

            # 3) Static 3D Scatter on Hemisphere
            st.subheader("3D ECoG Positions (Matplotlib)")

            fig3 = plt.figure(figsize=(6,6))
            ax3 = fig3.add_subplot(111, projection='3d')

            # hemisphere
            cx, cy = x_mm.mean(), y_mm.mean()
            radius = max(np.hypot(x_mm-cx, y_mm-cy)) * 1.1
            u = np.linspace(0, 2*np.pi, 60)
            v = np.linspace(0, np.pi/2, 30)
            uu, vv = np.meshgrid(u, v)
            X = radius*np.cos(uu)*np.sin(vv)
            Y = radius*np.sin(uu)*np.sin(vv)
            Z = radius*np.cos(vv)
            ax3.plot_surface(X, Y, Z, rstride=2, cstride=2, alpha=0.3, color="lightgray", linewidth=0)

            # electrodes
            ax3.scatter(
                x_mm-cx, y_mm-cy, np.ones_like(x_mm)*2,
                s=10, c=diam_um, cmap="turbo"
            )
            ax3.set_axis_off()
            ax3.set_title("ECoG Array on Approximate Cortex")
            st.pyplot(fig3)


        # Compute correlation matrix: shape (channels, channels)
        corr = np.corrcoef(clean)


        fig, ax = plt.subplots(figsize=(6,6))
        cax = ax.matshow(corr, vmin=-1, vmax=1)
        fig.colorbar(cax, ax=ax)
        ax.set_title("Channel×Channel Correlation")
        ax.set_xticks([])
        ax.set_yticks([])
        st.pyplot(fig)

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

# --- Hardware (SDK) Tab ---
with tabs[3]:
    st.header("SDK: BCI Client")
    from middleware.sdk.sdk import BCIClient

    client = BCIClient()
    if st.button("Connect"):
        try:
            client.connect()
            st.success("Connected to board")
        except Exception as e:
            st.error("Could not connect: {e}")
            
    if st.button("Start Stream"):
        try:
            client.start_stream(sampling_rate=250, packet_size=450)
            st.info("Streaming...")
        except Exception as e:
            st.error("Stream failed to start: {e}")

    if st.button("Stop Stream"):
        try:
            client.stop_stream()
            st.info("Stopped streaming")
        except Exception as e:
            st.error("Stream is already stopped: {e}")

    if st.button("Get Buffer Length"):
        try:
            buf = client.get_buffer()
            st.write(f"Buffered shape: {buf.shape}")
        except Exception as e:
            st.error("Could not retrieve buffer: {e}")

# --- API (Endpoint) Tab ---
with tabs[4]:
    st.header("Endpoint: /predict")
    mode = st.selectbox("Mode", ["EEG","ECoG"])
    if st.button("Call /predict"):
        try:
            resp = requests.post("http://localhost:8000/predict", json={"mode": mode})
            resp.raise_for_status()
            st.json(resp.json())
        except Exception as e:
            st.error(f"Request failed: {e}")