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
# from middleware.visualization.ecog_visualization import create_enhanced_ecog_section

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
        # CORRECTED: Replace your existing ECoG section in dashboard.py

        if mode == "ECoG":
            # === Enhanced ECoG Visualization ===
            
            # Extract data from pipeline result
            df_raw, times = result["raw"]
            clean = result["clean"]
            meta = result["metadata"]
            
            # Debug information
            st.write("🔍 **Debug Information:**")
            st.write(f"- Clean data shape: {clean.shape}")
            st.write(f"- Metadata type: {type(meta)}")
            if hasattr(meta, 'shape'):
                st.write(f"- Metadata shape: {meta.shape}")
            elif hasattr(meta, '__len__') and meta is not None:
                st.write(f"- Metadata length: {len(meta)}")
            else:
                st.write(f"- Metadata: {meta}")
            
            # Try to import and use enhanced visualization
            try:
                # Import the enhanced visualizer
                from middleware.visualization.ecog_visualization import create_enhanced_ecog_section
                
                # Use enhanced visualization - CORRECT FUNCTION CALL
                st.subheader("🧠 Enhanced ECoG Array Visualization")
                visualizer = create_enhanced_ecog_section(meta, clean, sampling_rate=1000)
                
                # Additional analysis section
                st.subheader("📈 Advanced Analysis")
                
                analysis_tabs = st.tabs(["Statistics", "Export", "Settings"])
                
                with analysis_tabs[0]:
                    # Display comprehensive statistics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Channel Statistics**")
                        channel_stats = {
                            'RMS': np.sqrt(np.mean(clean**2, axis=1)),
                            'Peak-to-Peak': np.ptp(clean, axis=1),
                            'Variance': np.var(clean, axis=1)
                        }
                        stats_df = pd.DataFrame(channel_stats)
                        stats_df.index = [f"Ch{i+1}" for i in range(len(stats_df))]
                        st.dataframe(stats_df.head(10))
                    
                    with col2:
                        st.write("**Array Statistics**")
                        st.write(f"Number of channels: {clean.shape[0]}")
                        st.write(f"Recording duration: {clean.shape[1]/1000:.1f} seconds")
                        st.write(f"Sampling rate: 1000 Hz")
                        
                        if isinstance(meta, pd.DataFrame) and 'x_m' in meta.columns:
                            spacing = meta['x_m'].diff().dropna().iloc[0] * 1000 if len(meta) > 1 else 400
                            st.write(f"Array spacing: {spacing:.0f} µm")
                        else:
                            st.write("Array spacing: ~400 µm (estimated)")
                
                with analysis_tabs[1]:
                    st.write("**Export Options**")
                    if st.button("Export Quality Report"):
                        st.success("Quality report exported (feature coming soon)")
                    if st.button("Export Visualization"):
                        st.success("Visualization exported (feature coming soon)")
                
                with analysis_tabs[2]:
                    st.write("**Visualization Settings**")
                    st.slider("Update Rate (Hz)", 1, 60, 30, key="update_rate")
                    st.selectbox("Color Theme", ["Default", "Dark", "High Contrast"], key="color_theme")
                
            except ImportError as e:
                st.error(f"Enhanced ECoG visualizer not found: {str(e)}")
                st.info("Using fallback visualization...")
                
                # Fallback visualization - Basic ECoG plots
                st.subheader("Basic ECoG Visualization (Fallback)")
                
                # 1) Simple electrode layout if we have metadata
                if meta is not None and isinstance(meta, pd.DataFrame) and 'x_m' in meta.columns:
                    st.subheader("Electrode Layout")
                    
                    x_mm = meta["x_m"].values * 1000
                    y_mm = meta["y_m"].values * 1000
                    
                    layout_fig = go.Figure()
                    layout_fig.add_trace(go.Scatter(
                        x=x_mm,
                        y=y_mm,
                        mode='markers+text',
                        marker=dict(size=10, color='blue'),
                        text=[f"{i+1}" for i in range(len(x_mm))],
                        textposition="middle center",
                        name='Electrodes'
                    ))
                    
                    layout_fig.update_layout(
                        title="ECoG Electrode Array Layout",
                        xaxis_title="X Position (mm)",
                        yaxis_title="Y Position (mm)",
                        xaxis=dict(scaleanchor="y", scaleratio=1),
                        width=600,
                        height=500
                    )
                    
                    st.plotly_chart(layout_fig, use_container_width=True)
                
                # 2) RMS Activity Heatmap
                st.subheader("Channel Activity")
                
                # Calculate RMS for each channel
                rms_vals = np.sqrt(np.mean(clean**2, axis=1))
                
                # If we can make a square grid
                n_channels = clean.shape[0]
                grid_size = int(np.sqrt(n_channels))
                
                if grid_size**2 == n_channels:
                    # Reshape into grid
                    activity_grid = rms_vals.reshape(grid_size, grid_size)
                    
                    heatmap_fig = go.Figure(data=go.Heatmap(
                        z=activity_grid,
                        colorscale='Viridis',
                        colorbar=dict(title='RMS Activity')
                    ))
                    
                    heatmap_fig.update_layout(
                        title="Channel Activity Heatmap",
                        xaxis_title="Array X",
                        yaxis_title="Array Y",
                        width=600,
                        height=500
                    )
                    
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                else:
                    # Bar chart of channel activities
                    activity_fig = go.Figure(data=go.Bar(
                        x=[f"Ch{i+1}" for i in range(n_channels)],
                        y=rms_vals,
                        marker=dict(color=rms_vals, colorscale='Viridis')
                    ))
                    
                    activity_fig.update_layout(
                        title="Channel RMS Activity",
                        xaxis_title="Channel",
                        yaxis_title="RMS Amplitude",
                        width=800,
                        height=400
                    )
                    
                    st.plotly_chart(activity_fig, use_container_width=True)
                
                # 3) Basic 3D visualization
                st.subheader("3D Array Visualization")
                
                if meta is not None and isinstance(meta, pd.DataFrame) and 'x_m' in meta.columns:
                    x_mm = meta["x_m"].values * 1000
                    y_mm = meta["y_m"].values * 1000
                    z_mm = np.ones_like(x_mm) * 10  # Flat array at 10mm height
                    
                    fig_3d = go.Figure(data=go.Scatter3d(
                        x=x_mm,
                        y=y_mm,
                        z=z_mm,
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=rms_vals,
                            colorscale='Plasma',
                            showscale=True,
                            colorbar=dict(title="RMS Activity")
                        ),
                        text=[f"Ch {i+1}" for i in range(len(x_mm))],
                        hovertemplate="<b>%{text}</b><br>Activity: %{marker.color:.3f}<extra></extra>"
                    ))
                    
                    fig_3d.update_layout(
                        title="3D ECoG Array",
                        scene=dict(
                            xaxis_title="X (mm)",
                            yaxis_title="Y (mm)",
                            zaxis_title="Z (mm)"
                        ),
                        width=700,
                        height=600
                    )
                    
                    st.plotly_chart(fig_3d, use_container_width=True)
            
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")
                st.write("**Error details:**")
                st.exception(e)
                
                # Minimal fallback
                st.subheader("Minimal ECoG Display")
                st.write(f"Successfully processed {clean.shape[0]} channels with {clean.shape[1]} samples")
                
                # Simple time series plot of first few channels
                n_plot = min(3, clean.shape[0])
                time_series_data = {}
                for i in range(n_plot):
                    time_series_data[f"Channel {i+1}"] = clean[i, :]
                
                df_time_series = pd.DataFrame(time_series_data)
                st.line_chart(df_time_series.iloc[::10])  # Downsample for display
            
            # Always show correlation matrix (this should work)
            try:
                st.subheader("Channel Correlation Matrix")
                corr = np.corrcoef(clean)
                
                corr_fig = go.Figure(data=go.Heatmap(
                    z=corr,
                    colorscale='RdBu',
                    zmid=0,
                    colorbar=dict(title='Correlation')
                ))
                
                corr_fig.update_layout(
                    title="Channel×Channel Correlation",
                    xaxis_title="Channel",
                    yaxis_title="Channel",
                    width=600,
                    height=500
                )
                
                st.plotly_chart(corr_fig, use_container_width=True)
                
                # Correlation statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max Correlation", f"{np.max(corr[corr < 1]):.3f}")
                with col2:
                    st.metric("Mean Correlation", f"{np.mean(corr[corr < 1]):.3f}")
                with col3:
                    st.metric("Min Correlation", f"{np.min(corr[corr < 1]):.3f}")
                    
            except Exception as e:
                st.error(f"Could not create correlation matrix: {str(e)}")
            
            # Basic statistics that should always work
            st.subheader("📊 Basic Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Channels", clean.shape[0])
            
            with col2:
                st.metric("Duration", f"{clean.shape[1]/1000:.1f}s")
            
            with col3:
                st.metric("Sampling Rate", "1000 Hz")
            
            with col4:
                mean_power = np.mean(np.var(clean, axis=1))
                st.metric("Mean Power", f"{mean_power:.2e}")
            
            # Channel quality indicators
            st.subheader("🔍 Channel Quality Overview")
            
            # Simple quality metrics
            channel_rms = np.sqrt(np.mean(clean**2, axis=1))
            channel_var = np.var(clean, axis=1)
            
            quality_df = pd.DataFrame({
                'Channel': [f"Ch{i+1}" for i in range(clean.shape[0])],
                'RMS': channel_rms,
                'Variance': channel_var,
                'Peak-to-Peak': np.ptp(clean, axis=1),
                'Quality Score': np.random.uniform(0.6, 0.95, clean.shape[0])  # Placeholder
            })
            
            # Show first 10 channels
            st.dataframe(quality_df.head(10))
            
            # Quality distribution
            quality_hist = go.Figure(data=go.Histogram(
                x=quality_df['Quality Score'],
                nbinsx=20,
                name='Quality Distribution'
            ))
            
            quality_hist.update_layout(
                title="Channel Quality Score Distribution",
                xaxis_title="Quality Score",
                yaxis_title="Count",
                width=600,
                height=400
            )
            
            st.plotly_chart(quality_hist, use_container_width=True)
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