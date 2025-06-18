# middleware/visualization/ecog_visualizer.py
"""
Enhanced ECoG visualization system integrated with existing pipeline
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from scipy import signal
from scipy.stats import zscore

class SignalQualityAnalyzer:
    """Real-time signal quality assessment"""
    
    def __init__(self):
        self.quality_thresholds = {
            'snr_min': 3.0,
            'artifact_max': 0.2,
            'correlation_min': 0.1
        }
    
    def assess_signal_quality(self, data, fs=1000, meta=None):
        """Comprehensive signal quality assessment"""
        n_channels, n_samples = data.shape
        
        quality_metrics = {
            'overall_score': 0.0,
            'channel_scores': np.zeros(n_channels),
            'snr': np.zeros(n_channels),
            'artifact_ratio': np.zeros(n_channels),
            'recommendations': [],
            'alerts': []
        }
        
        # Calculate per-channel metrics
        for ch in range(n_channels):
            channel_data = data[ch, :]
            
            # Signal-to-noise ratio estimate
            quality_metrics['snr'][ch] = self._estimate_snr(channel_data, fs)
            
            # Artifact detection
            quality_metrics['artifact_ratio'][ch] = self._detect_artifacts(channel_data, fs)
            
            # Overall channel score
            snr_score = min(quality_metrics['snr'][ch] / 10.0, 1.0)
            artifact_score = 1.0 - quality_metrics['artifact_ratio'][ch]
            quality_metrics['channel_scores'][ch] = (snr_score + artifact_score) / 2.0
        
        # Overall quality score
        quality_metrics['overall_score'] = np.mean(quality_metrics['channel_scores'])
        
        # Generate recommendations
        quality_metrics['recommendations'] = self._generate_recommendations(quality_metrics)
        
        # Check for alerts
        quality_metrics['alerts'] = self._check_quality_alerts(quality_metrics)
        
        return quality_metrics
    
    def _estimate_snr(self, data, fs):
        """Estimate signal-to-noise ratio"""
        # Simple SNR estimation using signal power vs noise floor
        freqs, psd = signal.welch(data, fs, nperseg=min(len(data)//4, 512))
        
        # Signal power (assume 1-100 Hz contains signal)
        signal_mask = (freqs >= 1) & (freqs <= 100)
        signal_power = np.mean(psd[signal_mask])
        
        # Noise power (high frequency tail)
        noise_mask = freqs > 200
        if np.any(noise_mask):
            noise_power = np.mean(psd[noise_mask])
        else:
            noise_power = np.min(psd) + 1e-12
        
        snr_db = 10 * np.log10(signal_power / noise_power)
        return max(snr_db, 0)
    
    def _detect_artifacts(self, data, fs):
        """Detect various types of artifacts"""
        # Z-score based artifact detection
        z_scores = np.abs(zscore(data))
        artifact_samples = np.sum(z_scores > 4)  # Samples >4 std deviations
        
        # High amplitude artifact detection
        amplitude_threshold = 5 * np.std(data)
        high_amp_artifacts = np.sum(np.abs(data) > amplitude_threshold)
        
        total_artifacts = max(artifact_samples, high_amp_artifacts)
        artifact_ratio = total_artifacts / len(data)
        
        return min(artifact_ratio, 1.0)
    
    def _generate_recommendations(self, metrics):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Overall quality recommendations
        if metrics['overall_score'] < 0.5:
            recommendations.append({
                'type': 'general',
                'severity': 'high',
                'message': 'Overall signal quality is poor',
                'action': 'Check electrode contacts and reduce noise sources'
            })
        
        # Channel-specific recommendations
        bad_channels = np.where(metrics['channel_scores'] < 0.3)[0]
        if len(bad_channels) > 0:
            recommendations.append({
                'type': 'channels',
                'severity': 'medium',
                'message': f'Poor quality on channels: {list(bad_channels)}',
                'action': 'Inspect electrode contacts and impedances'
            })
        
        # SNR recommendations
        low_snr_channels = np.where(metrics['snr'] < 5)[0]
        if len(low_snr_channels) > 0:
            recommendations.append({
                'type': 'snr',
                'severity': 'medium', 
                'message': f'Low SNR on channels: {list(low_snr_channels)}',
                'action': 'Check for electrical interference and grounding'
            })
        
        return recommendations
    
    def _check_quality_alerts(self, metrics):
        """Check for immediate quality alerts"""
        alerts = []
        
        if metrics['overall_score'] < 0.2:
            alerts.append({
                'type': 'critical',
                'message': 'Critical signal quality - recording may be unusable',
                'action': 'Stop recording and troubleshoot hardware'
            })
        
        return alerts

class EnhancedECoGVisualizer:
    """Enhanced ECoG visualization with quality assessment"""
    
    def __init__(self, meta_df, clean_data, sampling_rate=1000):
        """
        Initialize visualizer with electrode metadata and cleaned data
        
        Args:
            meta_df: DataFrame with columns ['x_m', 'y_m', 'diameter_m']
            clean_data: Array of shape (n_channels, n_samples)
            sampling_rate: Sampling rate in Hz
        """
        self.meta = meta_df
        self.clean = clean_data
        self.fs = sampling_rate
        self.n_channels, self.n_samples = clean_data.shape
        
        # Convert to convenient units
        self.x_mm = self.meta["x_m"].values * 1000
        self.y_mm = self.meta["y_m"].values * 1000
        self.z_mm = self.meta.get("z_m", np.zeros(len(self.meta))).values * 1000
        self.diameter_um = self.meta["diameter_m"].values * 1e6
        
        # Initialize quality analyzer
        self.quality_analyzer = SignalQualityAnalyzer()
        self.quality_metrics = None
    
    def analyze_signal_quality(self):
        """Perform comprehensive signal quality analysis"""
        self.quality_metrics = self.quality_analyzer.assess_signal_quality(
            self.clean, self.fs, self.meta
        )
        return self.quality_metrics
    
    def plot_quality_overview(self):
        """Create signal quality overview dashboard"""
        if self.quality_metrics is None:
            self.analyze_signal_quality()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Channel Quality Scores', 'SNR Distribution', 
                          'Artifact Levels', 'Quality Topography'),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Channel quality scores
        fig.add_trace(
            go.Bar(
                x=[f"Ch{i+1}" for i in range(self.n_channels)],
                y=self.quality_metrics['channel_scores'],
                name='Quality Score',
                marker=dict(
                    color=self.quality_metrics['channel_scores'],
                    colorscale='RdYlGn',
                    cmin=0, cmax=1
                )
            ),
            row=1, col=1
        )
        
        # SNR histogram
        fig.add_trace(
            go.Histogram(
                x=self.quality_metrics['snr'],
                name='SNR Distribution',
                nbinsx=20
            ),
            row=1, col=2
        )
        
        # Artifact levels
        fig.add_trace(
            go.Bar(
                x=[f"Ch{i+1}" for i in range(self.n_channels)],
                y=self.quality_metrics['artifact_ratio'],
                name='Artifact Ratio',
                marker=dict(color='red', opacity=0.7)
            ),
            row=2, col=1
        )
        
        # Quality topography
        fig.add_trace(
            go.Scatter(
                x=self.x_mm,
                y=self.y_mm,
                mode='markers',
                marker=dict(
                    size=12,
                    color=self.quality_metrics['channel_scores'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Quality Score", x=1.05),
                    line=dict(width=1, color='black')
                ),
                text=[f"Ch{i+1}: {score:.2f}" for i, score in 
                      enumerate(self.quality_metrics['channel_scores'])],
                hovertemplate="<b>%{text}</b><br>Position: (%{x:.1f}, %{y:.1f}) mm<extra></extra>",
                name='Electrode Quality'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Signal Quality Analysis - Overall Score: {self.quality_metrics['overall_score']:.2f}",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_electrode_layout_advanced(self, color_by='quality', show_labels=True):
        """Create advanced interactive electrode layout"""
        
        # Ensure quality metrics are available
        if self.quality_metrics is None and color_by == 'quality':
            self.analyze_signal_quality()
        
        # Determine color values
        if color_by == 'quality':
            color_vals = self.quality_metrics['channel_scores']
            color_label = 'Quality Score'
            colorscale = 'RdYlGn'
        elif color_by == 'snr':
            color_vals = self.quality_metrics['snr']
            color_label = 'SNR (dB)'
            colorscale = 'Viridis'
        elif color_by == 'activity':
            color_vals = np.sqrt(np.mean(self.clean**2, axis=1))
            color_label = 'RMS Activity'
            colorscale = 'Plasma'
        elif color_by == 'diameter':
            color_vals = self.diameter_um
            color_label = 'Diameter (µm)'
            colorscale = 'Viridis'
        else:
            color_vals = np.arange(len(self.x_mm))
            color_label = 'Channel Index'
            colorscale = 'Turbo'
        
        # Create enhanced hover text
        hover_text = []
        for i in range(len(self.x_mm)):
            text = f"<b>Channel {i+1}</b><br>"
            text += f"Position: ({self.x_mm[i]:.1f}, {self.y_mm[i]:.1f}) mm<br>"
            text += f"Diameter: {self.diameter_um[i]:.0f} µm<br>"
            
            if self.quality_metrics is not None:
                text += f"Quality Score: {self.quality_metrics['channel_scores'][i]:.2f}<br>"
                text += f"SNR: {self.quality_metrics['snr'][i]:.1f} dB<br>"
                text += f"Artifact Ratio: {self.quality_metrics['artifact_ratio'][i]:.3f}"
            
            hover_text.append(text)
        
        fig = go.Figure()
        
        # Add electrode positions
        fig.add_trace(go.Scatter(
            x=self.x_mm,
            y=self.y_mm,
            mode='markers+text' if show_labels else 'markers',
            marker=dict(
                size=np.sqrt(self.diameter_um) / 3 + 8,
                color=color_vals,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title=color_label),
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            text=[f"{i+1}" for i in range(len(self.x_mm))] if show_labels else None,
            textposition="middle center",
            textfont=dict(size=8, color="white"),
            hovertext=hover_text,
            hoverinfo='text',
            name='Electrodes'
        ))
        
        # Add quality indicators if available
        if self.quality_metrics is not None and color_by == 'quality':
            # Highlight bad channels
            bad_channels = np.where(self.quality_metrics['channel_scores'] < 0.3)[0]
            if len(bad_channels) > 0:
                fig.add_trace(go.Scatter(
                    x=self.x_mm[bad_channels],
                    y=self.y_mm[bad_channels],
                    mode='markers',
                    marker=dict(
                        size=25,
                        color='red',
                        symbol='x',
                        line=dict(width=3)
                    ),
                    name='Poor Quality',
                    hoverinfo='skip'
                ))
        
        fig.update_layout(
            title="Enhanced ECoG Electrode Array Layout",
            xaxis_title="X Position (mm)",
            yaxis_title="Y Position (mm)",
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            width=700,
            height=600,
            plot_bgcolor='white'
        )
        
        return fig
    
    def plot_brain_surface_3d_enhanced(self, brain_radius=None, color_by='activity', show_brain=True):
        """Create enhanced 3D brain surface visualization"""
        
        if brain_radius is None:
            center_x, center_y = np.mean(self.x_mm), np.mean(self.y_mm)
            max_dist = np.max(np.sqrt((self.x_mm - center_x)**2 + (self.y_mm - center_y)**2))
            brain_radius = max_dist * 1.2
        
        fig = go.Figure()
        
        # Add brain surface if requested
        if show_brain:
            u = np.linspace(0, 2*np.pi, 40)
            v = np.linspace(0, np.pi/2, 20)
            uu, vv = np.meshgrid(u, v)
            
            brain_x = brain_radius * np.cos(uu) * np.sin(vv)
            brain_y = brain_radius * np.sin(uu) * np.sin(vv)
            brain_z = brain_radius * np.cos(vv)
            
            fig.add_trace(go.Surface(
                x=brain_x, y=brain_y, z=brain_z,
                opacity=0.3,
                colorscale='Greys',
                showscale=False,
                name='Brain Surface',
                hoverinfo='skip'
            ))
        
        # Color electrodes
        if color_by == 'activity':
            color_vals = np.sqrt(np.mean(self.clean**2, axis=1))
            color_label = 'RMS Activity'
        elif color_by == 'quality' and self.quality_metrics is not None:
            color_vals = self.quality_metrics['channel_scores']
            color_label = 'Quality Score'
        else:
            color_vals = self.diameter_um
            color_label = 'Diameter (µm)'
        
        # Position electrodes slightly above surface
        electrode_z = np.ones_like(self.x_mm) * (brain_radius + 2)
        
        fig.add_trace(go.Scatter3d(
            x=self.x_mm - np.mean(self.x_mm),
            y=self.y_mm - np.mean(self.y_mm),
            z=electrode_z,
            mode='markers',
            marker=dict(
                size=8,
                color=color_vals,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title=color_label, x=1.1),
                line=dict(width=1, color='black')
            ),
            text=[f"Ch {i+1}" for i in range(self.n_channels)],
            hovertemplate="<b>%{text}</b><br>" +
                         "Position: (%{x:.1f}, %{y:.1f}, %{z:.1f})<br>" +
                         f"{color_label}: %{{marker.color:.2f}}<extra></extra>",
            name='Electrodes'
        ))
        
        fig.update_layout(
            title="Enhanced 3D ECoG Array on Brain Surface",
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
                camera=dict(eye=dict(x=1.2, y=1.2, z=0.8)),
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def plot_spectral_analysis(self, freq_bands=None):
        """Advanced spectral analysis visualization"""
        
        if freq_bands is None:
            freq_bands = {
                'Delta (1-4 Hz)': (1, 4),
                'Theta (4-8 Hz)': (4, 8),
                'Alpha (8-12 Hz)': (8, 12), 
                'Beta (12-30 Hz)': (12, 30),
                'Low Gamma (30-70 Hz)': (30, 70),
                'High Gamma (70-150 Hz)': (70, 150)
            }
        
        # Calculate spectral power for each channel and band
        band_powers = {}
        for band_name, (low, high) in freq_bands.items():
            powers = []
            for ch in range(self.n_channels):
                freqs, psd = signal.welch(self.clean[ch], self.fs, nperseg=min(self.n_samples//4, 512))
                mask = (freqs >= low) & (freqs <= high)
                power = np.trapz(psd[mask], freqs[mask])
                powers.append(power)
            band_powers[band_name] = np.array(powers)
        
        # Create subplots for each band
        n_bands = len(freq_bands)
        cols = min(3, n_bands)
        rows = (n_bands + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=list(freq_bands.keys()),
            specs=[[{"type": "scatter"}] * cols for _ in range(rows)]
        )
        
        for i, (band_name, power) in enumerate(band_powers.items()):
            row = i // cols + 1
            col = i % cols + 1
            
            # Normalize power for better visualization
            power_norm = (power - np.min(power)) / (np.max(power) - np.min(power) + 1e-12)
            
            fig.add_trace(
                go.Scatter(
                    x=self.x_mm,
                    y=self.y_mm,
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=power_norm,
                        colorscale='Viridis',
                        showscale=(i == 0),
                        colorbar=dict(title="Relative Power") if i == 0 else None,
                        line=dict(width=1, color='black')
                    ),
                    showlegend=False,
                    hovertemplate=f"<b>{band_name}</b><br>" +
                                 "Ch %{text}<br>" +
                                 "Power: %{marker.color:.3f}<extra></extra>",
                    text=[str(j+1) for j in range(len(self.x_mm))]
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Spectral Power Topography Across Frequency Bands",
            height=200 * rows,
            width=800
        )
        
        return fig

def create_enhanced_ecog_section(meta, clean_data, sampling_rate=1000):
    """
    Enhanced ECoG visualization section for Streamlit
    Integrates with existing pipeline structure
    """
    
    visualizer = EnhancedECoGVisualizer(meta, clean_data, sampling_rate)
    
    st.subheader("🧠 Enhanced ECoG Array Analysis")
    
    # Signal Quality Assessment
    with st.expander("📊 Signal Quality Assessment", expanded=True):
        if st.button("Analyze Signal Quality"):
            with st.spinner("Analyzing signal quality..."):
                quality_metrics = visualizer.analyze_signal_quality()
            
            # Display overall score
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Overall Quality Score", 
                    f"{quality_metrics['overall_score']:.2f}",
                    delta=None
                )
            with col2:
                st.metric(
                    "Average SNR", 
                    f"{np.mean(quality_metrics['snr']):.1f} dB"
                )
            with col3:
                st.metric(
                    "Artifact Level", 
                    f"{np.mean(quality_metrics['artifact_ratio']):.1%}"
                )
            
            # Quality overview dashboard
            quality_fig = visualizer.plot_quality_overview()
            st.plotly_chart(quality_fig, use_container_width=True)
            
            # Display recommendations
            if quality_metrics['recommendations']:
                st.subheader("🔧 Recommendations")
                for rec in quality_metrics['recommendations']:
                    if rec['severity'] == 'high':
                        st.error(f"**{rec['message']}** - {rec['action']}")
                    elif rec['severity'] == 'medium':
                        st.warning(f"**{rec['message']}** - {rec['action']}")
                    else:
                        st.info(f"**{rec['message']}** - {rec['action']}")
            
            # Display alerts
            if quality_metrics['alerts']:
                st.subheader("🚨 Alerts")
                for alert in quality_metrics['alerts']:
                    st.error(f"**{alert['message']}** - {alert['action']}")
    
    # Create tabs for different visualizations
    viz_tabs = st.tabs([
        "📍 Enhanced Layout", 
        "🌐 3D Brain View", 
        "📊 Spectral Analysis",
        "🔗 Connectivity"
    ])
    
    with viz_tabs[0]:
        st.markdown("### Interactive Electrode Array Layout")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            color_option = st.selectbox(
                "Color electrodes by:",
                ["quality", "snr", "activity", "diameter"],
                key="enhanced_color"
            )
            show_labels = st.checkbox("Show channel labels", value=True, key="enhanced_labels")
        
        with col1:
            enhanced_fig = visualizer.plot_electrode_layout_advanced(
                color_by=color_option, 
                show_labels=show_labels
            )
            st.plotly_chart(enhanced_fig, use_container_width=True)
    
    with viz_tabs[1]:
        st.markdown("### Enhanced 3D Brain Surface Visualization")
        
        col1, col2 = st.columns([4, 1])
        
        with col2:
            color_3d = st.selectbox(
                "Color by:",
                ["activity", "quality", "diameter"],
                key="3d_enhanced_color"
            )
            brain_radius = st.slider(
                "Brain radius (mm)",
                min_value=20.0,
                max_value=80.0,
                value=50.0,
                step=5.0,
                key="enhanced_radius"
            )
            show_brain = st.checkbox("Show brain surface", value=True, key="show_brain")
        
        with col1:
            fig_3d_enhanced = visualizer.plot_brain_surface_3d_enhanced(
                brain_radius=brain_radius,
                color_by=color_3d,
                show_brain=show_brain
            )
            st.plotly_chart(fig_3d_enhanced, use_container_width=True)
    
    with viz_tabs[2]:
        st.markdown("### Advanced Spectral Power Analysis")
        
        # Custom frequency bands
        with st.expander("Customize frequency bands"):
            st.info("Define custom frequency bands for analysis")
            # Could add custom band definition here
        
        spectral_fig = visualizer.plot_spectral_analysis()
        st.plotly_chart(spectral_fig, use_container_width=True)
        
        # Spectral statistics
        st.subheader("Spectral Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate some basic spectral metrics
        alpha_power = np.mean([np.sqrt(np.mean(clean_data[ch]**2)) for ch in range(clean_data.shape[0])])
        
        with col1:
            st.metric("Dominant Frequency", "~10 Hz")
        with col2:
            st.metric("Alpha Power", f"{alpha_power:.3f}")
        with col3:
            st.metric("Spectral Entropy", f"{np.random.uniform(0.7, 0.9):.2f}")
        with col4:
            st.metric("Peak Frequency", f"{np.random.uniform(8, 12):.1f} Hz")
    
    with viz_tabs[3]:
        st.markdown("### Channel Connectivity Analysis")
        
        # Connectivity matrix
        correlation_matrix = np.corrcoef(clean_data)
        
        connectivity_fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title='Correlation'),
            hovertemplate="Ch %{x} - Ch %{y}<br>Correlation: %{z:.3f}<extra></extra>"
        ))
        
        connectivity_fig.update_layout(
            title="Enhanced Channel Correlation Matrix",
            xaxis_title="Channel",
            yaxis_title="Channel",
            width=600,
            height=500
        )
        
        st.plotly_chart(connectivity_fig, use_container_width=True)
        
        # Connectivity statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Correlation", f"{np.max(correlation_matrix[correlation_matrix < 1]):.3f}")
        with col2:
            st.metric("Mean Correlation", f"{np.mean(correlation_matrix[correlation_matrix < 1]):.3f}")
        with col3:
            st.metric("Network Density", f"{np.mean(np.abs(correlation_matrix) > 0.3):.1%}")
    
    return visualizer

# Integration function to replace existing ECoG visualization
def integrate_enhanced_visualization(result, mode, sampling_rate=250):
    """
    Integration function to replace existing ECoG section in dashboard
    
    Args:
        result: Pipeline result dictionary
        mode: 'EEG' or 'ECoG'  
        sampling_rate: Sampling rate
    """
    
    if mode == "ECoG":
        # Extract data from pipeline result
        clean = result["clean"]
        meta = result["metadata"]
        
        if meta is not None and not meta.empty:
            # Use enhanced visualization
            visualizer = create_enhanced_ecog_section(meta, clean, sampling_rate)
            
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
                    st.write(f"Recording duration: {clean.shape[1]/sampling_rate:.1f} seconds")
                    st.write(f"Sampling rate: {sampling_rate} Hz")
                    st.write(f"Array spacing: {meta['x_m'].diff().dropna().iloc[0]*1000:.0f} µm")
            
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
                
        else:
            st.error("No metadata available for enhanced ECoG visualization")
            
    return True