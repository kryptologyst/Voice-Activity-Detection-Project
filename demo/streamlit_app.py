"""Streamlit demo for Voice Activity Detection."""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import streamlit as st
import torch
import librosa
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.vad.models import EnergyVAD, CNNVAD, TransformerVAD
from src.vad.utils import get_device, set_seed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Voice Activity Detection Demo",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Privacy disclaimer
PRIVACY_DISCLAIMER = """
**PRIVACY & ETHICS DISCLAIMER**

This is a research and educational demonstration of Voice Activity Detection (VAD) technology. 

**Important Notes:**
- This system is designed for research and educational purposes only
- No personal data is stored or transmitted
- Audio files are processed locally and not saved
- This technology should NOT be used for biometric identification in production
- Voice cloning or impersonation using this technology is strictly prohibited
- Users are responsible for complying with applicable privacy laws and regulations

**Intended Use:**
- Research and development of speech processing systems
- Educational demonstrations of audio analysis techniques
- Non-biometric applications such as noise reduction and audio segmentation

**Prohibited Uses:**
- Biometric identification or authentication
- Voice cloning or deepfake generation
- Surveillance or monitoring without consent
- Any use that violates privacy rights or applicable laws
"""


@st.cache_resource
def load_models():
    """Load VAD models."""
    device = get_device("cpu")  # Use CPU for demo
    
    models = {}
    
    # Energy-based VAD
    models["Energy-based"] = EnergyVAD(
        frame_size=1024,
        hop_size=512,
        threshold=0.02,
        energy_normalization=True,
        sample_rate=16000,
    )
    
    # CNN-based VAD
    models["CNN-based"] = CNNVAD(
        input_dim=80,
        hidden_dims=[128, 64, 32],
        kernel_sizes=[3, 3, 3],
        dropout=0.2,
        sample_rate=16000,
    )
    
    # Transformer-based VAD
    models["Transformer-based"] = TransformerVAD(
        input_dim=80,
        d_model=256,
        nhead=8,
        num_layers=6,
        dropout=0.1,
        sample_rate=16000,
    )
    
    return models


def load_audio_file(uploaded_file) -> Tuple[np.ndarray, int]:
    """Load audio from uploaded file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        audio, sr = librosa.load(tmp_file_path, sr=16000, mono=True)
        return audio, sr
    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        return None, None
    finally:
        Path(tmp_file_path).unlink()


def generate_synthetic_audio(duration: float, sample_rate: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic audio with known VAD labels."""
    length = int(duration * sample_rate)
    audio = np.zeros(length)
    vad_labels = np.zeros(length)
    
    # Generate speech segments
    current_pos = 0
    while current_pos < length:
        # Random segment length (0.5 to 3 seconds)
        segment_length = np.random.uniform(0.5, 3.0)
        segment_samples = int(segment_length * sample_rate)
        
        # Check if we should add speech or silence
        if np.random.random() < 0.6:  # 60% chance of speech
            # Generate speech-like signal
            freq = np.random.uniform(100, 400)
            t = np.linspace(0, segment_length, segment_samples)
            
            # Generate harmonic speech-like signal
            speech = np.sin(2 * np.pi * freq * t)
            speech += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
            speech += 0.1 * np.sin(2 * np.pi * freq * 3 * t)
            
            # Add amplitude modulation
            modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)
            speech *= modulation
            
            # Add to audio
            end_pos = min(current_pos + segment_samples, length)
            actual_length = end_pos - current_pos
            audio[current_pos:end_pos] = speech[:actual_length]
            vad_labels[current_pos:end_pos] = 1.0
        else:
            # Generate silence with background noise
            noise = np.random.normal(0, 0.01, segment_samples)
            end_pos = min(current_pos + segment_samples, length)
            actual_length = end_pos - current_pos
            audio[current_pos:end_pos] = noise[:actual_length]
            vad_labels[current_pos:end_pos] = 0.0
        
        current_pos += segment_samples
    
    # Normalize audio
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    return audio, vad_labels


def plot_audio_and_vad(audio: np.ndarray, vad_predictions: np.ndarray, sample_rate: int, title: str = "VAD Results"):
    """Plot audio waveform and VAD predictions."""
    time = np.arange(len(audio)) / sample_rate
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Audio Waveform", "Voice Activity Detection"),
        vertical_spacing=0.1
    )
    
    # Plot audio waveform
    fig.add_trace(
        go.Scatter(
            x=time,
            y=audio,
            mode='lines',
            name='Audio',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    # Plot VAD predictions
    fig.add_trace(
        go.Scatter(
            x=time,
            y=vad_predictions,
            mode='lines',
            name='VAD Predictions',
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=600,
        showlegend=True
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="VAD (0=Silence, 1=Speech)", row=2, col=1)
    
    return fig


def compute_vad_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """Compute VAD evaluation metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Ensure binary predictions
    pred_binary = (predictions > 0.5).astype(int)
    target_binary = targets.astype(int)
    
    metrics = {
        "Accuracy": accuracy_score(target_binary, pred_binary),
        "Precision": precision_score(target_binary, pred_binary, zero_division=0),
        "Recall": recall_score(target_binary, pred_binary, zero_division=0),
        "F1 Score": f1_score(target_binary, pred_binary, zero_division=0),
    }
    
    return metrics


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">🎤 Voice Activity Detection Demo</h1>', unsafe_allow_html=True)
    
    # Privacy disclaimer
    with st.expander("⚠️ Privacy & Ethics Disclaimer", expanded=False):
        st.markdown(PRIVACY_DISCLAIMER)
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select VAD Model",
        ["Energy-based", "CNN-based", "Transformer-based"],
        help="Choose the Voice Activity Detection model to use"
    )
    
    # Audio input method
    input_method = st.sidebar.radio(
        "Audio Input Method",
        ["Upload Audio File", "Generate Synthetic Audio"],
        help="Choose how to provide audio input"
    )
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
    
    model = models[model_name]
    
    # Audio input
    audio = None
    vad_labels = None
    sample_rate = 16000
    
    if input_method == "Upload Audio File":
        st.header("📁 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Upload an audio file to analyze for voice activity"
        )
        
        if uploaded_file is not None:
            with st.spinner("Loading audio file..."):
                audio, sample_rate = load_audio_file(uploaded_file)
                if audio is not None:
                    st.success(f"Audio loaded successfully! Duration: {len(audio)/sample_rate:.2f} seconds")
    
    else:  # Generate Synthetic Audio
        st.header("🎵 Generate Synthetic Audio")
        
        duration = st.slider(
            "Audio Duration (seconds)",
            min_value=1.0,
            max_value=30.0,
            value=10.0,
            step=1.0,
            help="Duration of synthetic audio to generate"
        )
        
        if st.button("Generate Synthetic Audio"):
            with st.spinner("Generating synthetic audio..."):
                audio, vad_labels = generate_synthetic_audio(duration, sample_rate)
                st.success(f"Synthetic audio generated! Duration: {duration} seconds")
    
    # VAD Analysis
    if audio is not None:
        st.header("🔍 Voice Activity Detection Analysis")
        
        # Run VAD
        with st.spinner("Running Voice Activity Detection..."):
            vad_predictions = model.predict(audio)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Audio Visualization")
            fig = plot_audio_and_vad(audio, vad_predictions, sample_rate, f"{model_name} VAD Results")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 VAD Statistics")
            
            # Basic statistics
            speech_frames = np.sum(vad_predictions > 0.5)
            total_frames = len(vad_predictions)
            speech_percentage = (speech_frames / total_frames) * 100
            
            st.metric("Speech Frames", f"{speech_frames:,}")
            st.metric("Total Frames", f"{total_frames:,}")
            st.metric("Speech Percentage", f"{speech_percentage:.1f}%")
            
            # Audio duration
            duration = len(audio) / sample_rate
            st.metric("Audio Duration", f"{duration:.2f} seconds")
            
            # Frame rate
            frame_rate = len(vad_predictions) / duration
            st.metric("VAD Frame Rate", f"{frame_rate:.1f} Hz")
        
        # Evaluation metrics (if ground truth available)
        if vad_labels is not None:
            st.subheader("📊 Evaluation Metrics")
            
            metrics = compute_vad_metrics(vad_predictions, vad_labels)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{metrics['Precision']:.3f}")
            with col3:
                st.metric("Recall", f"{metrics['Recall']:.3f}")
            with col4:
                st.metric("F1 Score", f"{metrics['F1 Score']:.3f}")
        
        # Download results
        st.subheader("💾 Download Results")
        
        # Create results data
        results_data = {
            "audio_duration": len(audio) / sample_rate,
            "sample_rate": sample_rate,
            "total_frames": len(vad_predictions),
            "speech_frames": int(np.sum(vad_predictions > 0.5)),
            "speech_percentage": float((np.sum(vad_predictions > 0.5) / len(vad_predictions)) * 100),
            "vad_predictions": vad_predictions.tolist(),
        }
        
        if vad_labels is not None:
            results_data["ground_truth"] = vad_labels.tolist()
            results_data["metrics"] = metrics
        
        # Download button
        import json
        results_json = json.dumps(results_data, indent=2)
        
        st.download_button(
            label="Download VAD Results (JSON)",
            data=results_json,
            file_name="vad_results.json",
            mime="application/json"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Voice Activity Detection Demo** | "
        "Research & Educational Use Only | "
        "Privacy-Preserving Technology"
    )


if __name__ == "__main__":
    main()
