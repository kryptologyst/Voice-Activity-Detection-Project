"""Voice Activity Detection models."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.audio_utils import (
    compute_energy,
    extract_mel_spectrogram,
    extract_mfcc,
    extract_spectral_features,
)

logger = logging.getLogger(__name__)


class BaseVAD(ABC, nn.Module):
    """Base class for Voice Activity Detection models."""
    
    def __init__(self, sample_rate: int = 16000):
        """Initialize base VAD model.
        
        Args:
            sample_rate: Sample rate of input audio.
        """
        super().__init__()
        self.sample_rate = sample_rate
    
    @abstractmethod
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            audio: Input audio tensor.
            
        Returns:
            VAD predictions tensor.
        """
        pass
    
    @abstractmethod
    def predict(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict VAD for input audio.
        
        Args:
            audio: Input audio array or tensor.
            
        Returns:
            VAD predictions as numpy array.
        """
        pass


class EnergyVAD(BaseVAD):
    """Energy-based Voice Activity Detection model."""
    
    def __init__(
        self,
        frame_size: int = 1024,
        hop_size: int = 512,
        threshold: float = 0.02,
        energy_normalization: bool = True,
        sample_rate: int = 16000,
    ):
        """Initialize energy-based VAD.
        
        Args:
            frame_size: Frame size in samples.
            hop_size: Hop size in samples.
            threshold: Energy threshold for speech detection.
            energy_normalization: Whether to normalize energy.
            sample_rate: Sample rate of input audio.
        """
        super().__init__(sample_rate)
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.threshold = threshold
        self.energy_normalization = energy_normalization
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass for energy-based VAD.
        
        Args:
            audio: Input audio tensor.
            
        Returns:
            VAD predictions tensor.
        """
        # Compute energy
        energy = self._compute_energy_torch(audio)
        
        # Normalize energy if requested
        if self.energy_normalization:
            energy = energy / (torch.max(energy) + 1e-8)
        
        # Apply threshold
        vad = (energy > self.threshold).float()
        
        return vad
    
    def _compute_energy_torch(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute frame-wise energy using PyTorch.
        
        Args:
            audio: Input audio tensor.
            
        Returns:
            Energy tensor.
        """
        # Pad audio if necessary
        if len(audio) < self.frame_size:
            audio = F.pad(audio, (0, self.frame_size - len(audio)))
        
        # Compute energy for each frame
        energy = []
        for i in range(0, len(audio) - self.frame_size + 1, self.hop_size):
            frame = audio[i:i + self.frame_size]
            frame_energy = torch.sum(frame ** 2)
            energy.append(frame_energy)
        
        return torch.stack(energy)
    
    def predict(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict VAD for input audio.
        
        Args:
            audio: Input audio array or tensor.
            
        Returns:
            VAD predictions as numpy array.
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        self.eval()
        with torch.no_grad():
            vad = self.forward(audio)
        
        return vad.numpy()


class SpectralVAD(BaseVAD):
    """Spectral feature-based Voice Activity Detection model."""
    
    def __init__(
        self,
        frame_size: int = 1024,
        hop_size: int = 512,
        n_mels: int = 80,
        n_mfcc: int = 13,
        threshold: float = 0.5,
        sample_rate: int = 16000,
    ):
        """Initialize spectral feature-based VAD.
        
        Args:
            frame_size: Frame size in samples.
            hop_size: Hop size in samples.
            n_mels: Number of mel filter banks.
            n_mfcc: Number of MFCC coefficients.
            threshold: Classification threshold.
            sample_rate: Sample rate of input audio.
        """
        super().__init__(sample_rate)
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.threshold = threshold
        
        # Feature dimension
        self.feature_dim = n_mels + n_mfcc + 4  # mel + mfcc + spectral features
        
        # Simple MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass for spectral feature-based VAD.
        
        Args:
            audio: Input audio tensor.
            
        Returns:
            VAD predictions tensor.
        """
        # Extract features
        features = self._extract_features_torch(audio)
        
        # Classify
        vad_probs = self.classifier(features)
        vad = (vad_probs > self.threshold).float()
        
        return vad.squeeze(-1)
    
    def _extract_features_torch(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract spectral features using PyTorch.
        
        Args:
            audio: Input audio tensor.
            
        Returns:
            Feature tensor.
        """
        # Convert to numpy for librosa processing
        audio_np = audio.numpy()
        
        # Extract mel-spectrogram
        mel_spec = extract_mel_spectrogram(
            audio_np,
            sample_rate=self.sample_rate,
            n_fft=self.frame_size,
            hop_length=self.hop_size,
            n_mels=self.n_mels,
        )
        
        # Extract MFCC
        mfcc = extract_mfcc(
            audio_np,
            sample_rate=self.sample_rate,
            n_fft=self.frame_size,
            hop_length=self.hop_size,
            n_mels=self.n_mels,
        )
        
        # Extract spectral features
        spectral_features = extract_spectral_features(
            audio_np,
            sample_rate=self.sample_rate,
            n_fft=self.frame_size,
            hop_length=self.hop_size,
        )
        
        # Combine features
        features = np.vstack([
            mel_spec.mean(axis=0),  # Average across frequency bins
            mfcc.mean(axis=0),      # Average across frequency bins
            spectral_features.mean(axis=0),  # Average across frequency bins
        ])
        
        return torch.from_numpy(features).float()
    
    def predict(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict VAD for input audio.
        
        Args:
            audio: Input audio array or tensor.
            
        Returns:
            VAD predictions as numpy array.
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        self.eval()
        with torch.no_grad():
            vad = self.forward(audio)
        
        return vad.numpy()


class CNNVAD(BaseVAD):
    """CNN-based Voice Activity Detection model."""
    
    def __init__(
        self,
        input_dim: int = 80,
        hidden_dims: list = [128, 64, 32],
        kernel_sizes: list = [3, 3, 3],
        dropout: float = 0.2,
        sample_rate: int = 16000,
    ):
        """Initialize CNN-based VAD.
        
        Args:
            input_dim: Input feature dimension (mel-spectrogram bins).
            hidden_dims: Hidden layer dimensions.
            kernel_sizes: Kernel sizes for each layer.
            dropout: Dropout rate.
            sample_rate: Sample rate of input audio.
        """
        super().__init__(sample_rate)
        self.input_dim = input_dim
        
        # Build CNN layers
        layers = []
        in_channels = 1  # Single channel input
        out_channels = input_dim
        
        for i, (hidden_dim, kernel_size) in enumerate(zip(hidden_dims, kernel_sizes)):
            layers.extend([
                nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_channels = hidden_dim
        
        # Global average pooling and classification
        layers.extend([
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid(),
        ])
        
        self.cnn = nn.Sequential(*layers)
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass for CNN-based VAD.
        
        Args:
            audio: Input audio tensor.
            
        Returns:
            VAD predictions tensor.
        """
        # Extract mel-spectrogram features
        mel_spec = self._extract_mel_spectrogram_torch(audio)
        
        # Add channel dimension
        mel_spec = mel_spec.unsqueeze(0)  # [1, n_mels, time]
        
        # Forward through CNN
        vad_probs = self.cnn(mel_spec)
        
        return vad_probs.squeeze()
    
    def _extract_mel_spectrogram_torch(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract mel-spectrogram using PyTorch.
        
        Args:
            audio: Input audio tensor.
            
        Returns:
            Mel-spectrogram tensor.
        """
        # Convert to numpy for librosa processing
        audio_np = audio.numpy()
        
        # Extract mel-spectrogram
        mel_spec = extract_mel_spectrogram(
            audio_np,
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=self.input_dim,
        )
        
        return torch.from_numpy(mel_spec).float()
    
    def predict(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict VAD for input audio.
        
        Args:
            audio: Input audio array or tensor.
            
        Returns:
            VAD predictions as numpy array.
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        self.eval()
        with torch.no_grad():
            vad = self.forward(audio)
        
        return vad.numpy()


class TransformerVAD(BaseVAD):
    """Transformer-based Voice Activity Detection model."""
    
    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        sample_rate: int = 16000,
    ):
        """Initialize Transformer-based VAD.
        
        Args:
            input_dim: Input feature dimension.
            d_model: Model dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer layers.
            dropout: Dropout rate.
            sample_rate: Sample rate of input audio.
        """
        super().__init__(sample_rate)
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass for Transformer-based VAD.
        
        Args:
            audio: Input audio tensor.
            
        Returns:
            VAD predictions tensor.
        """
        # Extract mel-spectrogram features
        mel_spec = self._extract_mel_spectrogram_torch(audio)
        
        # Project to model dimension
        x = self.input_projection(mel_spec.T)  # [time, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)  # [time, d_model]
        
        # Classification
        vad_probs = self.classifier(x)  # [time, 1]
        
        return vad_probs.squeeze(-1)
    
    def _extract_mel_spectrogram_torch(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract mel-spectrogram using PyTorch.
        
        Args:
            audio: Input audio tensor.
            
        Returns:
            Mel-spectrogram tensor.
        """
        # Convert to numpy for librosa processing
        audio_np = audio.numpy()
        
        # Extract mel-spectrogram
        mel_spec = extract_mel_spectrogram(
            audio_np,
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=self.input_dim,
        )
        
        return torch.from_numpy(mel_spec).float()
    
    def predict(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict VAD for input audio.
        
        Args:
            audio: Input audio array or tensor.
            
        Returns:
            VAD predictions as numpy array.
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        self.eval()
        with torch.no_grad():
            vad = self.forward(audio)
        
        return vad.numpy()


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension.
            dropout: Dropout rate.
            max_len: Maximum sequence length.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding.
        
        Args:
            x: Input tensor.
            
        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
