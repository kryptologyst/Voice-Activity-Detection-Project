"""Audio processing utilities for VAD."""

import logging
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torchaudio
from scipy import signal

logger = logging.getLogger(__name__)


def load_audio(
    file_path: str,
    sample_rate: int = 16000,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load audio file with librosa.
    
    Args:
        file_path: Path to audio file.
        sample_rate: Target sample rate.
        mono: Whether to convert to mono.
        normalize: Whether to normalize audio.
        
    Returns:
        Tuple of (audio_array, sample_rate).
    """
    try:
        audio, sr = librosa.load(
            file_path,
            sr=sample_rate,
            mono=mono,
            dtype=np.float32,
        )
        
        if normalize:
            audio = librosa.util.normalize(audio)
            
        return audio, sr
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio to target sample rate.
    
    Args:
        audio: Input audio array.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.
        
    Returns:
        Resampled audio array.
    """
    if orig_sr == target_sr:
        return audio
        
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def extract_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    win_length: int = 1024,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Extract mel-spectrogram features.
    
    Args:
        audio: Input audio array.
        sample_rate: Sample rate of audio.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        win_length: Window length for STFT.
        n_mels: Number of mel filter banks.
        fmin: Minimum frequency.
        fmax: Maximum frequency.
        
    Returns:
        Mel-spectrogram features.
    """
    if fmax is None:
        fmax = sample_rate // 2
        
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec


def extract_mfcc(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 80,
) -> np.ndarray:
    """Extract MFCC features.
    
    Args:
        audio: Input audio array.
        sample_rate: Sample rate of audio.
        n_mfcc: Number of MFCC coefficients.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        n_mels: Number of mel filter banks.
        
    Returns:
        MFCC features.
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    
    return mfcc


def extract_spectral_features(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    """Extract spectral features (spectral centroid, rolloff, bandwidth).
    
    Args:
        audio: Input audio array.
        sample_rate: Sample rate of audio.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        
    Returns:
        Spectral features array.
    """
    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )
    
    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )
    
    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        audio, frame_length=n_fft, hop_length=hop_length
    )
    
    # Combine features
    features = np.vstack([
        spectral_centroid,
        spectral_rolloff,
        spectral_bandwidth,
        zcr,
    ])
    
    return features


def compute_energy(
    audio: np.ndarray,
    frame_size: int = 1024,
    hop_size: int = 512,
) -> np.ndarray:
    """Compute frame-wise energy.
    
    Args:
        audio: Input audio array.
        frame_size: Frame size in samples.
        hop_size: Hop size in samples.
        
    Returns:
        Energy values for each frame.
    """
    energy = []
    
    for i in range(0, len(audio) - frame_size + 1, hop_size):
        frame = audio[i:i + frame_size]
        frame_energy = np.sum(frame ** 2)
        energy.append(frame_energy)
    
    return np.array(energy)


def apply_preemphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply pre-emphasis filter.
    
    Args:
        audio: Input audio array.
        coeff: Pre-emphasis coefficient.
        
    Returns:
        Pre-emphasized audio.
    """
    return signal.lfilter([1, -coeff], [1], audio)


def add_noise(
    audio: np.ndarray,
    noise_level: float = 0.1,
    noise_type: str = "gaussian",
) -> np.ndarray:
    """Add noise to audio signal.
    
    Args:
        audio: Input audio array.
        noise_level: Noise level (0-1).
        noise_type: Type of noise ("gaussian", "uniform").
        
    Returns:
        Audio with added noise.
    """
    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_level, len(audio))
    elif noise_type == "uniform":
        noise = np.random.uniform(-noise_level, noise_level, len(audio))
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return audio + noise


def apply_speed_perturbation(
    audio: np.ndarray,
    sample_rate: int,
    speed_factor: float = 1.0,
) -> np.ndarray:
    """Apply speed perturbation to audio.
    
    Args:
        audio: Input audio array.
        sample_rate: Sample rate of audio.
        speed_factor: Speed factor (1.0 = original speed).
        
    Returns:
        Speed-perturbed audio.
    """
    if speed_factor == 1.0:
        return audio
        
    return librosa.effects.time_stretch(audio, rate=speed_factor)


def apply_pitch_shift(
    audio: np.ndarray,
    sample_rate: int,
    n_steps: float = 0.0,
) -> np.ndarray:
    """Apply pitch shift to audio.
    
    Args:
        audio: Input audio array.
        sample_rate: Sample rate of audio.
        n_steps: Number of semitones to shift.
        
    Returns:
        Pitch-shifted audio.
    """
    if n_steps == 0.0:
        return audio
        
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
