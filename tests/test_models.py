"""Unit tests for Voice Activity Detection models."""

import pytest
import numpy as np
import torch

from src.vad.models import EnergyVAD, CNNVAD, TransformerVAD
from src.vad.data import SyntheticVADDataset
from src.vad.metrics import VADMetrics
from src.vad.utils import get_device, set_seed


class TestEnergyVAD:
    """Test cases for Energy-based VAD."""
    
    def test_initialization(self):
        """Test EnergyVAD initialization."""
        model = EnergyVAD(
            frame_size=1024,
            hop_size=512,
            threshold=0.02,
            energy_normalization=True,
            sample_rate=16000,
        )
        
        assert model.frame_size == 1024
        assert model.hop_size == 512
        assert model.threshold == 0.02
        assert model.energy_normalization is True
        assert model.sample_rate == 16000
    
    def test_forward_pass(self):
        """Test forward pass through EnergyVAD."""
        model = EnergyVAD()
        
        # Create test audio
        audio = torch.randn(16000)  # 1 second at 16kHz
        
        # Forward pass
        output = model(audio)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] > 0
        assert torch.all((output >= 0) & (output <= 1))
    
    def test_predict(self):
        """Test prediction method."""
        model = EnergyVAD()
        
        # Create test audio
        audio = np.random.randn(16000)  # 1 second at 16kHz
        
        # Predict
        predictions = model.predict(audio)
        
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] > 0
        assert np.all((predictions >= 0) & (predictions <= 1))


class TestCNNVAD:
    """Test cases for CNN-based VAD."""
    
    def test_initialization(self):
        """Test CNNVAD initialization."""
        model = CNNVAD(
            input_dim=80,
            hidden_dims=[128, 64, 32],
            kernel_sizes=[3, 3, 3],
            dropout=0.2,
            sample_rate=16000,
        )
        
        assert model.input_dim == 80
        assert model.hidden_dims == [128, 64, 32]
        assert model.kernel_sizes == [3, 3, 3]
        assert model.dropout == 0.2
        assert model.sample_rate == 16000
    
    def test_forward_pass(self):
        """Test forward pass through CNNVAD."""
        model = CNNVAD()
        
        # Create test audio
        audio = torch.randn(16000)  # 1 second at 16kHz
        
        # Forward pass
        output = model(audio)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] > 0
        assert torch.all((output >= 0) & (output <= 1))
    
    def test_predict(self):
        """Test prediction method."""
        model = CNNVAD()
        
        # Create test audio
        audio = np.random.randn(16000)  # 1 second at 16kHz
        
        # Predict
        predictions = model.predict(audio)
        
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] > 0
        assert np.all((predictions >= 0) & (predictions <= 1))


class TestTransformerVAD:
    """Test cases for Transformer-based VAD."""
    
    def test_initialization(self):
        """Test TransformerVAD initialization."""
        model = TransformerVAD(
            input_dim=80,
            d_model=256,
            nhead=8,
            num_layers=6,
            dropout=0.1,
            sample_rate=16000,
        )
        
        assert model.input_dim == 80
        assert model.d_model == 256
        assert model.nhead == 8
        assert model.num_layers == 6
        assert model.dropout == 0.1
        assert model.sample_rate == 16000
    
    def test_forward_pass(self):
        """Test forward pass through TransformerVAD."""
        model = TransformerVAD()
        
        # Create test audio
        audio = torch.randn(16000)  # 1 second at 16kHz
        
        # Forward pass
        output = model(audio)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] > 0
        assert torch.all((output >= 0) & (output <= 1))
    
    def test_predict(self):
        """Test prediction method."""
        model = TransformerVAD()
        
        # Create test audio
        audio = np.random.randn(16000)  # 1 second at 16kHz
        
        # Predict
        predictions = model.predict(audio)
        
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] > 0
        assert np.all((predictions >= 0) & (predictions <= 1))


class TestSyntheticDataset:
    """Test cases for Synthetic VAD Dataset."""
    
    def test_initialization(self):
        """Test SyntheticVADDataset initialization."""
        dataset = SyntheticVADDataset(
            num_samples=100,
            duration=5.0,
            sample_rate=16000,
            speech_probability=0.6,
            noise_level=0.1,
            augment=False,
        )
        
        assert len(dataset) == 100
        assert dataset.duration == 5.0
        assert dataset.sample_rate == 16000
        assert dataset.speech_probability == 0.6
        assert dataset.noise_level == 0.1
        assert dataset.augment is False
    
    def test_getitem(self):
        """Test dataset item retrieval."""
        dataset = SyntheticVADDataset(num_samples=10, duration=2.0)
        
        # Get an item
        audio, labels = dataset[0]
        
        assert isinstance(audio, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert audio.shape[0] > 0
        assert labels.shape[0] > 0
        assert audio.shape[0] == labels.shape[0]
        assert torch.all((labels >= 0) & (labels <= 1))
    
    def test_augmentation(self):
        """Test data augmentation."""
        dataset = SyntheticVADDataset(
            num_samples=10,
            duration=2.0,
            augment=True,
        )
        
        # Get an item
        audio, labels = dataset[0]
        
        assert isinstance(audio, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert audio.shape[0] > 0
        assert labels.shape[0] > 0


class TestVADMetrics:
    """Test cases for VAD Metrics."""
    
    def test_initialization(self):
        """Test VADMetrics initialization."""
        metrics = VADMetrics(
            collar_tolerance=0.1,
            min_speech_duration=0.1,
            min_silence_duration=0.1,
            sample_rate=16000,
        )
        
        assert metrics.collar_tolerance == 0.1
        assert metrics.min_speech_duration == 0.1
        assert metrics.min_silence_duration == 0.1
        assert metrics.sample_rate == 16000
    
    def test_compute_frame_metrics(self):
        """Test frame-level metrics computation."""
        metrics = VADMetrics()
        
        # Create test data
        predictions = np.array([1, 1, 0, 0, 1, 0, 1, 1])
        targets = np.array([1, 0, 0, 1, 1, 0, 1, 0])
        
        # Compute metrics
        frame_metrics = metrics.compute_frame_metrics(predictions, targets)
        
        assert "frame_accuracy" in frame_metrics
        assert "frame_precision" in frame_metrics
        assert "frame_recall" in frame_metrics
        assert "frame_f1" in frame_metrics
        assert "frame_auc" in frame_metrics
        
        # Check metric ranges
        assert 0 <= frame_metrics["frame_accuracy"] <= 1
        assert 0 <= frame_metrics["frame_precision"] <= 1
        assert 0 <= frame_metrics["frame_recall"] <= 1
        assert 0 <= frame_metrics["frame_f1"] <= 1
        assert 0 <= frame_metrics["frame_auc"] <= 1
    
    def test_compute_segment_metrics(self):
        """Test segment-level metrics computation."""
        metrics = VADMetrics()
        
        # Create test data
        predictions = np.array([1, 1, 0, 0, 1, 0, 1, 1])
        targets = np.array([1, 0, 0, 1, 1, 0, 1, 0])
        
        # Compute metrics
        segment_metrics = metrics.compute_segment_metrics(predictions, targets)
        
        assert "segment_f1" in segment_metrics
        assert 0 <= segment_metrics["segment_f1"] <= 1
    
    def test_compute_boundary_metrics(self):
        """Test boundary detection metrics computation."""
        metrics = VADMetrics()
        
        # Create test data
        predictions = np.array([1, 1, 0, 0, 1, 0, 1, 1])
        targets = np.array([1, 0, 0, 1, 1, 0, 1, 0])
        
        # Compute metrics
        boundary_metrics = metrics.compute_boundary_metrics(predictions, targets)
        
        assert "boundary_f1" in boundary_metrics
        assert 0 <= boundary_metrics["boundary_f1"] <= 1
    
    def test_compute_all_metrics(self):
        """Test computation of all metrics."""
        metrics = VADMetrics()
        
        # Create test data
        predictions = np.array([1, 1, 0, 0, 1, 0, 1, 1])
        targets = np.array([1, 0, 0, 1, 1, 0, 1, 0])
        
        # Compute all metrics
        all_metrics = metrics.compute_all_metrics(predictions, targets)
        
        # Check that all metric types are present
        assert "frame_accuracy" in all_metrics
        assert "frame_precision" in all_metrics
        assert "frame_recall" in all_metrics
        assert "frame_f1" in all_metrics
        assert "frame_auc" in all_metrics
        assert "segment_f1" in all_metrics
        assert "boundary_f1" in all_metrics


class TestUtils:
    """Test cases for utility functions."""
    
    def test_get_device(self):
        """Test device selection."""
        # Test auto device selection
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        # Test specific device selection
        cpu_device = get_device("cpu")
        assert cpu_device.type == "cpu"
    
    def test_set_seed(self):
        """Test random seed setting."""
        # Set seed
        set_seed(42)
        
        # Generate random numbers
        rand1 = np.random.randn(10)
        
        # Set seed again
        set_seed(42)
        
        # Generate random numbers again
        rand2 = np.random.randn(10)
        
        # Should be the same
        np.testing.assert_array_equal(rand1, rand2)


if __name__ == "__main__":
    pytest.main([__file__])
