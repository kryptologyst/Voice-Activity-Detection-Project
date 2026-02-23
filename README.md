# Voice Activity Detection (VAD) Project

Showcase-ready implementation of Voice Activity Detection systems for research and educational purposes.

## Privacy & Ethics Disclaimer

**IMPORTANT: This is a research and educational demonstration only.**

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

## Overview

Voice Activity Detection (VAD) is the process of detecting the presence or absence of human speech in an audio signal. This project implements multiple VAD approaches:

- **Energy-based VAD**: Traditional energy thresholding method
- **CNN-based VAD**: Convolutional neural network for frame-level classification
- **Transformer-based VAD**: Transformer architecture for sequence modeling
- **Spectral feature-based VAD**: Uses MFCC and spectral features

## Features

- Multiple VAD model architectures
- Comprehensive evaluation metrics (frame-level, segment-level, boundary detection)
- Synthetic dataset generation for testing
- Interactive Streamlit demo
- Privacy-preserving design
- Modern PyTorch implementation
- Configurable training pipeline
- Leaderboard system for model comparison

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA/MPS support (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Voice-Activity-Detection-Project.git
cd Voice-Activity-Detection-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### 1. Run the Interactive Demo

Launch the Streamlit demo application:

```bash
streamlit run demo/streamlit_app.py
```

The demo allows you to:
- Upload audio files for VAD analysis
- Generate synthetic audio with known ground truth
- Compare different VAD models
- Visualize results with interactive plots
- Download analysis results

### 2. Train a Model

Train a VAD model using the synthetic dataset:

```bash
python scripts/train.py --model energy_vad --data synthetic
```

Available models:
- `energy_vad`: Energy-based VAD
- `cnn_vad`: CNN-based VAD
- `transformer_vad`: Transformer-based VAD

### 3. Evaluate Models

Compare multiple models:

```bash
python scripts/evaluate.py --models energy_vad,cnn_vad,transformer_vad
```

## Project Structure

```
0687_Voice_Activity_Detection/
├── src/vad/                    # Core VAD implementation
│   ├── models/                 # VAD model architectures
│   ├── data/                   # Data loading and preprocessing
│   ├── train/                  # Training utilities
│   ├── eval/                   # Evaluation metrics and tools
│   ├── metrics/                # VAD-specific metrics
│   └── utils/                  # Utility functions
├── configs/                    # Configuration files
│   ├── model/                  # Model configurations
│   ├── data/                   # Dataset configurations
│   ├── training/               # Training configurations
│   └── evaluation/             # Evaluation configurations
├── scripts/                    # Training and evaluation scripts
├── demo/                       # Interactive demo application
├── tests/                      # Unit tests
├── notebooks/                  # Jupyter notebooks for analysis
├── assets/                     # Generated assets and visualizations
├── data/                       # Dataset storage
├── checkpoints/                # Model checkpoints
└── logs/                       # Training logs
```

## Configuration

The project uses Hydra/OmegaConf for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/model/`: Model-specific configurations
- `configs/data/`: Dataset configurations
- `configs/training/`: Training hyperparameters
- `configs/evaluation/`: Evaluation settings

### Example Configuration

```yaml
# Model configuration
model:
  _target_: src.vad.models.energy_vad.EnergyVAD
  frame_size: 1024
  hop_size: 512
  threshold: 0.02
  energy_normalization: true

# Data configuration
data:
  _target_: src.vad.data.synthetic_dataset.SyntheticVADDataset
  num_samples: 1000
  duration: 10.0
  sample_rate: 16000
  speech_probability: 0.6
```

## Models

### Energy-based VAD

Traditional energy thresholding approach:
- Computes frame-wise energy
- Applies threshold for speech detection
- Fast and lightweight
- Good baseline for comparison

### CNN-based VAD

Convolutional neural network approach:
- Uses mel-spectrogram features
- 1D CNN architecture
- Frame-level classification
- Better performance than energy-based

### Transformer-based VAD

Transformer architecture:
- Self-attention mechanism
- Sequence modeling capabilities
- State-of-the-art performance
- More computationally intensive

## Evaluation Metrics

The project provides comprehensive evaluation metrics:

### Frame-level Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve

### Segment-level Metrics
- **Segment F1**: F1 score for speech segments
- **Boundary F1**: F1 score for boundary detection

### Custom Metrics
- Collar tolerance for boundary detection
- Minimum speech/silence duration filtering
- Temporal smoothing options

## Dataset

### Synthetic Dataset

The project includes a synthetic dataset generator that creates:
- Audio with known speech/silence segments
- Configurable speech probability
- Background noise
- Data augmentation options

### Dataset Schema

```csv
id,path,sample_rate,duration,speech_percentage,split
sample_001,data/wav/sample_001.wav,16000,10.0,0.6,train
sample_002,data/wav/sample_002.wav,16000,10.0,0.4,val
```

## Training

### Training Pipeline

1. **Data Loading**: Load and preprocess audio data
2. **Feature Extraction**: Extract audio features (mel-spectrogram, MFCC, etc.)
3. **Model Training**: Train VAD model with configurable hyperparameters
4. **Validation**: Monitor training progress with validation metrics
5. **Early Stopping**: Prevent overfitting with early stopping
6. **Checkpointing**: Save best models automatically

### Training Commands

```bash
# Train energy-based VAD
python scripts/train.py --model energy_vad --data synthetic

# Train CNN-based VAD
python scripts/train.py --model cnn_vad --data synthetic

# Train transformer-based VAD
python scripts/train.py --model transformer_vad --data synthetic

# Custom configuration
python scripts/train.py --config configs/custom_config.yaml
```

## Evaluation

### Model Comparison

Compare multiple models on the same test data:

```bash
python scripts/evaluate.py --models energy_vad,cnn_vad,transformer_vad
```

### Leaderboard

The evaluation system generates a leaderboard with:
- Model rankings
- Performance metrics
- Statistical significance testing
- Visualization of results

## Demo Application

### Streamlit Demo Features

- **Audio Upload**: Upload audio files for analysis
- **Synthetic Generation**: Generate test audio with known ground truth
- **Model Selection**: Choose between different VAD models
- **Interactive Visualization**: Plot audio waveforms and VAD predictions
- **Metrics Display**: Show evaluation metrics
- **Results Download**: Export analysis results

### Running the Demo

```bash
streamlit run demo/streamlit_app.py
```

## API Reference

### Core Classes

#### `BaseVAD`
Base class for all VAD models.

```python
from src.vad.models import BaseVAD

class CustomVAD(BaseVAD):
    def forward(self, audio):
        # Implement VAD logic
        pass
    
    def predict(self, audio):
        # Implement prediction logic
        pass
```

#### `VADTrainer`
Training utility for VAD models.

```python
from src.vad.train import VADTrainer

trainer = VADTrainer(
    model=model,
    device=device,
    learning_rate=0.001
)
```

#### `VADEvaluator`
Evaluation utility for VAD models.

```python
from src.vad.eval import VADEvaluator

evaluator = VADEvaluator(
    model=model,
    device=device
)
```

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Style

The project uses:
- Black for code formatting
- Ruff for linting
- Pre-commit hooks for quality assurance

```bash
# Format code
black src/ scripts/ demo/

# Lint code
ruff check src/ scripts/ demo/

# Install pre-commit hooks
pre-commit install
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{vad_project,
  title={Voice Activity Detection},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Voice-Activity-Detection-Project}
}
```

## Acknowledgments

- PyTorch team for the deep learning framework
- Librosa for audio processing utilities
- Streamlit for the demo framework
- The open-source community for various dependencies

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the demo application

## Changelog

### Version 1.0.0
- Initial release
- Energy-based, CNN-based, and Transformer-based VAD models
- Synthetic dataset generation
- Interactive Streamlit demo
- Comprehensive evaluation metrics
- Privacy-preserving design
# Voice-Activity-Detection-Project
