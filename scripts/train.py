#!/usr/bin/env python3
"""Main training script for Voice Activity Detection."""

import argparse
import logging
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.vad.models import EnergyVAD, CNNVAD, TransformerVAD
from src.vad.data import SyntheticVADDataset, create_data_loaders
from src.vad.train import VADTrainer
from src.vad.eval import VADEvaluator
from src.vad.utils import get_device, set_seed, load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_model(model_config: dict, sample_rate: int = 16000):
    """Create VAD model from configuration.
    
    Args:
        model_config: Model configuration dictionary.
        sample_rate: Sample rate of audio.
        
    Returns:
        VAD model instance.
    """
    model_type = model_config.get('_target_', '').split('.')[-1]
    
    if model_type == 'EnergyVAD':
        return EnergyVAD(
            frame_size=model_config.get('frame_size', 1024),
            hop_size=model_config.get('hop_size', 512),
            threshold=model_config.get('threshold', 0.02),
            energy_normalization=model_config.get('energy_normalization', True),
            sample_rate=sample_rate,
        )
    elif model_type == 'CNNVAD':
        return CNNVAD(
            input_dim=model_config.get('input_dim', 80),
            hidden_dims=model_config.get('hidden_dims', [128, 64, 32]),
            kernel_sizes=model_config.get('kernel_sizes', [3, 3, 3]),
            dropout=model_config.get('dropout', 0.2),
            sample_rate=sample_rate,
        )
    elif model_type == 'TransformerVAD':
        return TransformerVAD(
            input_dim=model_config.get('input_dim', 80),
            d_model=model_config.get('d_model', 256),
            nhead=model_config.get('nhead', 8),
            num_layers=model_config.get('num_layers', 6),
            dropout=model_config.get('dropout', 0.1),
            sample_rate=sample_rate,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_dataset(data_config: dict):
    """Create dataset from configuration.
    
    Args:
        data_config: Data configuration dictionary.
        
    Returns:
        Dataset instance.
    """
    dataset_type = data_config.get('_target_', '').split('.')[-1]
    
    if dataset_type == 'SyntheticVADDataset':
        return SyntheticVADDataset(
            num_samples=data_config.get('num_samples', 1000),
            duration=data_config.get('duration', 10.0),
            sample_rate=data_config.get('sample_rate', 16000),
            speech_probability=data_config.get('speech_probability', 0.6),
            noise_level=data_config.get('noise_level', 0.1),
            augment=data_config.get('augment', False),
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Voice Activity Detection model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="energy_vad",
        choices=["energy_vad", "cnn_vad", "transformer_vad"],
        help="Model type to train"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="synthetic",
        choices=["synthetic"],
        help="Dataset type to use"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, mps)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    config.model = OmegaConf.load(f"configs/model/{args.model}.yaml")
    config.data = OmegaConf.load(f"configs/data/{args.data}.yaml")
    config.device = args.device
    config.seed = args.seed
    
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    logger.info("Creating dataset...")
    dataset = create_dataset(config.data)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset,
        batch_size=config.training.batch_size,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    
    logger.info(f"Dataset created with {len(dataset)} samples")
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config.model, config.data.sample_rate)
    logger.info(f"Model created: {type(model).__name__}")
    
    # Create trainer
    trainer = VADTrainer(
        model=model,
        device=device,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.optimizer.weight_decay,
        gradient_clip_val=config.training.gradient_clip_val,
        early_stopping_patience=config.training.early_stopping.patience,
        min_delta=config.training.early_stopping.min_delta,
        monitor=config.training.early_stopping.monitor,
    )
    
    # Train model
    logger.info("Starting training...")
    training_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs,
        save_best=True,
        checkpoint_dir=str(output_dir / "checkpoints"),
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = VADEvaluator(
        model=model,
        device=device,
        collar_tolerance=config.evaluation.collar_tolerance,
        min_speech_duration=config.evaluation.min_speech_duration,
        min_silence_duration=config.evaluation.min_silence_duration,
        sample_rate=config.data.sample_rate,
    )
    
    # Test evaluation
    test_metrics = evaluator.evaluate_dataset(test_loader, "test")
    
    # Create leaderboard
    leaderboard_entry = evaluator.create_leaderboard(
        test_loader,
        f"{args.model}_{args.data}"
    )
    
    # Print results
    evaluator.print_leaderboard([leaderboard_entry])
    
    # Save results
    results = {
        "config": OmegaConf.to_container(config),
        "training_history": training_history,
        "test_metrics": test_metrics,
        "leaderboard_entry": leaderboard_entry,
    }
    
    evaluator.save_results(results, str(output_dir / "results.json"))
    
    # Save configuration
    OmegaConf.save(config, str(output_dir / "config.yaml"))
    
    logger.info(f"Training completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
