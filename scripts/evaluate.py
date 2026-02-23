#!/usr/bin/env python3
"""Evaluation script for Voice Activity Detection models."""

import argparse
import logging
from pathlib import Path
from typing import List

import torch
from omegaconf import OmegaConf

from src.vad.models import EnergyVAD, CNNVAD, TransformerVAD
from src.vad.data import SyntheticVADDataset, create_data_loaders
from src.vad.eval import VADEvaluator
from src.vad.utils import get_device, set_seed, load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_model(model_name: str, sample_rate: int = 16000):
    """Create VAD model from name.
    
    Args:
        model_name: Name of the model to create.
        sample_rate: Sample rate of audio.
        
    Returns:
        VAD model instance.
    """
    if model_name == "energy_vad":
        return EnergyVAD(
            frame_size=1024,
            hop_size=512,
            threshold=0.02,
            energy_normalization=True,
            sample_rate=sample_rate,
        )
    elif model_name == "cnn_vad":
        return CNNVAD(
            input_dim=80,
            hidden_dims=[128, 64, 32],
            kernel_sizes=[3, 3, 3],
            dropout=0.2,
            sample_rate=sample_rate,
        )
    elif model_name == "transformer_vad":
        return TransformerVAD(
            input_dim=80,
            d_model=256,
            nhead=8,
            num_layers=6,
            dropout=0.1,
            sample_rate=sample_rate,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Voice Activity Detection models")
    parser.add_argument(
        "--models",
        type=str,
        default="energy_vad,cnn_vad,transformer_vad",
        help="Comma-separated list of models to evaluate"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="synthetic",
        help="Dataset type to use"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples for evaluation"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration of each sample in seconds"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
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
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse model names
    model_names = [name.strip() for name in args.models.split(",")]
    logger.info(f"Evaluating models: {model_names}")
    
    # Create dataset
    logger.info("Creating evaluation dataset...")
    dataset = SyntheticVADDataset(
        num_samples=args.num_samples,
        duration=args.duration,
        sample_rate=16000,
        speech_probability=0.6,
        noise_level=0.1,
        augment=False,
    )
    
    # Create data loaders
    _, _, test_loader = create_data_loaders(
        dataset,
        batch_size=32,
        train_split=0.0,
        val_split=0.0,
        test_split=1.0,
        num_workers=4,
        pin_memory=True,
    )
    
    logger.info(f"Evaluation dataset created with {len(dataset)} samples")
    
    # Create models and evaluators
    models = []
    evaluators = []
    
    for model_name in model_names:
        logger.info(f"Creating {model_name} model...")
        model = create_model(model_name, sample_rate=16000)
        evaluator = VADEvaluator(
            model=model,
            device=device,
            collar_tolerance=0.1,
            min_speech_duration=0.1,
            min_silence_duration=0.1,
            sample_rate=16000,
        )
        
        models.append((model, model_name))
        evaluators.append(evaluator)
    
    # Evaluate models
    logger.info("Starting evaluation...")
    
    all_results = []
    
    for evaluator, model_name in zip(evaluators, model_names):
        logger.info(f"Evaluating {model_name}...")
        
        # Evaluate on test data
        test_metrics = evaluator.evaluate_dataset(test_loader, "test")
        
        # Create leaderboard entry
        leaderboard_entry = evaluator.create_leaderboard(test_loader, model_name)
        
        # Store results
        results = {
            "model": model_name,
            "test_metrics": test_metrics,
            "leaderboard_entry": leaderboard_entry,
        }
        
        all_results.append(results)
        
        # Log results
        logger.info(f"Results for {model_name}:")
        for metric_name, value in test_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
    
    # Create leaderboard
    leaderboard = [result["leaderboard_entry"] for result in all_results]
    
    # Print leaderboard
    if evaluators:
        evaluators[0].print_leaderboard(leaderboard)
    
    # Save results
    import json
    
    results_data = {
        "evaluation_config": {
            "models": model_names,
            "data": args.data,
            "num_samples": args.num_samples,
            "duration": args.duration,
            "device": str(device),
            "seed": args.seed,
        },
        "results": all_results,
        "leaderboard": leaderboard,
    }
    
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {results_path}")
    
    # Save leaderboard as CSV
    import pandas as pd
    
    df = pd.DataFrame(leaderboard)
    csv_path = output_dir / "leaderboard.csv"
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Leaderboard saved to {csv_path}")


if __name__ == "__main__":
    main()
