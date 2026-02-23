"""Evaluation module for Voice Activity Detection."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models import BaseVAD
from ..metrics import VADMetrics
from ..utils import get_device

logger = logging.getLogger(__name__)


class VADEvaluator:
    """Evaluator class for Voice Activity Detection models."""
    
    def __init__(
        self,
        model: BaseVAD,
        device: torch.device,
        collar_tolerance: float = 0.1,
        min_speech_duration: float = 0.1,
        min_silence_duration: float = 0.1,
        sample_rate: int = 16000,
    ):
        """Initialize VAD evaluator.
        
        Args:
            model: VAD model to evaluate.
            device: Device to use for evaluation.
            collar_tolerance: Tolerance for boundary detection in seconds.
            min_speech_duration: Minimum speech duration in seconds.
            min_silence_duration: Minimum silence duration in seconds.
            sample_rate: Sample rate of audio.
        """
        self.model = model.to(device)
        self.device = device
        self.metrics = VADMetrics(
            collar_tolerance=collar_tolerance,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            sample_rate=sample_rate,
        )
    
    def evaluate_dataset(
        self,
        data_loader: DataLoader,
        dataset_name: str = "test",
    ) -> Dict[str, float]:
        """Evaluate model on a dataset.
        
        Args:
            data_loader: Data loader for evaluation.
            dataset_name: Name of the dataset.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info(f"Evaluating on {dataset_name} dataset")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for audio, targets in data_loader:
                # Move to device
                audio = audio.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(audio)
                
                # Store predictions and targets for metrics
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.metrics.compute_all_metrics(all_predictions, all_targets)
        
        # Log results
        logger.info(f"Evaluation results for {dataset_name}:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        return metrics
    
    def evaluate_audio(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        targets: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Evaluate model on a single audio file.
        
        Args:
            audio: Input audio array or tensor.
            targets: Ground truth VAD labels (optional).
            
        Returns:
            Dictionary containing predictions and optionally metrics.
        """
        self.model.eval()
        
        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Move to device
        audio = audio.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(audio)
        
        # Convert predictions to numpy
        predictions = predictions.cpu().numpy()
        
        result = {"predictions": predictions}
        
        # Compute metrics if targets are provided
        if targets is not None:
            if isinstance(targets, torch.Tensor):
                targets = targets.cpu().numpy()
            
            metrics = self.metrics.compute_all_metrics(predictions, targets)
            result.update(metrics)
        
        return result
    
    def create_leaderboard(
        self,
        test_loader: DataLoader,
        model_name: str = "VAD Model",
    ) -> Dict[str, Union[str, float]]:
        """Create a leaderboard entry for the model.
        
        Args:
            test_loader: Test data loader.
            model_name: Name of the model.
            
        Returns:
            Dictionary containing leaderboard entry.
        """
        # Evaluate model
        metrics = self.evaluate_dataset(test_loader, "test")
        
        # Create leaderboard entry
        leaderboard_entry = {
            "model": model_name,
            "frame_accuracy": metrics["frame_accuracy"],
            "frame_f1": metrics["frame_f1"],
            "frame_precision": metrics["frame_precision"],
            "frame_recall": metrics["frame_recall"],
            "frame_auc": metrics["frame_auc"],
            "segment_f1": metrics["segment_f1"],
            "boundary_f1": metrics["boundary_f1"],
        }
        
        return leaderboard_entry
    
    def compare_models(
        self,
        models: List[Tuple[BaseVAD, str]],
        test_loader: DataLoader,
    ) -> List[Dict[str, Union[str, float]]]:
        """Compare multiple models on the same test data.
        
        Args:
            models: List of (model, name) tuples.
            test_loader: Test data loader.
            
        Returns:
            List of leaderboard entries.
        """
        leaderboard = []
        
        for model, name in models:
            # Create evaluator for this model
            evaluator = VADEvaluator(
                model=model,
                device=self.device,
                collar_tolerance=self.metrics.collar_tolerance,
                min_speech_duration=self.metrics.min_speech_duration,
                min_silence_duration=self.metrics.min_silence_duration,
                sample_rate=self.metrics.sample_rate,
            )
            
            # Evaluate model
            entry = evaluator.create_leaderboard(test_loader, name)
            leaderboard.append(entry)
        
        # Sort by frame F1 score
        leaderboard.sort(key=lambda x: x["frame_f1"], reverse=True)
        
        return leaderboard
    
    def print_leaderboard(
        self,
        leaderboard: List[Dict[str, Union[str, float]]],
    ) -> None:
        """Print a formatted leaderboard.
        
        Args:
            leaderboard: List of leaderboard entries.
        """
        print("\n" + "="*80)
        print("VOICE ACTIVITY DETECTION LEADERBOARD")
        print("="*80)
        print(f"{'Rank':<4} {'Model':<20} {'Frame F1':<10} {'Frame Acc':<10} {'Segment F1':<12} {'Boundary F1':<12}")
        print("-"*80)
        
        for i, entry in enumerate(leaderboard):
            print(
                f"{i+1:<4} "
                f"{entry['model']:<20} "
                f"{entry['frame_f1']:<10.4f} "
                f"{entry['frame_accuracy']:<10.4f} "
                f"{entry['segment_f1']:<12.4f} "
                f"{entry['boundary_f1']:<12.4f}"
            )
        
        print("="*80)
    
    def save_results(
        self,
        results: Dict[str, float],
        output_path: str,
    ) -> None:
        """Save evaluation results to file.
        
        Args:
            results: Evaluation results dictionary.
            output_path: Path to save results.
        """
        import json
        import os
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def load_results(self, input_path: str) -> Dict[str, float]:
        """Load evaluation results from file.
        
        Args:
            input_path: Path to results file.
            
        Returns:
            Evaluation results dictionary.
        """
        import json
        
        with open(input_path, 'r') as f:
            results = json.load(f)
        
        return results
