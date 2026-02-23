"""Training module for Voice Activity Detection."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models import BaseVAD
from ..metrics import VADMetrics
from ..utils import get_device, set_seed

logger = logging.getLogger(__name__)


class VADTrainer:
    """Trainer class for Voice Activity Detection models."""
    
    def __init__(
        self,
        model: BaseVAD,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        gradient_clip_val: float = 1.0,
        early_stopping_patience: int = 20,
        min_delta: float = 0.001,
        monitor: str = "val_loss",
    ):
        """Initialize VAD trainer.
        
        Args:
            model: VAD model to train.
            device: Device to use for training.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for optimizer.
            gradient_clip_val: Gradient clipping value.
            early_stopping_patience: Early stopping patience.
            min_delta: Minimum change for early stopping.
            monitor: Metric to monitor for early stopping.
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.monitor = monitor
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            verbose=True,
        )
        
        # Initialize loss function
        self.criterion = nn.BCELoss()
        
        # Initialize metrics
        self.metrics = VADMetrics()
        
        # Training state
        self.best_score = float('inf')
        self.patience_counter = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.
            
        Returns:
            Tuple of (average_loss, metrics_dict).
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            leave=False,
        )
        
        for batch_idx, (audio, targets) in enumerate(progress_bar):
            # Move to device
            audio = audio.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(audio)
            
            # Compute loss
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_val,
                )
            
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Store predictions and targets for metrics
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(targets.detach().cpu())
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Compute average loss
        avg_loss = total_loss / len(train_loader)
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.metrics.compute_all_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def validate_epoch(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader.
            epoch: Current epoch number.
            
        Returns:
            Tuple of (average_loss, metrics_dict).
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for audio, targets in val_loader:
                # Move to device
                audio = audio.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(audio)
                
                # Compute loss
                loss = self.criterion(predictions, targets)
                
                # Accumulate loss
                total_loss += loss.item()
                
                # Store predictions and targets for metrics
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Compute average loss
        avg_loss = total_loss / len(val_loader)
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.metrics.compute_all_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        save_best: bool = True,
        checkpoint_dir: str = "checkpoints",
    ) -> Dict[str, list]:
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Number of epochs to train.
            save_best: Whether to save best model.
            checkpoint_dir: Directory to save checkpoints.
            
        Returns:
            Training history dictionary.
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Train epoch
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate epoch
            val_loss, val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Store history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["train_metrics"].append(train_metrics)
            self.training_history["val_metrics"].append(val_metrics)
            
            # Log progress
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val F1: {val_metrics['frame_f1']:.4f}"
            )
            
            # Early stopping check
            if self._check_early_stopping(val_metrics):
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Save best model
            if save_best and self._is_best_score(val_metrics):
                self._save_checkpoint(checkpoint_dir, epoch, val_metrics)
        
        return self.training_history
    
    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """Check if early stopping criteria is met.
        
        Args:
            metrics: Validation metrics.
            
        Returns:
            True if early stopping should be triggered.
        """
        current_score = metrics.get(self.monitor, float('inf'))
        
        if current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.early_stopping_patience
    
    def _is_best_score(self, metrics: Dict[str, float]) -> bool:
        """Check if current score is the best.
        
        Args:
            metrics: Validation metrics.
            
        Returns:
            True if current score is the best.
        """
        current_score = metrics.get(self.monitor, float('inf'))
        return current_score < self.best_score
    
    def _save_checkpoint(
        self,
        checkpoint_dir: str,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Save model checkpoint.
        
        Args:
            checkpoint_dir: Directory to save checkpoint.
            epoch: Current epoch.
            metrics: Validation metrics.
        """
        import os
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_score": self.best_score,
            "metrics": metrics,
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Saved best model checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_score = checkpoint["best_score"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def evaluate(
        self,
        test_loader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for audio, targets in test_loader:
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
        
        return metrics
