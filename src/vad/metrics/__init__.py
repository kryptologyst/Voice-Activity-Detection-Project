"""Evaluation metrics for Voice Activity Detection."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class VADMetrics:
    """Metrics calculator for Voice Activity Detection."""
    
    def __init__(
        self,
        collar_tolerance: float = 0.1,
        min_speech_duration: float = 0.1,
        min_silence_duration: float = 0.1,
        sample_rate: int = 16000,
    ):
        """Initialize VAD metrics calculator.
        
        Args:
            collar_tolerance: Tolerance for boundary detection in seconds.
            min_speech_duration: Minimum speech duration in seconds.
            min_silence_duration: Minimum silence duration in seconds.
            sample_rate: Sample rate of audio.
        """
        self.collar_tolerance = collar_tolerance
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.sample_rate = sample_rate
        
        # Convert to samples
        self.collar_samples = int(collar_tolerance * sample_rate)
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        self.min_silence_samples = int(min_silence_duration * sample_rate)
    
    def compute_frame_metrics(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, float]:
        """Compute frame-level metrics.
        
        Args:
            predictions: VAD predictions.
            targets: Ground truth VAD labels.
            
        Returns:
            Dictionary of frame-level metrics.
        """
        # Convert to numpy arrays
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Flatten arrays
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Ensure binary predictions
        predictions = (predictions > 0.5).astype(int)
        targets = targets.astype(int)
        
        # Compute metrics
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, zero_division=0)
        recall = recall_score(targets, predictions, zero_division=0)
        f1 = f1_score(targets, predictions, zero_division=0)
        
        # Compute AUC if we have probability predictions
        try:
            auc = roc_auc_score(targets, predictions)
        except ValueError:
            auc = 0.0
        
        return {
            "frame_accuracy": accuracy,
            "frame_precision": precision,
            "frame_recall": recall,
            "frame_f1": f1,
            "frame_auc": auc,
        }
    
    def compute_segment_metrics(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, float]:
        """Compute segment-level metrics.
        
        Args:
            predictions: VAD predictions.
            targets: Ground truth VAD labels.
            
        Returns:
            Dictionary of segment-level metrics.
        """
        # Convert to numpy arrays
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Flatten arrays
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Ensure binary predictions
        predictions = (predictions > 0.5).astype(int)
        targets = targets.astype(int)
        
        # Extract segments
        pred_segments = self._extract_segments(predictions)
        target_segments = self._extract_segments(targets)
        
        # Compute segment-level metrics
        segment_f1 = self._compute_segment_f1(pred_segments, target_segments)
        
        return {
            "segment_f1": segment_f1,
        }
    
    def compute_boundary_metrics(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, float]:
        """Compute boundary detection metrics.
        
        Args:
            predictions: VAD predictions.
            targets: Ground truth VAD labels.
            
        Returns:
            Dictionary of boundary metrics.
        """
        # Convert to numpy arrays
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Flatten arrays
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Ensure binary predictions
        predictions = (predictions > 0.5).astype(int)
        targets = targets.astype(int)
        
        # Extract boundaries
        pred_boundaries = self._extract_boundaries(predictions)
        target_boundaries = self._extract_boundaries(targets)
        
        # Compute boundary metrics
        boundary_f1 = self._compute_boundary_f1(pred_boundaries, target_boundaries)
        
        return {
            "boundary_f1": boundary_f1,
        }
    
    def compute_all_metrics(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, float]:
        """Compute all VAD metrics.
        
        Args:
            predictions: VAD predictions.
            targets: Ground truth VAD labels.
            
        Returns:
            Dictionary of all metrics.
        """
        # Frame-level metrics
        frame_metrics = self.compute_frame_metrics(predictions, targets)
        
        # Segment-level metrics
        segment_metrics = self.compute_segment_metrics(predictions, targets)
        
        # Boundary metrics
        boundary_metrics = self.compute_boundary_metrics(predictions, targets)
        
        # Combine all metrics
        all_metrics = {**frame_metrics, **segment_metrics, **boundary_metrics}
        
        return all_metrics
    
    def _extract_segments(self, vad_labels: np.ndarray) -> List[Tuple[int, int]]:
        """Extract speech segments from VAD labels.
        
        Args:
            vad_labels: VAD labels array.
            
        Returns:
            List of (start, end) segment tuples.
        """
        segments = []
        in_segment = False
        start = 0
        
        for i, label in enumerate(vad_labels):
            if label == 1 and not in_segment:
                # Start of speech segment
                in_segment = True
                start = i
            elif label == 0 and in_segment:
                # End of speech segment
                in_segment = False
                segments.append((start, i))
        
        # Handle case where segment continues to end
        if in_segment:
            segments.append((start, len(vad_labels)))
        
        return segments
    
    def _extract_boundaries(self, vad_labels: np.ndarray) -> List[int]:
        """Extract boundary positions from VAD labels.
        
        Args:
            vad_labels: VAD labels array.
            
        Returns:
            List of boundary positions.
        """
        boundaries = []
        
        for i in range(1, len(vad_labels)):
            if vad_labels[i] != vad_labels[i-1]:
                boundaries.append(i)
        
        return boundaries
    
    def _compute_segment_f1(
        self,
        pred_segments: List[Tuple[int, int]],
        target_segments: List[Tuple[int, int]],
    ) -> float:
        """Compute segment-level F1 score.
        
        Args:
            pred_segments: Predicted segments.
            target_segments: Target segments.
            
        Returns:
            Segment F1 score.
        """
        if not pred_segments and not target_segments:
            return 1.0
        if not pred_segments or not target_segments:
            return 0.0
        
        # Compute precision and recall
        tp = 0
        fp = 0
        fn = 0
        
        # Check each predicted segment
        for pred_start, pred_end in pred_segments:
            matched = False
            for target_start, target_end in target_segments:
                # Check if segments overlap significantly
                overlap_start = max(pred_start, target_start)
                overlap_end = min(pred_end, target_end)
                overlap_length = max(0, overlap_end - overlap_start)
                
                pred_length = pred_end - pred_start
                target_length = target_end - target_start
                
                # Consider it a match if overlap is significant
                if (overlap_length / max(pred_length, target_length)) > 0.5:
                    tp += 1
                    matched = True
                    break
            
            if not matched:
                fp += 1
        
        # Check for missed target segments
        for target_start, target_end in target_segments:
            matched = False
            for pred_start, pred_end in pred_segments:
                overlap_start = max(pred_start, target_start)
                overlap_end = min(pred_end, target_end)
                overlap_length = max(0, overlap_end - overlap_start)
                
                pred_length = pred_end - pred_start
                target_length = target_end - target_start
                
                if (overlap_length / max(pred_length, target_length)) > 0.5:
                    matched = True
                    break
            
            if not matched:
                fn += 1
        
        # Compute F1 score
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    def _compute_boundary_f1(
        self,
        pred_boundaries: List[int],
        target_boundaries: List[int],
    ) -> float:
        """Compute boundary detection F1 score.
        
        Args:
            pred_boundaries: Predicted boundary positions.
            target_boundaries: Target boundary positions.
            
        Returns:
            Boundary F1 score.
        """
        if not pred_boundaries and not target_boundaries:
            return 1.0
        if not pred_boundaries or not target_boundaries:
            return 0.0
        
        # Compute precision and recall
        tp = 0
        fp = 0
        fn = 0
        
        # Check each predicted boundary
        for pred_boundary in pred_boundaries:
            matched = False
            for target_boundary in target_boundaries:
                if abs(pred_boundary - target_boundary) <= self.collar_samples:
                    tp += 1
                    matched = True
                    break
            
            if not matched:
                fp += 1
        
        # Check for missed target boundaries
        for target_boundary in target_boundaries:
            matched = False
            for pred_boundary in pred_boundaries:
                if abs(pred_boundary - target_boundary) <= self.collar_samples:
                    matched = True
                    break
            
            if not matched:
                fn += 1
        
        # Compute F1 score
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return f1
