"""Utility functions for Voice Activity Detection project."""

import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device preference ("auto", "cpu", "cuda", "mps").
        
    Returns:
        torch.device: The selected device.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        DictConfig: Loaded configuration.
    """
    from omegaconf import OmegaConf
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save.
        config_path: Path to save configuration.
    """
    from omegaconf import OmegaConf
    
    OmegaConf.save(config, config_path)


def anonymize_filename(filename: str) -> str:
    """Anonymize filename by removing potentially identifying information.
    
    Args:
        filename: Original filename.
        
    Returns:
        str: Anonymized filename.
    """
    import hashlib
    import os
    
    # Extract extension
    name, ext = os.path.splitext(filename)
    
    # Create hash of original name
    hash_obj = hashlib.md5(name.encode())
    anonymized_name = hash_obj.hexdigest()[:8]
    
    return f"{anonymized_name}{ext}"


def remove_pii_from_text(text: str) -> str:
    """Remove potentially identifying information from text.
    
    Args:
        text: Input text.
        
    Returns:
        str: Text with PII removed.
    """
    import re
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    
    # Remove SSN-like patterns
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    
    return text


class PrivacyGuard:
    """Privacy guard for handling sensitive data."""
    
    def __init__(self, anonymize_filenames: bool = True, remove_pii: bool = True):
        """Initialize privacy guard.
        
        Args:
            anonymize_filenames: Whether to anonymize filenames.
            remove_pii: Whether to remove PII from text.
        """
        self.anonymize_filenames = anonymize_filenames
        self.remove_pii = remove_pii
    
    def process_filename(self, filename: str) -> str:
        """Process filename for privacy.
        
        Args:
            filename: Original filename.
            
        Returns:
            str: Processed filename.
        """
        if self.anonymize_filenames:
            return anonymize_filename(filename)
        return filename
    
    def process_text(self, text: str) -> str:
        """Process text for privacy.
        
        Args:
            text: Original text.
            
        Returns:
            str: Processed text.
        """
        if self.remove_pii:
            return remove_pii_from_text(text)
        return text
