# ============================================================================
# BT_Classification/seed_config.py
# Global seed configuration for reproducible training
# ============================================================================

"""
Seed Configuration Module

This module MUST be imported FIRST in all pipeline stages to ensure
reproducible results across DVC pipeline runs.

Usage:
    from BT_Classification.seed_config import GLOBAL_SEED
    # Seeds are automatically set when module is imported
"""

import os
import random
import numpy as np
import tensorflow as tf

# Global seed value
SEED = 42


def set_all_seeds(seed=SEED):
    """
    Set all random seeds for reproducibility
    
    This function sets seeds for:
    - Python's random module
    - NumPy's random number generator
    - TensorFlow's random operations
    - Python's hash seed
    - TensorFlow deterministic operations
    
    Args:
        seed (int): Seed value (default: 42)
    
    Returns:
        int: The seed value that was set
    """
    
    # Python random seed
    random.seed(seed)
    
    # NumPy random seed
    np.random.seed(seed)
    
    # TensorFlow random seed
    tf.random.set_seed(seed)
    
    # Python hash seed (for dictionary ordering)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # TensorFlow deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # Disable TensorFlow warnings (optional)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print("="*70)
    print("REPRODUCIBLE MODE ENABLED")
    print("="*70)
    print(f"✓ Python random seed: {seed}")
    print(f"✓ NumPy random seed: {seed}")
    print(f"✓ TensorFlow random seed: {seed}")
    print(f"✓ Python hash seed: {seed}")
    print("✓ TensorFlow deterministic operations: enabled")
    print("✓ All pipeline stages will produce consistent results")
    print("="*70)
    
    return seed


# Automatically set seeds when this module is imported
# This ensures all DVC stages are seeded properly
GLOBAL_SEED = set_all_seeds(SEED)


# Export for easy access
__all__ = ['GLOBAL_SEED', 'set_all_seeds', 'SEED']