"""
Training Module
==============

This module handles model training with configuration management.
"""

from .trainer import AMRTrainer
from .model_config import ModelConfig

__all__ = ['AMRTrainer', 'ModelConfig']
