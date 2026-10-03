"""
Training package for multi-stage disentangled prompt tuning.
Manages Stage 1 anchor task training and Stage 2 language prompt adaptation.
"""

from .trainer import MultiStagePromptTrainer

__all__ = ["MultiStagePromptTrainer"]
