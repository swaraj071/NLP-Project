"""
Evaluation package for span-level NER evaluation.
Implements seqeval metrics computation with subword token alignment handling.
"""

from .metrics import NEREvaluator
from .benchmark import run_comparative_benchmark

__all__ = ["NEREvaluator", "run_comparative_benchmark"]

