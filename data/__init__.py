"""
Data processing package for cross-lingual NER in indigenous Indian languages.
Includes script normalization, agglutinative postposition stripping, dataset loading, and synthetic generation.
"""

from .normalizer import ScriptNormalizer, AgglutinativePostProcessor
from .loader import NERDataset, create_dataloaders
from .synthetic import SyntheticNERGenerator
from .real_datasets import RealNERDatasetLoader
from .examples import MULTILINGUAL_EXAMPLES, DATASET_CATEGORIES, LANGUAGE_LIST, get_examples_for_lang_and_dataset

__all__ = [
    "ScriptNormalizer",
    "AgglutinativePostProcessor",
    "NERDataset",
    "create_dataloaders",
    "SyntheticNERGenerator",
    "RealNERDatasetLoader",
    "MULTILINGUAL_EXAMPLES",
    "DATASET_CATEGORIES",
    "LANGUAGE_LIST",
    "get_examples_for_lang_and_dataset",
]


