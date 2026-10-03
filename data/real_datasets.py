"""
Real Dataset Loader and Converter for Cross-Lingual NER.

Supports loading CoNLL-2003, JSON/JSONL format (WikiANN / IndicNER),
and standardizing raw texts through script normalization and postposition processing.
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional
from data.normalizer import ScriptNormalizer, AgglutinativePostProcessor


class RealNERDatasetLoader:
    """
    Parser and preprocessor for real-world NER datasets (CoNLL, IndicNER, WikiANN).
    """

    def __init__(self, preprocess: bool = True):
        self.preprocess = preprocess
        self.normalizer = ScriptNormalizer()
        self.postprocessor = AgglutinativePostProcessor()

    def load_conll_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads dataset from a CoNLL-formatted text file (Token Tag lines separated by blank lines).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CoNLL file not found: {file_path}")

        samples = []
        current_tokens = []
        current_tags = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("-DOCSTART-"):
                    if current_tokens:
                        samples.append(self._process_sample(current_tokens, current_tags))
                        current_tokens, current_tags = [], []
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    token = parts[0]
                    tag = parts[-1]  # Tag is typically the last column
                    current_tokens.append(token)
                    current_tags.append(tag)

            if current_tokens:
                samples.append(self._process_sample(current_tokens, current_tags))

        return samples

    def load_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads dataset from JSONL format (e.g. IndicNER / WikiANN format).
        Expected keys per line: "tokens": [...], "ner_tags" or "tags": [...]
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSONL file not found: {file_path}")

        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                tokens = data.get("tokens", [])
                tags = data.get("ner_tags", data.get("tags", []))
                if tokens and tags:
                    samples.append(self._process_sample(tokens, tags))

        return samples

    def load_indic_sample_corpus(self, language: str = "bhojpuri", dataset_category: str = "All Datasets") -> List[Dict[str, Any]]:
        """
        Returns a curated benchmark subset of real-world style sentences for target languages.
        """
        from data.examples import MULTILINGUAL_EXAMPLES, get_examples_for_lang_and_dataset
        
        # Standardize language key lookup
        matched_lang = "Bhojpuri"
        for lkey in MULTILINGUAL_EXAMPLES.keys():
            if lkey.lower() in language.lower() or language.lower() in lkey.lower():
                matched_lang = lkey
                break
        
        raw_samples = get_examples_for_lang_and_dataset(matched_lang, dataset_category)
        return [self._process_sample(s["tokens"], s["tags"], matched_lang.lower()) for s in raw_samples]


    def _process_sample(self, tokens: List[str], tags: List[str], language: str = "indic") -> Dict[str, Any]:
        """Applies normalization and postposition processing to tokens and tags."""
        if not self.preprocess:
            return {"tokens": tokens, "tags": tags, "language": language}

        # 1. Unicode normalization & transliteration
        norm_tokens = self.normalizer.normalize_tokens(tokens)

        # 2. Agglutinative postposition stripping
        proc_tokens, proc_tags = self.postprocessor.process_sentence_entities(norm_tokens, tags)

        return {
            "tokens": proc_tokens,
            "tags": proc_tags,
            "language": language
        }
