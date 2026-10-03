"""
Script Normalization and Agglutinative Postposition Processing Module.

Provides Unicode script unification across low-resource Indian languages
(Bhojpuri, Maithili, Santali, Hindi) and strips agglutinative case clitics
to preserve clean entity span boundaries.
"""

import re
import unicodedata
from typing import List, Tuple, Dict, Optional


class ScriptNormalizer:
    """
    Handles script unification, Unicode NFKC normalization, diacritic cleanup,
    and transliteration mapping for low-resource Indian language scripts.
    """

    # Mapping Ol Chiki (Santali script) ranges to Devanagari equivalents for cross-lingual alignment
    OL_CHIKI_TO_DEVANAGARI = {
        '\u1c5a': 'अ', '\u1c5b': 'आ', '\u1c5c': 'इ', '\u1c5d': 'उ', '\u1c5e': 'ए',
        '\u1c5f': 'ओ', '\u1c60': 'क', '\u1c61': 'ग', '\u1c62': 'च', '\u1c63': 'ज',
        '\u1c64': 'ट', '\u1c65': 'ड', '\u1c66': 'त', '\u1c67': 'द', '\u1c68': 'न',
        '\u1c69': 'प', '\u1c6a': 'ब', '\u1c6b': 'म', '\u1c6c': 'य', '\u1c6d': 'र',
        '\u1c6e': 'ल', '\u1c6f': 'व', '\u1c70': 'स', '\u1c71': 'ह', '\u1c72': 'ङ',
        '\u1c73': 'ञ', '\u1c74': 'ण', '\u1c75': 'फ', '\u1c76': 'भ', '\u1c77': 'ं'
    }

    # Transliteration mapping from Tirhuta (Maithili script) to Devanagari
    TIRHUTA_TO_DEVANAGARI = {
        '\u11480': 'अ', '\u11481': 'आ', '\u11482': 'इ', '\u11483': 'ई', '\u11484': 'उ',
        '\u11485': 'ऊ', '\u1148e': 'क', '\u1148f': 'ख', '\u11490': 'ग', '\u11491': 'घ',
        '\u11493': 'च', '\u11494': 'छ', '\u11495': 'ज', '\u11496': 'झ', '\u1149d': 'त',
        '\u1149e': 'थ', '\u1149f': 'द', '\u114a0': 'ध', '\u114a1': 'न', '\u114a2': 'प',
        '\u114a3': 'फ', '\u114a4': 'ब', '\u114a5': 'भ', '\u114a6': 'म', '\u114a7': 'य',
        '\u114a8': 'र', '\u114a9': 'ल', '\u114ab': 'श', '\u114ad': 'स', '\u114ae': 'ह'
    }

    def __init__(self, target_script: str = "devanagari"):
        """
        Args:
            target_script: Script target for normalization ('devanagari', 'latin', or 'raw').
        """
        self.target_script = target_script.lower()

    def normalize(self, text: str) -> str:
        """
        Performs full script normalization:
        1. Unicode NFKC normalization.
        2. Removal of zero-width joiners/non-joiners.
        3. Script unification (Ol Chiki/Tirhuta -> Devanagari if applicable).
        4. Nukta & whitespace standardization.
        """
        if not text:
            return ""

        # Step 1: NFKC Normalization
        normalized_text = unicodedata.normalize('NFKC', text)

        # Step 2: Remove zero-width formatting characters
        normalized_text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', normalized_text)

        # Step 3: Script Transliteration to Devanagari if present
        char_list = []
        for char in normalized_text:
            if char in self.OL_CHIKI_TO_DEVANAGARI:
                char_list.append(self.OL_CHIKI_TO_DEVANAGARI[char])
            elif char in self.TIRHUTA_TO_DEVANAGARI:
                char_list.append(self.TIRHUTA_TO_DEVANAGARI[char])
            else:
                char_list.append(char)
        normalized_text = "".join(char_list)

        # Step 4: Nukta unification (e.g. क़ -> क, ख़ -> ख for standardizing embeddings)
        nukta_map = {
            'क़': 'क', 'ख़': 'ख', 'ग़': 'ग', 'ज़': 'ज', 'ड़': 'ड', 'ढ़': 'ढ', 'फ़': 'फ', 'य़': 'य'
        }
        for k, v in nukta_map.items():
            normalized_text = normalized_text.replace(k, v)

        # Step 5: Collapse multiple spaces
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()

        return normalized_text

    def normalize_tokens(self, tokens: List[str]) -> List[str]:
        """Normalize a list of input text tokens."""
        return [self.normalize(token) for token in tokens]


class AgglutinativePostProcessor:
    """
    Strips agglutinative postposition case clitics attached to entity tokens
    in Indic languages (Bhojpuri, Maithili, Santali, Hindi).
    
    Example: 'पटनामें' -> base stem 'पटना', postposition 'में'
             'रामके'  -> base stem 'राम', postposition 'के'
    """

    # Postposition list sorted by length descending to match longer suffixes first
    POSTPOSITIONS_HINDI_BHOJPURI = [
        'का', 'की', 'के', 'को', 'में', 'से', 'पर', 'ने', 'तक', 'द्वारा',
        'सों', 'ला', 'हूँ', 'बाती', 'खातिर', 'अउर', 'लोगन', 'तनिक'
    ]

    POSTPOSITIONS_MAITHILI = [
        'सँ', 'कें', 'सौं', 'लेल', 'सँहु', 'पर', 'मे', 'केर', 'तक', 'द्वारा'
    ]

    POSTPOSITIONS_SANTALI = [
        'रे', 'ते', 'खान', 'खोन', 'रेन', 'रेयाक्', 'तेyaക്', 'लागीत्'
    ]

    POSTPOSITIONS_MARATHI = [
        'मध्ये', 'ना', 'ला', 'ने', 'वर', 'त', 'शी', 'हून', 'चा', 'ची', 'चे', 'च्या', 'पासून', 'बद्दल', 'सुद्धा'
    ]

    POSTPOSITIONS_BENGALI = [
        'থেকে', 'চেয়ে', 'দিয়ে', 'জন্য', 'মতো', 'পরে', 'সাথে', 'বিনা'
    ]

    POSTPOSITIONS_GUJARATI = [
        'માં', 'થી', 'ને', 'નો', 'ની', 'નું', 'ના', 'વડે'
    ]

    def __init__(self, min_stem_len: int = 2):
        """
        Args:
            min_stem_len: Minimum character length of root stem to prevent over-stemming short words.
        """
        self.min_stem_len = min_stem_len
        # Combine and sort all postpositions by length descending
        all_postpositions = set(
            self.POSTPOSITIONS_HINDI_BHOJPURI + 
            self.POSTPOSITIONS_MAITHILI + 
            self.POSTPOSITIONS_SANTALI +
            self.POSTPOSITIONS_MARATHI +
            self.POSTPOSITIONS_BENGALI +
            self.POSTPOSITIONS_GUJARATI
        )
        self.postpositions = sorted(list(all_postpositions), key=len, reverse=True)

    def strip_postposition(self, token: str) -> Tuple[str, Optional[str]]:
        """
        Strips attached postposition from a single token if stem length >= min_stem_len.

        Returns:
            Tuple of (base_stem, stripped_postposition)
        """
        if len(token) <= self.min_stem_len:
            return token, None

        for pp in self.postpositions:
            if token.endswith(pp) and (len(token) - len(pp)) >= self.min_stem_len:
                base_stem = token[:-len(pp)]
                return base_stem, pp

        return token, None

    def process_sentence_entities(
        self, tokens: List[str], tags: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Processes a sequence of tokens and IOB tags, splitting attached postpositions
        out of entity tokens so that the postposition itself does not share the entity tag.

        Example:
            Input:  tokens=['पटनामें', 'रहते', 'हैं'], tags=['B-LOC', 'O', 'O']
            Output: tokens=['पटना', 'में', 'रहते', 'हैं'], tags=['B-LOC', 'O', 'O', 'O']
        """
        new_tokens = []
        new_tags = []

        for token, tag in zip(tokens, tags):
            # Only strip postpositions from entity tokens (B- or I- tags)
            if tag != 'O':
                stem, stripped_pp = self.strip_postposition(token)
                if stripped_pp:
                    new_tokens.append(stem)
                    new_tags.append(tag)
                    new_tokens.append(stripped_pp)
                    new_tags.append('O')
                    continue
            new_tokens.append(token)
            new_tags.append(tag)

        return new_tokens, new_tags
