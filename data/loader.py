"""
PyTorch Dataset and DataLoader Utilities for Prompt-Tuned NER.

Handles text normalization, subword tokenization, label sequence alignment,
and virtual prompt token index shifting.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple

from .normalizer import ScriptNormalizer, AgglutinativePostProcessor
from .synthetic import SyntheticNERGenerator


class NERDataset(Dataset):
    """
    PyTorch Dataset for Named Entity Recognition with support for subword label alignment,
    script normalization, and virtual soft-prompt label index padding.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: Any,
        label2id: Dict[str, int] = SyntheticNERGenerator.LABEL2ID,
        max_length: int = 128,
        normalize_script: bool = True,
        strip_postpositions: bool = True,
        prompt_length: int = 0
    ):
        """
        Args:
            samples: List of dictionaries with 'tokens' and 'ner_tags' (or 'ner_ids').
            tokenizer: Hugging Face PreTrainedTokenizer instance.
            label2id: Mapping from string tags to integer label IDs.
            max_length: Maximum sequence token length.
            normalize_script: Whether to apply Unicode & script normalization.
            strip_postpositions: Whether to detach agglutinative postposition clitics.
            prompt_length: Total virtual soft-prompt length (P_task + P_lang) to shift label indices.
        """
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.prompt_length = prompt_length

        self.normalizer = ScriptNormalizer() if normalize_script else None
        self.postprocessor = AgglutinativePostProcessor() if strip_postpositions else None

        self.processed_data = self._process_samples(samples)

    def _process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        processed = []

        for sample in samples:
            raw_tokens = sample["tokens"]
            raw_tags = sample.get("ner_tags", [])
            if not raw_tags and "ner_ids" in sample:
                raw_tags = [SyntheticNERGenerator.ID2LABEL[idx] for idx in sample["ner_ids"]]

            # Step 1: Normalize script if enabled
            if self.normalizer:
                raw_tokens = self.normalizer.normalize_tokens(raw_tokens)

            # Step 2: Strip agglutinative postpositions if enabled
            if self.postprocessor and raw_tags:
                raw_tokens, raw_tags = self.postprocessor.process_sentence_entities(raw_tokens, raw_tags)

            # Step 3: Subword Tokenization & Label Alignment
            input_ids = []
            labels = []

            # Add CLS / BOS token
            cls_id = self.tokenizer.cls_token_id or self.tokenizer.bos_token_id or 101
            input_ids.append(cls_id)
            labels.append(-100)  # Ignore CLS token in loss calculation

            for word, tag in zip(raw_tokens, raw_tags):
                word_subwords = self.tokenizer.encode(word, add_special_tokens=False)
                tag_id = self.label2id.get(tag, 0)

                for i, subword in enumerate(word_subwords):
                    input_ids.append(subword)
                    # Align first subword token with entity tag, subsequent subwords with -100
                    if i == 0:
                        labels.append(tag_id)
                    else:
                        labels.append(-100)

            # Add SEP / EOS token
            sep_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id or 102
            input_ids.append(sep_id)
            labels.append(-100)

            # Step 4: Truncate or Pad to max_length
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]

            attention_mask = [1] * len(input_ids)

            pad_id = self.tokenizer.pad_token_id or 0
            pad_len = self.max_length - len(input_ids)
            if pad_len > 0:
                input_ids.extend([pad_id] * pad_len)
                labels.extend([-100] * pad_len)
                attention_mask.extend([0] * pad_len)

            # Step 5: Shift labels for virtual soft prompt tokens if prompt_length > 0
            # Virtual prompt tokens prepend L_prompt slots filled with -100 index
            if self.prompt_length > 0:
                prompt_label_pad = [-100] * self.prompt_length
                shifted_labels = prompt_label_pad + labels
                prompt_mask_pad = [1] * self.prompt_length
                shifted_attention_mask = prompt_mask_pad + attention_mask
            else:
                shifted_labels = labels
                shifted_attention_mask = attention_mask

            processed.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(shifted_attention_mask, dtype=torch.long),
                "labels": torch.tensor(shifted_labels, dtype=torch.long)
            })

        return processed

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.processed_data[idx]


def create_dataloaders(
    samples: List[Dict[str, Any]],
    tokenizer: Any,
    batch_size: int = 8,
    train_ratio: float = 0.8,
    prompt_length: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits samples into train/validation sets and creates PyTorch DataLoaders.
    """
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    train_dataset = NERDataset(train_samples, tokenizer, prompt_length=prompt_length)
    val_dataset = NERDataset(val_samples, tokenizer, prompt_length=prompt_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
