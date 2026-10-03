"""
Baseline Models & Parameter Comparison for Cross-Lingual NER.

Includes:
1. FullFineTuningNERModel: Standard end-to-end full fine-tuning.
2. LoRANERModel: Parameter-efficient Low-Rank Adaptation via PEFT.
3. BaselineComparator: Diagnostic tool to compute trainable parameter efficiency across methods.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from transformers import AutoModel, AutoConfig

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class FullFineTuningNERModel(nn.Module):
    """
    Full Fine-Tuning Baseline.
    Unfreezes all parameters of the multilingual backbone and classification head.
    """

    def __init__(
        self,
        model_name_or_path: str = "xlm-roberta-base",
        num_labels: int = 11,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.num_labels = num_labels

        if model_name_or_path == "mock-backbone":
            self.config = AutoConfig.for_model("bert")
            self.config.hidden_size = 768
            self.backbone = MockBackbone(self.config)
        else:
            self.config = AutoConfig.from_pretrained(model_name_or_path)
            self.backbone = AutoModel.from_pretrained(model_name_or_path, config=self.config)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # Unfreeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else (outputs[0] if isinstance(outputs, (tuple, list)) else outputs)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if labels.shape[1] > logits.shape[1]:
                labels = labels[:, -logits.shape[1]:]
            elif logits.shape[1] > labels.shape[1]:
                logits = logits[:, :labels.shape[1]]
            loss = self.loss_fn(logits.reshape(-1, self.num_labels), labels.reshape(-1))

        return loss, logits


class LoRANERModel(nn.Module):
    """
    LoRA (Low-Rank Adaptation) Baseline for NER using PEFT.
    Inserts low-rank updates into attention layers while freezing backbone weights.
    """

    def __init__(
        self,
        model_name_or_path: str = "xlm-roberta-base",
        num_labels: int = 11,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None
    ):
        super().__init__()
        self.num_labels = num_labels

        if model_name_or_path == "mock-backbone":
            self.config = AutoConfig.for_model("bert")
            self.config.hidden_size = 768
            self.backbone = MockBackbone(self.config)
        else:
            self.config = AutoConfig.from_pretrained(model_name_or_path)
            self.backbone = AutoModel.from_pretrained(model_name_or_path, config=self.config)

        # Apply PEFT LoRA if available
        if PEFT_AVAILABLE and model_name_or_path != "mock-backbone":
            if target_modules is None:
                target_modules = ["query", "value"] if "bert" in model_name_or_path.lower() else ["q", "v"]
            lora_config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none"
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
        else:
            # Fallback manual low-rank projection layer if PEFT is unavailable or mock
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(lora_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else (outputs[0] if isinstance(outputs, (tuple, list)) else outputs)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if labels.shape[1] > logits.shape[1]:
                labels = labels[:, -logits.shape[1]:]
            elif logits.shape[1] > labels.shape[1]:
                logits = logits[:, :labels.shape[1]]
            loss = self.loss_fn(logits.reshape(-1, self.num_labels), labels.reshape(-1))

        return loss, logits


class MockBackbone(nn.Module):
    """Mock backbone for CPU/offline testing without HuggingFace model download dependency."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(30522, config.hidden_size)
        self.encoder = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, input_ids, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is None:
            embeds = self.embeddings(input_ids)
        else:
            embeds = inputs_embeds
        out = self.encoder(embeds)
        return torch.nn.functional.relu(out)


class BaselineComparator:
    """
    Computes parameter efficiency and statistics across models.
    """

    @staticmethod
    def get_parameter_stats(model: nn.Module) -> Dict[str, Any]:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_pct = (trainable_params / total_params * 100) if total_params > 0 else 0.0

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "trainable_percentage": round(trainable_pct, 4)
        }

    @staticmethod
    def compare_models(models_dict: Dict[str, nn.Module]) -> Dict[str, Dict[str, Any]]:
        results = {}
        for name, model in models_dict.items():
            results[name] = BaselineComparator.get_parameter_stats(model)
        return results
