"""
Multi-Stage Prompt Trainer Module.

Executes Stage 1 (P_task anchor task training) and Stage 2 (P_lang target language adaptation).
Supports custom learning rates, optimizer setup, evaluation callbacks, and model checkpointing.
"""

import os
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
from evaluation.metrics import NEREvaluator


class MultiStagePromptTrainer:
    """
    Manages multi-stage prompt tuning execution for cross-lingual NER models.
    """

    def __init__(

        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 2e-3,
        weight_decay: float = 0.01,
        device: Optional[torch.device] = None,
        evaluator: Optional[NEREvaluator] = None
    ):
        """
        Args:
            model: Instance of DisentangledPromptNERModel.
            train_loader: PyTorch DataLoader for training samples.
            val_loader: Optional PyTorch DataLoader for validation.
            learning_rate: Higher learning rate suitable for soft prompt tuning (e.g. 1e-3 to 5e-3).
            weight_decay: Weight decay factor for AdamW optimizer.
            device: PyTorch device (cuda, mps, cpu).
            evaluator: Instance of NEREvaluator for calculating span F1 scores.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)

        self.evaluator = evaluator or NEREvaluator()

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """
        Filters trainable parameters (requires_grad=True) and initializes AdamW optimizer.
        """
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)

    def train_epoch(self, optimizer: torch.optim.Optimizer) -> float:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            optimizer.zero_grad()
            loss, _ = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train_stage(self, stage: int = 1, epochs: int = 3) -> Dict[str, Any]:
        """
        Executes training for a specific prompt tuning stage.

        Args:
            stage: Stage number (1 for P_task, 2 for P_lang).
            epochs: Number of training epochs.

        Returns:
            Dictionary containing epoch loss history and evaluation metrics.
        """
        print(f"--- Starting Stage {stage} Prompt Training ({'P_task Anchor' if stage == 1 else 'P_lang Adaptation'}) ---")
        self.model.configure_stage(stage)

        optimizer = self._get_optimizer()
        history = {"loss": [], "val_f1": []}

        for epoch in range(1, epochs + 1):
            epoch_loss = self.train_epoch(optimizer)
            history["loss"].append(epoch_loss)

            val_metrics = {}
            if self.val_loader:
                val_metrics = self.evaluate(self.val_loader)
                val_f1 = val_metrics.get("overall_f1", 0.0)
                history["val_f1"].append(val_f1)
                print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} | Val F1: {val_f1:.4f}")
            else:
                print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}")

        return history

    def evaluate(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Runs evaluation over a DataLoader and computes span-level seqeval metrics."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                _, logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(logits, dim=-1)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        # Concatenate predictions across batches
        preds_tensor = torch.cat(all_preds, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        # Calculate span metrics using NEREvaluator
        metrics = self.evaluator.compute_metrics(preds_tensor, labels_tensor)
        return metrics

    def save_prompt_checkpoint(self, save_directory: str):
        """Saves tuned soft prompt parameters and model configuration."""
        os.makedirs(save_directory, exist_ok=True)
        checkpoint_path = os.path.join(save_directory, "soft_prompt_weights.pt")
        torch.save(self.model.soft_prompt.state_dict(), checkpoint_path)
        print(f"Soft prompt checkpoint saved to {checkpoint_path}")

    def load_prompt_checkpoint(self, load_directory: str):
        """Loads soft prompt parameters from a saved checkpoint."""
        checkpoint_path = os.path.join(load_directory, "soft_prompt_weights.pt")
        if os.path.exists(checkpoint_path):
            self.model.soft_prompt.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"Soft prompt weights loaded successfully from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint path {checkpoint_path} not found.")
