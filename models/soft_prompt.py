"""
Disentangled Soft Prompt Tuning Module.

Implements task-specific (P_task) and language-specific (P_lang) learnable soft prompts
that prepend continuous virtual token embeddings to input word embeddings.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class DisentangledSoftPrompt(nn.Module):
    """
    Disentangled Soft Prompt module containing:
    - P_task: Learnable prompt tensor for task instructions (e.g. NER span tagging).
    - P_lang: Learnable prompt tensor for language/script characteristics (e.g. Bhojpuri, Maithili, Santali).
    
    Optional MLP projection head reparameterizes prompts during early training iterations for stability.
    """

    def __init__(
        self,
        task_prompt_len: int = 10,
        lang_prompt_len: int = 10,
        hidden_size: int = 768,
        use_mlp_projection: bool = True,
        mid_dim: int = 512,
        init_range: float = 0.02
    ):
        """
        Args:
            task_prompt_len: Number of virtual tokens for P_task.
            lang_prompt_len: Number of virtual tokens for P_lang.
            hidden_size: Hidden embedding dimension of the transformer backbone.
            use_mlp_projection: Whether to use an MLP bottleneck projector during tuning.
            mid_dim: Bottleneck dimension for MLP projection.
            init_range: Uniform initialization range for prompt embeddings.
        """
        super().__init__()
        self.task_prompt_len = task_prompt_len
        self.lang_prompt_len = lang_prompt_len
        self.hidden_size = hidden_size
        self.total_prompt_len = task_prompt_len + lang_prompt_len
        self.use_mlp_projection = use_mlp_projection

        if self.use_mlp_projection:
            # Low-rank embeddings with MLP projection head
            self.task_raw_prompt = nn.Parameter(
                torch.empty(task_prompt_len, mid_dim).uniform_(-init_range, init_range)
            )
            self.lang_raw_prompt = nn.Parameter(
                torch.empty(lang_prompt_len, mid_dim).uniform_(-init_range, init_range)
            )
            self.mlp_projector = nn.Sequential(
                nn.Linear(mid_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size)
            )
        else:
            # Direct soft prompt parameters
            self.task_prompt = nn.Parameter(
                torch.empty(task_prompt_len, hidden_size).uniform_(-init_range, init_range)
            )
            self.lang_prompt = nn.Parameter(
                torch.empty(lang_prompt_len, hidden_size).uniform_(-init_range, init_range)
            )

    def get_prompt_embeddings(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Computes and concatenates P_task and P_lang virtual embeddings.

        Returns:
            Tensor of shape (batch_size, task_prompt_len + lang_prompt_len, hidden_size)
        """
        if self.use_mlp_projection:
            task_embeds = self.mlp_projector(self.task_raw_prompt)
            lang_embeds = self.mlp_projector(self.lang_raw_prompt)
        else:
            task_embeds = self.task_prompt
            lang_embeds = self.lang_prompt

        # Concatenate P_task and P_lang along prompt length dimension
        combined_prompt = torch.cat([task_embeds, lang_embeds], dim=0)  # (total_prompt_len, hidden_size)

        # Expand across batch dimension
        return combined_prompt.unsqueeze(0).expand(batch_size, -1, -1).to(device)

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """
        Prepends soft prompt embeddings to input word embeddings.

        Args:
            input_embeds: Word embeddings tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Combined embeddings tensor of shape (batch_size, total_prompt_len + seq_len, hidden_size).
        """
        batch_size = input_embeds.shape[0]
        prompt_embeds = self.get_prompt_embeddings(batch_size, input_embeds.device)
        return torch.cat([prompt_embeds, input_embeds], dim=1)

    def freeze_task_prompt(self):
        """Freeze P_task parameters."""
        if self.use_mlp_projection:
            self.task_raw_prompt.requires_grad = False
        else:
            self.task_prompt.requires_grad = False

    def unfreeze_task_prompt(self):
        """Unfreeze P_task parameters."""
        if self.use_mlp_projection:
            self.task_raw_prompt.requires_grad = True
        else:
            self.task_prompt.requires_grad = True

    def freeze_lang_prompt(self):
        """Freeze P_lang parameters."""
        if self.use_mlp_projection:
            self.lang_raw_prompt.requires_grad = False
        else:
            self.lang_prompt.requires_grad = False

    def unfreeze_lang_prompt(self):
        """Unfreeze P_lang parameters."""
        if self.use_mlp_projection:
            self.lang_raw_prompt.requires_grad = True
        else:
            self.lang_prompt.requires_grad = True
