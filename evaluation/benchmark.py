"""
Comparative Benchmark Evaluation Runner.

Executes comparative training & evaluation across:
1. Full Fine-Tuning (Full model backpropagation)
2. LoRA (Low-Rank Adaptation)
3. Disentangled Soft-Prompt Tuning (P_task + P_lang)

Generates markdown and JSON benchmark reports.
"""

import sys
import os
import time
import json
import torch
import torch.nn as nn
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ner_model import DisentangledPromptNERModel
from models.baselines import FullFineTuningNERModel, LoRANERModel, BaselineComparator
from data.synthetic import SyntheticNERGenerator
from data.loader import create_dataloaders
from data.real_datasets import RealNERDatasetLoader
from evaluation.metrics import NEREvaluator



class MockTokenizer:
    """Mock tokenizer for offline/test environment benchmarking."""
    def __init__(self):
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        tokens = [hash(word) % 25000 + 1000 for word in text.split()]
        if not tokens:
            tokens = [1000]
        if add_special_tokens:
            tokens = [self.cls_token_id] + tokens + [self.sep_token_id]
        return tokens


def run_comparative_benchmark(
    num_samples: int = 50,
    epochs: int = 2,
    batch_size: int = 4,
    use_real_data: bool = True
) -> Dict[str, Any]:
    """
    Runs training and evaluation loops for all three paradigms and returns comparative stats.
    """
    print("==================================================")
    print(" STARTING COMPARATIVE MODEL BENCHMARK")
    print("==================================================")

    # 1. Load Data
    if use_real_data:
        print("Loading Real Indic Benchmark Corpora (Bhojpuri/Maithili/Santali)...")
        loader = RealNERDatasetLoader(preprocess=True)
        train_samples = loader.load_indic_sample_corpus("bhojpuri") + loader.load_indic_sample_corpus("maithili")
    else:
        print("Generating Synthetic Data...")
        generator = SyntheticNERGenerator()
        train_samples = generator.generate_dataset(num_samples=num_samples)

    tokenizer = MockTokenizer()
    train_loader, val_loader = create_dataloaders(
        train_samples, tokenizer, batch_size=batch_size, prompt_length=16
    )

    # 2. Instantiate Models
    models = {
        "Full Fine-Tuning": FullFineTuningNERModel(model_name_or_path="mock-backbone"),
        "LoRA (Low-Rank Adaptation)": LoRANERModel(model_name_or_path="mock-backbone"),
        "Disentangled Soft-Prompt (Ours)": DisentangledPromptNERModel(
            model_name_or_path="mock-backbone", task_prompt_len=8, lang_prompt_len=8
        )
    }

    evaluator = NEREvaluator()
    benchmark_results = {}

    for name, model in models.items():
        print(f"\nEvaluating Paradigm: {name}...")
        stats = BaselineComparator.get_parameter_stats(model)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )

        start_time = time.time()
        model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                if "Disentangled" in name:
                    model.configure_stage(2)

                loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        elapsed_time = round(time.time() - start_time, 3)

        # Validation Evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(logits, dim=-1)

                all_preds.append(preds)
                all_labels.append(labels)

        if all_preds:
            cat_preds = torch.cat(all_preds, dim=0)
            cat_labels = torch.cat(all_labels, dim=0)
            eval_metrics = evaluator.compute_metrics(cat_preds, cat_labels)
        else:
            eval_metrics = {"overall_f1": 0.0, "overall_precision": 0.0, "overall_recall": 0.0}

        benchmark_results[name] = {
            "trainable_parameters": stats["trainable_parameters"],
            "total_parameters": stats["total_parameters"],
            "trainable_pct": f"{stats['trainable_percentage']:.2f}%",
            "training_time_sec": elapsed_time,
            "overall_f1": round(eval_metrics.get("overall_f1", 0.0), 4),
            "overall_precision": round(eval_metrics.get("overall_precision", 0.0), 4),
            "overall_recall": round(eval_metrics.get("overall_recall", 0.0), 4)
        }

    # Print Summary Markdown Table
    print("\n" + "=" * 80)
    print(" COMPARATIVE BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(f"| {'Model Approach':<30} | {'Trainable Params':<18} | {'% Params':<10} | {'Time (s)':<10} | {'F1 Score':<8} |")
    print("|" + "-" * 32 + "|" + "-" * 20 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 10 + "|")
    for name, res in benchmark_results.items():
        print(f"| {name:<30} | {res['trainable_parameters']:<18,d} | {res['trainable_pct']:<10} | {res['training_time_sec']:<10.2f} | {res['overall_f1']:<8.4f} |")
    print("=" * 80)

    return benchmark_results


if __name__ == "__main__":
    run_comparative_benchmark()
