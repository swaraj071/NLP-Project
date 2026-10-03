"""
Sanity Check & Integration Unit Tests for Cross-Lingual Prompt-Tuned NER.

Verifies script normalizer, agglutinative postposition stripper, soft prompt tensor shapes,
model forward passes, loss computation, and span-level evaluation.
"""

import sys
import os
import torch

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


from data.normalizer import ScriptNormalizer, AgglutinativePostProcessor
from data.synthetic import SyntheticNERGenerator
from data.loader import NERDataset
from models.soft_prompt import DisentangledSoftPrompt
from models.ner_model import DisentangledPromptNERModel
from evaluation.metrics import NEREvaluator


class DummyTokenizer:
    """Mock tokenizer for offline unit test execution without internet dependency."""
    def __init__(self):
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.pad_token_id = 0
        self.bos_token_id = 101
        self.eos_token_id = 102

    def encode(self, text, add_special_tokens=True):
        # Maps string length to deterministic dummy integer subword IDs
        tokens = [hash(word) % 25000 + 1000 for word in text.split()]
        if not tokens:
            tokens = [1000]
        if add_special_tokens:
            tokens = [self.cls_token_id] + tokens + [self.sep_token_id]
        return tokens


def test_normalizer_and_postprocessor():
    print("Testing Script Normalizer & Agglutinative PostProcessor...")
    normalizer = ScriptNormalizer()

    # Test NFKC & Zero-width joiner cleanup
    raw_text = "पटना​में\u200d रहते हैं।"
    norm_text = normalizer.normalize(raw_text)
    assert "​" not in norm_text and "\u200d" not in norm_text, "Failed zero-width cleanup"
    print(f"  Normalized text: '{raw_text}' -> '{norm_text}'")

    # Test agglutinative postposition stripping
    postprocessor = AgglutinativePostProcessor()
    stem, pp = postprocessor.strip_postposition("पटनामें")
    assert stem == "पटना" and pp == "में", f"Failed postposition stripping: stem={stem}, pp={pp}"
    print(f"  Postposition Stripped: 'पटनामें' -> stem='{stem}', pp='{pp}'")

    tokens, tags = postprocessor.process_sentence_entities(["पटनामें", "रहते"], ["B-LOC", "O"])
    assert tokens == ["पटना", "में", "रहते"] and tags == ["B-LOC", "O", "O"], "Failed sentence entity postposition split"
    print("  Agglutinative sentence processing passed!")


def test_soft_prompt_injection():
    print("Testing Disentangled Soft Prompt Tensor Shapes...")
    task_len = 8
    lang_len = 6
    hidden_dim = 768

    soft_prompt = DisentangledSoftPrompt(
        task_prompt_len=task_len,
        lang_prompt_len=lang_len,
        hidden_size=hidden_dim,
        use_mlp_projection=True
    )

    batch_size = 4
    seq_len = 16
    dummy_input_embeds = torch.randn(batch_size, seq_len, hidden_dim)

    output_embeds = soft_prompt(dummy_input_embeds)
    expected_len = task_len + lang_len + seq_len

    assert output_embeds.shape == (batch_size, expected_len, hidden_dim), \
        f"Shape mismatch! Expected {(batch_size, expected_len, hidden_dim)}, got {output_embeds.shape}"
    print(f"  Soft Prompt Shape Verified: {output_embeds.shape} (Expected seq length {expected_len})")


def test_ner_model_forward():
    print("Testing DisentangledPromptNERModel Forward Pass & Loss...")
    model = DisentangledPromptNERModel(
        model_name_or_path="mock-backbone",
        num_labels=11,
        task_prompt_len=10,
        lang_prompt_len=10,
        use_mlp_projection=True
    )

    batch_size = 2
    seq_len = 12
    input_ids = torch.randint(100, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    labels = torch.randint(0, 10, (batch_size, seq_len))

    # Test stage 1 configuration
    model.configure_stage(1)
    loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    expected_len = 10 + 10 + seq_len
    assert logits.shape == (batch_size, expected_len, 11), f"Logits shape mismatch: {logits.shape}"
    assert loss is not None and not torch.isnan(loss), "Loss is None or NaN!"
    print(f"  Forward Pass Successful! Loss = {loss.item():.4f}, Logits shape = {logits.shape}")


def test_evaluator():
    print("Testing NEREvaluator Span Metric Calculation...")
    evaluator = NEREvaluator()

    preds = torch.tensor([[1, 2, 0, 5, 0], [0, 1, 2, 0, 0]])
    labels = torch.tensor([[1, 2, 0, 5, 0], [0, 1, 2, 0, 0]])

    metrics = evaluator.compute_metrics(preds, labels)
    assert "overall_f1" in metrics, "Missing overall_f1 key in evaluation metrics"
    print(f"  Span F1 Metric = {metrics['overall_f1']:.4f}")


from data.real_datasets import RealNERDatasetLoader
from models.baselines import FullFineTuningNERModel, LoRANERModel, BaselineComparator


def test_real_dataset_loader():
    print("Testing RealNERDatasetLoader Parsing & Normalization...")
    loader = RealNERDatasetLoader(preprocess=True)
    samples = loader.load_indic_sample_corpus("bhojpuri")
    assert len(samples) > 0, "Failed to load sample Indic corpus"
    assert "tokens" in samples[0] and "tags" in samples[0], "Sample missing tokens/tags"
    print(f"  Loaded {len(samples)} Indic samples! Sample 0 tokens: {samples[0]['tokens']}")


def test_baseline_models():
    print("Testing Baseline Models (Full Fine-Tuning & LoRA)...")
    full_model = FullFineTuningNERModel(model_name_or_path="mock-backbone")
    lora_model = LoRANERModel(model_name_or_path="mock-backbone")

    full_stats = BaselineComparator.get_parameter_stats(full_model)
    lora_stats = BaselineComparator.get_parameter_stats(lora_model)

    assert full_stats["trainable_percentage"] > 90.0, "Full fine-tuning should have high trainable %"
    assert lora_stats["trainable_percentage"] < 5.0, "LoRA should have low trainable %"
    print(f"  Full FT Trainable: {full_stats['trainable_percentage']}% | LoRA Trainable: {lora_stats['trainable_percentage']}%")


def run_all_sanity_checks():
    print("==================================================")
    print(" RUNNING SYSTEM SANITY CHECKS & UNIT TESTS")
    print("==================================================")
    test_normalizer_and_postprocessor()
    print("-" * 50)
    test_real_dataset_loader()
    print("-" * 50)
    test_soft_prompt_injection()
    print("-" * 50)
    test_ner_model_forward()
    print("-" * 50)
    test_baseline_models()
    print("-" * 50)
    test_evaluator()
    print("==================================================")
    print(" ALL SANITY CHECKS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_all_sanity_checks()
