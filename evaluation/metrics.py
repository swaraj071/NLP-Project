"""
Span-Level NER Metrics Evaluation Module.

Uses seqeval to compute entity span Precision, Recall, and F1 scores while correctly
filtering out subword padding tokens and virtual soft-prompt token slots (-100 indices).
"""

import torch
from typing import List, Dict, Any, Union
from data.synthetic import SyntheticNERGenerator

try:
    from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
    SEQEVAL_AVAILABLE = True
except ImportError:
    SEQEVAL_AVAILABLE = False


class NEREvaluator:
    """
    Evaluator for calculating span-level NER precision, recall, and F1.
    """

    def __init__(self, id2label: Dict[int, str] = SyntheticNERGenerator.ID2LABEL):
        """
        Args:
            id2label: Dictionary mapping integer class IDs to string IOB tags.
        """
        self.id2label = id2label

    def decode_predictions(
        self, preds: torch.Tensor, labels: torch.Tensor
    ) -> tuple[List[List[str]], List[List[str]]]:
        """
        Converts prediction IDs and label IDs to sequence lists of string tags,
        ignoring -100 index slots (prompt virtual tokens and subword padding).
        """
        preds_list = preds.tolist() if isinstance(preds, torch.Tensor) else preds
        labels_list = labels.tolist() if isinstance(labels, torch.Tensor) else labels

        true_predictions = []
        true_labels = []

        for pred_seq, label_seq in zip(preds_list, labels_list):
            seq_preds = []
            seq_labels = []

            for p_id, l_id in zip(pred_seq, label_seq):
                if l_id != -100:
                    seq_preds.append(self.id2label.get(p_id, "O"))
                    seq_labels.append(self.id2label.get(l_id, "O"))

            if seq_labels:
                true_predictions.append(seq_preds)
                true_labels.append(seq_labels)

        return true_predictions, true_labels

    def compute_metrics(
        self, preds: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Computes span-level seqeval metrics.

        Returns:
            Dictionary containing 'overall_precision', 'overall_recall', 'overall_f1', and 'report'.
        """
        true_preds, true_labels = self.decode_predictions(preds, labels)

        if not true_labels or not any(len(s) > 0 for s in true_labels):
            return {
                "overall_precision": 0.0,
                "overall_recall": 0.0,
                "overall_f1": 0.0,
                "report": "No valid labels evaluated."
            }

        if SEQEVAL_AVAILABLE:
            p = precision_score(true_labels, true_preds)
            r = recall_score(true_labels, true_preds)
            f1 = f1_score(true_labels, true_preds)
            report = classification_report(true_labels, true_preds)

            return {
                "overall_precision": float(p),
                "overall_recall": float(r),
                "overall_f1": float(f1),
                "report": report
            }
        else:
            # Fallback simple token-level accuracy and F1 if seqeval is absent
            correct = 0
            total = 0
            for pred_seq, label_seq in zip(true_preds, true_labels):
                for p_tag, l_tag in zip(pred_seq, label_seq):
                    if p_tag == l_tag:
                        correct += 1
                    total += 1
            accuracy = correct / max(total, 1)
            return {
                "overall_precision": float(accuracy),
                "overall_recall": float(accuracy),
                "overall_f1": float(accuracy),
                "report": f"Fallback token accuracy: {accuracy:.4f}"
            }
