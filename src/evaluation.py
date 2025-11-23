"""Evaluation metrics for NER extraction."""

import json
from typing import Dict, List

import pandas as pd


class NEREvaluator:
    """Evaluate NER extraction results."""

    def __init__(self, entity_types: List[str] = None):
        """
        Initialize evaluator.

        Args:
            entity_types: List of entity types to evaluate
        """
        self.entity_types = entity_types or ["person", "organizations", "address"]

    def exact_match_accuracy(self, predictions: List[Dict], ground_truth: List[Dict]) -> float:
        """
        Calculate exact match accuracy.

        Args:
            predictions: List of predicted entities
            ground_truth: List of ground truth entities

        Returns:
            Accuracy score (0-1)
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")

        correct = 0
        for pred, gt in zip(predictions, ground_truth):
            if self._exact_match(pred, gt):
                correct += 1

        return correct / len(predictions)

    def _exact_match(self, pred: Dict, gt: Dict) -> bool:
        """Check if prediction exactly matches ground truth."""
        for entity_type in self.entity_types:
            pred_entities = sorted(pred.get(entity_type, []))
            gt_entities = sorted(gt.get(entity_type, []))

            if pred_entities != gt_entities:
                return False

        return True

    def partial_match_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score with partial matching.

        Args:
            predictions: List of predicted entities
            ground_truth: List of ground truth entities

        Returns:
            Dictionary with precision, recall, and F1 scores per entity type
        """
        metrics = {}

        for entity_type in self.entity_types:
            tp = 0  # True positives
            fp = 0  # False positives
            fn = 0  # False negatives

            for pred, gt in zip(predictions, ground_truth):
                pred_entities = set(pred.get(entity_type, []))
                gt_entities = set(gt.get(entity_type, []))

                tp += len(pred_entities & gt_entities)
                fp += len(pred_entities - gt_entities)
                fn += len(gt_entities - pred_entities)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            metrics[entity_type] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

        # Calculate macro averages
        avg_precision = sum(m["precision"] for m in metrics.values()) / len(metrics)
        avg_recall = sum(m["recall"] for m in metrics.values()) / len(metrics)
        avg_f1 = sum(m["f1"] for m in metrics.values()) / len(metrics)

        metrics["macro_avg"] = {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
        }

        return metrics

    def evaluate_all(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """
        Evaluate all metrics.

        Args:
            predictions: List of predicted entities
            ground_truth: List of ground truth entities

        Returns:
            Dictionary with all evaluation metrics
        """
        exact_match = self.exact_match_accuracy(predictions, ground_truth)
        partial_metrics = self.partial_match_metrics(predictions, ground_truth)

        return {
            "exact_match_accuracy": exact_match,
            "partial_match_metrics": partial_metrics,
        }

    @staticmethod
    def save_results(results: Dict, output_path: str):
        """Save evaluation results to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    @staticmethod
    def print_results(results: Dict):
        """Print evaluation results in a readable format."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        print(f"\nExact Match Accuracy: {results['exact_match_accuracy']:.4f}")

        print("\n" + "-" * 60)
        print("Partial Match Metrics (Per Entity Type)")
        print("-" * 60)

        partial_metrics = results["partial_match_metrics"]

        for entity_type in ["person", "organizations", "address"]:
            if entity_type in partial_metrics:
                metrics = partial_metrics[entity_type]
                print(f"\n{entity_type.upper()}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1 Score:  {metrics['f1']:.4f}")
                print(f"  TP/FP/FN:  {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")

        if "macro_avg" in partial_metrics:
            print(f"\nMACRO AVERAGE:")
            print(f"  Precision: {partial_metrics['macro_avg']['precision']:.4f}")
            print(f"  Recall:    {partial_metrics['macro_avg']['recall']:.4f}")
            print(f"  F1 Score:  {partial_metrics['macro_avg']['f1']:.4f}")

        print("\n" + "=" * 60 + "\n")


def compare_methods(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare results from multiple methods.

    Args:
        results_dict: Dictionary mapping method names to their results

    Returns:
        DataFrame with comparison
    """
    comparison_data = []

    for method_name, results in results_dict.items():
        row = {
            "Method": method_name,
            "Exact Match Accuracy": results["exact_match_accuracy"],
        }

        partial_metrics = results["partial_match_metrics"]

        # Add per-entity metrics
        for entity_type in ["person", "organizations", "address"]:
            if entity_type in partial_metrics:
                metrics = partial_metrics[entity_type]
                row[f"{entity_type}_f1"] = metrics["f1"]
                row[f"{entity_type}_precision"] = metrics["precision"]
                row[f"{entity_type}_recall"] = metrics["recall"]

        # Add macro averages
        if "macro_avg" in partial_metrics:
            row["macro_f1"] = partial_metrics["macro_avg"]["f1"]
            row["macro_precision"] = partial_metrics["macro_avg"]["precision"]
            row["macro_recall"] = partial_metrics["macro_avg"]["recall"]

        comparison_data.append(row)

    return pd.DataFrame(comparison_data)
