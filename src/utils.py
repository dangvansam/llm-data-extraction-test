import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from loguru import logger

from src.config import RESULTS_DIR


def calculate_metrics(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """
    Calculate precision, recall, F1 score, and accuracy.

    Args:
        predictions: List of predicted entities
        ground_truth: List of ground truth entities

    Returns:
        Dictionary with metrics for each entity type and overall
    """
    metrics = {"person": {}, "organizations": {}, "address": {}, "overall": {}}

    for entity_type in ["person", "organizations", "address"]:
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives
        tn = 0  # True negatives (for accuracy calculation)

        for pred, truth in zip(predictions, ground_truth):
            pred_entities = set(pred.get(entity_type, []))
            truth_entities = set(truth.get(entity_type, []))

            # Calculate TP, FP, FN
            tp += len(pred_entities & truth_entities)
            fp += len(pred_entities - truth_entities)
            fn += len(truth_entities - pred_entities)

            # Calculate TN: both predicted and actual have no entities for this type
            if len(pred_entities) == 0 and len(truth_entities) == 0:
                tn += 1

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        metrics[entity_type] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    # Overall metrics
    total_tp = sum(metrics[et]["tp"] for et in ["person", "organizations", "address"])
    total_fp = sum(metrics[et]["fp"] for et in ["person", "organizations", "address"])
    total_fn = sum(metrics[et]["fn"] for et in ["person", "organizations", "address"])
    total_tn = sum(metrics[et]["tn"] for et in ["person", "organizations", "address"])

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0
    )
    overall_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0

    metrics["overall"] = {
        "precision": round(overall_precision, 4),
        "recall": round(overall_recall, 4),
        "f1": round(overall_f1, 4),
        "accuracy": round(overall_accuracy, 4),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
    }

    return metrics


def display_metrics(metrics: Dict, title: str = "EVALUATION METRICS"):
    """Display metrics in a formatted way."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print("\nOVERALL METRICS:")
    print(f"  Precision: {metrics['overall']['precision']:.4f}")
    print(f"  Recall   : {metrics['overall']['recall']:.4f}")
    print(f"  F1 Score : {metrics['overall']['f1']:.4f}")
    print(f"  Accuracy : {metrics['overall']['accuracy']:.4f}")
    print(f"  TP: {metrics['overall']['tp']}, FP: {metrics['overall']['fp']}, FN: {metrics['overall']['fn']}, TN: {metrics['overall']['tn']}")
    print("\nPER-ENTITY METRICS:")
    for entity_type in ["person", "organizations", "address"]:
        m = metrics[entity_type]
        print(f"\n  {entity_type.upper()}:")
        print(f"    Precision: {m['precision']:.4f}")
        print(f"    Recall   : {m['recall']:.4f}")
        print(f"    F1 Score : {m['f1']:.4f}")
        print(f"    Accuracy : {m['accuracy']:.4f}")
        print(f"    TP: {m['tp']}, FP: {m['fp']}, FN: {m['fn']}, TN: {m['tn']}")
    print("=" * 80)


def save_experiment_results(
    experiment_name: str,
    config: Dict,
    metrics: Dict,
    performance: Dict,
    predictions: List[Dict] = None,
    texts: List[str] = None,
    ground_truth: List[Dict] = None
):
    """Save experiment results to disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = RESULTS_DIR / "prompt_pipeline_experiments" / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "config": config,
        "metrics": metrics,
        "performance": performance,
    }

    summary_file = exp_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.success(f"Summary saved to {summary_file}")

    if predictions and texts and ground_truth:
        details = {
            "predictions": predictions,
            "ground_truth": ground_truth,
            "texts": texts
        }
        details_file = exp_dir / "detailed_results.json"
        with open(details_file, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)
        logger.success(f"Detailed results saved to {details_file}")

    return exp_dir


def compare_experiments(exp_dirs: List[Path]):
    """Compare multiple experiments."""
    comparisons = []

    for exp_dir in exp_dirs:
        summary_file = exp_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file, "r") as f:
                summary = json.load(f)
                comparisons.append({
                    "experiment": summary["experiment_name"],
                    "timestamp": summary["timestamp"],
                    "f1_overall": summary["metrics"]["overall"]["f1"],
                    "precision": summary["metrics"]["overall"]["precision"],
                    "recall": summary["metrics"]["overall"]["recall"],
                    "throughput": summary["performance"]["throughput"],
                    "avg_time": summary["performance"]["avg_time_per_sample"],
                })

    if comparisons:
        df = pd.DataFrame(comparisons)
        df = df.sort_values("f1_overall", ascending=False)
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPARISON")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
        return df
    else:
        print("No experiments found for comparison")
        return None


def parse_response(response: str, entity_types: List[str]) -> Dict[str, List[str]]:
    """
    Parse JSON response from model.

    Args:
        response: Model output
        entity_types: List of entity types to extract

    Returns:
        Parsed entities dictionary
    """
    entities = {entity_type: [] for entity_type in entity_types}

    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())

            for key in entity_types:
                if key in parsed:
                    value = parsed[key]
                    if isinstance(value, list):
                        entities[key] = value
                    elif isinstance(value, str):
                        entities[key] = [value] if value else []
        else:
            logger.info(f"No JSON found in response: {response[:100]}")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")

    return entities
