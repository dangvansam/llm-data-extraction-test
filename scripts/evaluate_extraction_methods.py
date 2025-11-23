import json
import sys
import time
from pathlib import Path
from typing import Dict, List

from loguru import logger

sys.path.append(".")

from src.config import (
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    ExtractionMode,
    NERPromptEngineeringConfig,
)
from src.prompt_engineering import PromptNERExtractor
from src.services.data_processor import DataProcessorService


class NERMethodEvaluator:
    """Evaluator for comparing different NER extraction methods."""

    def __init__(self, test_dataset_path: Path):
        """
        Initialize evaluator.

        Args:
            test_dataset_path: Path to test dataset JSON file
        """
        self.test_dataset_path = test_dataset_path
        self.test_dataset = None
        self.results = {}

    def load_test_dataset(self):
        """Load test dataset from file."""
        logger.info(f"Loading test dataset from {self.test_dataset_path}")
        self.test_dataset = DataProcessorService.load_dataset(self.test_dataset_path)
        logger.info(f"Loaded {len(self.test_dataset)} test samples")

    def calculate_metrics(
        self, predictions: List[Dict], ground_truth: List[Dict]
    ) -> Dict:
        """
        Calculate precision, recall, and F1 score.

        Args:
            predictions: List of predicted entities
            ground_truth: List of ground truth entities

        Returns:
            Dictionary with metrics
        """
        metrics = {"person": {}, "organizations": {}, "address": {}, "overall": {}}

        # Calculate per entity type
        for entity_type in ["person", "organizations", "address"]:
            tp = 0  # True positives
            fp = 0  # False positives
            fn = 0  # False negatives

            for pred, truth in zip(predictions, ground_truth):
                pred_entities = set(pred.get(entity_type, []))
                truth_entities = set(truth.get(entity_type, []))

                tp += len(pred_entities & truth_entities)
                fp += len(pred_entities - truth_entities)
                fn += len(truth_entities - pred_entities)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            metrics[entity_type] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

        # Calculate overall metrics
        total_tp = sum(metrics[et]["tp"] for et in ["person", "organizations", "address"])
        total_fp = sum(metrics[et]["fp"] for et in ["person", "organizations", "address"])
        total_fn = sum(metrics[et]["fn"] for et in ["person", "organizations", "address"])

        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = (
            2 * overall_precision * overall_recall / (overall_precision + overall_recall)
            if (overall_precision + overall_recall) > 0
            else 0
        )

        metrics["overall"] = {
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
            "f1": round(overall_f1, 4),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        }

        return metrics

    def evaluate_method(
        self,
        method_name: str,
        config: NERPromptEngineeringConfig,
        sample_limit: int = None,
    ) -> Dict:
        """
        Evaluate a single extraction method.

        Args:
            method_name: Name of the method (for logging)
            config: Configuration for the method
            sample_limit: Optional limit on number of samples to evaluate

        Returns:
            Dictionary with evaluation results
        """
        logger.info("=" * 60)
        logger.info(f"Evaluating: {method_name}")
        logger.info("=" * 60)

        # Create extractor
        try:
            extractor = PromptNERExtractor(config)
        except ImportError as e:
            logger.error(f"Failed to initialize {method_name}: {e}")
            return {
                "method": method_name,
                "status": "failed",
                "error": str(e),
            }

        # Prepare test data
        test_samples = self.test_dataset[:sample_limit] if sample_limit else self.test_dataset
        texts = [sample["text"] for sample in test_samples]
        ground_truth = [sample["entities"] for sample in test_samples]

        logger.info(f"Evaluating on {len(test_samples)} samples")

        # Run extraction with timing
        start_time = time.time()
        try:
            predictions = extractor.batch_extract(texts, show_progress=True)
        except Exception as e:
            logger.error(f"Extraction failed for {method_name}: {e}")
            return {
                "method": method_name,
                "status": "failed",
                "error": str(e),
            }
        elapsed_time = time.time() - start_time

        # Calculate metrics
        metrics = self.calculate_metrics(predictions, ground_truth)

        # Calculate throughput
        throughput = len(test_samples) / elapsed_time if elapsed_time > 0 else 0

        result = {
            "method": method_name,
            "status": "success",
            "config": {
                "model_name": config.model_name,
                "extraction_mode": config.extraction_mode.value,
                "use_chat_format": config.use_chat_format,
                "add_schema": config.add_schema,
                "temperature": config.temperature,
                "max_new_tokens": config.max_new_tokens,
            },
            "performance": {
                "total_samples": len(test_samples),
                "elapsed_time": round(elapsed_time, 2),
                "throughput": round(throughput, 2),
                "avg_time_per_sample": round(elapsed_time / len(test_samples), 2),
            },
            "metrics": metrics,
        }

        logger.info(f"Overall F1: {metrics['overall']['f1']:.4f}")
        logger.info(f"Time: {elapsed_time:.2f}s ({throughput:.2f} samples/s)")
        logger.info("")

        return result

    def run_evaluation(
        self,
        methods_to_test: List[str] = None,
        sample_limit: int = None,
    ):
        """
        Run evaluation for all specified methods.

        Args:
            methods_to_test: List of methods to test (RAW, STRUCTURED_OUTPUT, OUTLINES)
                           If None, tests all available methods
            sample_limit: Optional limit on number of samples
        """
        if methods_to_test is None:
            methods_to_test = ["RAW", "STRUCTURED_OUTPUT", "OUTLINES"]

        self.load_test_dataset()

        # Define configurations for each method
        configs = {
            "RAW": NERPromptEngineeringConfig(
                extraction_mode=ExtractionMode.RAW,
                use_chat_format=False,
                add_schema=False,
            ),
            "RAW_WITH_SCHEMA": NERPromptEngineeringConfig(
                extraction_mode=ExtractionMode.RAW,
                use_chat_format=False,
                add_schema=True,
            ),
            "RAW_CHAT_FORMAT": NERPromptEngineeringConfig(
                extraction_mode=ExtractionMode.RAW,
                use_chat_format=True,
                add_schema=False
            ),
            "STRUCTURED_OUTPUT": NERPromptEngineeringConfig(
                extraction_mode=ExtractionMode.STRUCTURED_OUTPUT,
                use_chat_format=True,
                add_schema=True
            ),
            "OUTLINES": NERPromptEngineeringConfig(
                extraction_mode=ExtractionMode.OUTLINES,
                use_chat_format=False,
                add_schema=False
            ),
        }

        for method in methods_to_test:
            if method not in configs:
                logger.warning(f"Unknown method: {method}, skipping")
                continue

            result = self.evaluate_method(method, configs[method], sample_limit)
            self.results[method] = result

    def generate_comparison_report(self) -> str:
        """
        Generate comparison report.

        Returns:
            Formatted report string
        """
        if not self.results:
            return "No results to report"

        report = []
        report.append("=" * 80)
        report.append("NER EXTRACTION METHODS - EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Performance comparison
        report.append("PERFORMANCE COMPARISON")
        report.append("-" * 80)
        report.append(
            f"{'Method':<25} {'Samples':<10} {'Time(s)':<10} {'Throughput':<15} {'Avg/Sample':<12}"
        )
        report.append("-" * 80)

        for method, result in self.results.items():
            if result["status"] == "success":
                perf = result["performance"]
                report.append(
                    f"{method:<25} {perf['total_samples']:<10} {perf['elapsed_time']:<10.2f} "
                    f"{perf['throughput']:<15.2f} {perf['avg_time_per_sample']:<12.2f}"
                )
            else:
                report.append(f"{method:<25} FAILED: {result.get('error', 'Unknown error')}")

        report.append("")

        # Metrics comparison
        report.append("METRICS COMPARISON (OVERALL)")
        report.append("-" * 80)
        report.append(
            f"{'Method':<25} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'TP':<8} {'FP':<8} {'FN':<8}"
        )
        report.append("-" * 80)

        for method, result in self.results.items():
            if result["status"] == "success":
                overall = result["metrics"]["overall"]
                report.append(
                    f"{method:<25} {overall['precision']:<12.4f} {overall['recall']:<12.4f} "
                    f"{overall['f1']:<12.4f} {overall['tp']:<8} {overall['fp']:<8} {overall['fn']:<8}"
                )

        report.append("")

        # Per-entity type metrics
        for entity_type in ["person", "organizations", "address"]:
            report.append(f"METRICS BY ENTITY TYPE: {entity_type.upper()}")
            report.append("-" * 80)
            report.append(
                f"{'Method':<25} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}"
            )
            report.append("-" * 80)

            for method, result in self.results.items():
                if result["status"] == "success":
                    metrics = result["metrics"][entity_type]
                    report.append(
                        f"{method:<25} {metrics['precision']:<12.4f} "
                        f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}"
                    )

            report.append("")

        # Best method summary
        report.append("SUMMARY")
        report.append("-" * 80)

        # Find best by F1 score
        successful_results = {
            k: v for k, v in self.results.items() if v["status"] == "success"
        }
        if successful_results:
            best_f1_method = max(
                successful_results.items(),
                key=lambda x: x[1]["metrics"]["overall"]["f1"],
            )
            best_speed_method = max(
                successful_results.items(),
                key=lambda x: x[1]["performance"]["throughput"],
            )

            report.append(
                f"Best F1 Score: {best_f1_method[0]} "
                f"(F1={best_f1_method[1]['metrics']['overall']['f1']:.4f})"
            )
            report.append(
                f"Fastest: {best_speed_method[0]} "
                f"({best_speed_method[1]['performance']['throughput']:.2f} samples/s)"
            )

        report.append("=" * 80)

        return "\n".join(report)

    def save_results(self, output_dir: Path):
        """
        Save evaluation results to files.

        Args:
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results as JSON
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved detailed results to {results_file}")

        # Save comparison report as text
        report = self.generate_comparison_report()
        report_file = output_dir / "evaluation_report.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Saved comparison report to {report_file}")

        # Print report to console
        logger.info("\n" + report)


def main():
    """Main evaluation function."""
    logger.info("NER Extraction Methods Evaluation")
    logger.info("=" * 80)

    # Configuration
    test_dataset_path = PROCESSED_DATA_DIR / "test" / "test.json"
    output_dir = RESULTS_DIR / "method_comparison"

    # Check if test dataset exists
    if not test_dataset_path.exists():
        logger.error(f"Test dataset not found: {test_dataset_path}")
        logger.info("Please run data generation script first")
        return

    # Initialize evaluator
    evaluator = NERMethodEvaluator(test_dataset_path)

    # Define which methods to test
    # Comment out methods you don't want to test or don't have dependencies for
    methods_to_test = [
        "RAW",
        "RAW_WITH_SCHEMA",
        "RAW_CHAT_FORMAT",
        "STRUCTURED_OUTPUT",
        "OUTLINES",
    ]

    # Optional: limit number of samples for faster testing
    sample_limit = 10  # Set to a number (e.g., 10) for quick testing

    logger.info(f"Methods to evaluate: {', '.join(methods_to_test)}")
    if sample_limit:
        logger.info(f"Sample limit: {sample_limit}")
    logger.info("")

    # Run evaluation
    evaluator.run_evaluation(
        methods_to_test=methods_to_test,
        sample_limit=sample_limit,
    )

    # Save and display results
    evaluator.save_results(output_dir)

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
