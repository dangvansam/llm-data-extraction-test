"""Benchmarking utilities for comparing NER extraction methods."""

import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tabulate import tabulate

from .config import BaseNERConfig
from .evaluation import NEREvaluator, compare_methods


class NERBenchmark:
    """Benchmark multiple NER extraction methods."""

    def __init__(self, config: BaseNERConfig = None):
        """
        Initialize benchmark.

        Args:
            config: Configuration object (any NER config class)
        """
        self.config = config or BaseNERConfig()
        self.evaluator = NEREvaluator(entity_types=self.config.entity_types)
        self.results = {}

    def run_benchmark(
        self, method_name: str, extractor, test_dataset: List[Dict], verbose: bool = True
    ) -> Dict:
        """
        Run benchmark for a single method.

        Args:
            method_name: Name of the method
            extractor: Extractor instance with evaluate_on_dataset method
            test_dataset: Test dataset
            verbose: Whether to print results

        Returns:
            Benchmark results
        """
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Running benchmark: {method_name}")
            print(f"{'=' * 70}\n")

        # Measure inference time
        start_time = time.time()
        predictions, ground_truth = extractor.evaluate_on_dataset(test_dataset)
        inference_time = time.time() - start_time

        # Evaluate
        eval_results = self.evaluator.evaluate_all(predictions, ground_truth)

        # Add timing information
        eval_results["inference_time"] = inference_time
        eval_results["samples_per_second"] = len(test_dataset) / inference_time
        eval_results["method_name"] = method_name

        # Store results
        self.results[method_name] = eval_results

        if verbose:
            self._print_method_results(method_name, eval_results)

        return eval_results

    def _print_method_results(self, method_name: str, results: Dict):
        """Print results for a single method."""
        print(f"\nResults for {method_name}:")
        print("-" * 70)
        print(f"Exact Match Accuracy: {results['exact_match_accuracy']:.4f}")
        print(f"Inference Time: {results['inference_time']:.2f}s")
        print(f"Samples/Second: {results['samples_per_second']:.2f}")

        partial_metrics = results["partial_match_metrics"]

        print("\nPer-Entity Metrics:")
        entity_data = []
        for entity_type in ["person", "organizations", "address"]:
            if entity_type in partial_metrics:
                metrics = partial_metrics[entity_type]
                entity_data.append(
                    [
                        entity_type,
                        f"{metrics['precision']:.4f}",
                        f"{metrics['recall']:.4f}",
                        f"{metrics['f1']:.4f}",
                    ]
                )

        print(tabulate(entity_data, headers=["Entity Type", "Precision", "Recall", "F1"], tablefmt="grid"))

        if "macro_avg" in partial_metrics:
            print(f"\nMacro Average F1: {partial_metrics['macro_avg']['f1']:.4f}")

    def compare_all_methods(self) -> pd.DataFrame:
        """
        Compare all benchmarked methods.

        Returns:
            DataFrame with comparison
        """
        if not self.results:
            raise ValueError("No benchmark results available. Run benchmarks first.")

        comparison_df = compare_methods(self.results)

        # Add performance metrics
        perf_data = []
        for method_name, results in self.results.items():
            perf_data.append(
                {
                    "Method": method_name,
                    "Inference Time (s)": results["inference_time"],
                    "Samples/Second": results["samples_per_second"],
                }
            )

        perf_df = pd.DataFrame(perf_data)

        # Merge dataframes
        final_df = pd.merge(comparison_df, perf_df, on="Method")

        return final_df

    def print_comparison(self):
        """Print comparison table."""
        df = self.compare_all_methods()

        print("\n" + "=" * 100)
        print("BENCHMARK COMPARISON")
        print("=" * 100 + "\n")

        # Select key columns for display
        display_cols = [
            "Method",
            "Exact Match Accuracy",
            "macro_f1",
            "macro_precision",
            "macro_recall",
            "Inference Time (s)",
            "Samples/Second",
        ]

        display_df = df[display_cols].copy()
        display_df.columns = ["Method", "Exact Match", "F1", "Precision", "Recall", "Time (s)", "Samples/s"]

        print(tabulate(display_df, headers="keys", tablefmt="grid", showindex=False, floatfmt=".4f"))

        # Find best method
        best_accuracy = display_df.loc[display_df["Exact Match"].idxmax()]
        best_f1 = display_df.loc[display_df["F1"].idxmax()]
        fastest = display_df.loc[display_df["Samples/s"].idxmax()]

        print("\n" + "-" * 100)
        print("SUMMARY:")
        print(f"  Best Exact Match Accuracy: {best_accuracy['Method']} ({best_accuracy['Exact Match']:.4f})")
        print(f"  Best F1 Score: {best_f1['Method']} ({best_f1['F1']:.4f})")
        print(f"  Fastest: {fastest['Method']} ({fastest['Samples/s']:.2f} samples/s)")
        print("-" * 100 + "\n")

    def save_results(self, output_dir: Path):
        """
        Save benchmark results.

        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual method results
        for method_name, results in self.results.items():
            output_path = output_dir / f"{method_name}_results.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        # Save comparison table
        comparison_df = self.compare_all_methods()
        comparison_df.to_csv(output_dir / "comparison.csv", index=False)

        print(f"\nResults saved to {output_dir}")

    def save_predictions(self, method_name: str, predictions: List[Dict], output_path: Path):
        """
        Save predictions to file.

        Args:
            method_name: Name of the method
            predictions: List of predictions
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {"method": method_name, "predictions": predictions, "timestamp": time.time()}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Predictions saved to {output_path}")


def run_full_benchmark(
    prompt_extractor=None,
    rag_extractor=None,
    finetuned_extractor=None,
    test_dataset: List[Dict] = None,
    output_dir: Path = None,
) -> pd.DataFrame:
    """
    Run full benchmark comparing all three methods.

    Args:
        prompt_extractor: Prompt engineering extractor
        rag_extractor: RAG extractor
        finetuned_extractor: Fine-tuned extractor
        test_dataset: Test dataset
        output_dir: Directory to save results

    Returns:
        Comparison DataFrame
    """
    benchmark = NERBenchmark()

    if prompt_extractor:
        benchmark.run_benchmark("Prompt Engineering", prompt_extractor, test_dataset)

    if rag_extractor:
        benchmark.run_benchmark("RAG", rag_extractor, test_dataset)

    if finetuned_extractor:
        benchmark.run_benchmark("Fine-tuning", finetuned_extractor, test_dataset)

    # Print comparison
    benchmark.print_comparison()

    # Save results
    if output_dir:
        benchmark.save_results(output_dir)

    return benchmark.compare_all_methods()
