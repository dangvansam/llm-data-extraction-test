import json
import sys
import time

from loguru import logger
from tqdm import tqdm

sys.path.append(".")

from src.config import (
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    ExtractionMode,
    NERPromptEngineeringConfig,
)
from src.data_processor import DataProcessor
from src.pipeline.prompt_engineering import PromptNERExtractor
from src.utils import calculate_metrics

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Prompt Engineering NER Evaluation - RAW Chat Mode")
    logger.info("=" * 80)

    test_dataset_path = PROCESSED_DATA_DIR / "test.json"
    output_dir = RESULTS_DIR / "prompt_engineering_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not test_dataset_path.exists():
        logger.error(f"Test dataset not found: {test_dataset_path}")
        raise ValueError("Please run data preparation script first")

    # Load test dataset
    logger.info(f"Loading test dataset from {test_dataset_path}")
    test_dataset = DataProcessor.load_dataset(test_dataset_path)
    logger.info(f"Loaded {len(test_dataset)} test samples")

    # Configuration: RAW mode with chat format
    # config = NERPromptEngineeringConfig(
    #     model_name="Qwen/Qwen3-4B",
    #     extraction_mode=ExtractionMode.RAW,
    #     enable_thinking=False
    # )

    # config = NERPromptEngineeringConfig(
    #     model_name="Qwen/Qwen3-4B",
    #     extraction_mode=ExtractionMode.RAW,
    #     enable_thinking=True
    # )

    # EXPERIMENT_NAME = "qwen3_4b_structured_no_thinking"

    # config = NERPromptEngineeringConfig(
    #     model_name="Qwen/Qwen3-4B",
    #     extraction_mode=ExtractionMode.STRUCTURED_OUTPUT,
    #     enable_thinking=False
    # )

    # EXPERIMENT_NAME = "qwen3_4b_structured_thinking"

    config = NERPromptEngineeringConfig(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        extraction_mode=ExtractionMode.STRUCTURED_OUTPUT,
        enable_thinking=True
    )

    logger.info(f"Model: {config.model_name}")
    logger.info(f"Extraction mode: {config.extraction_mode.value}")
    logger.info(f"Add schema: {config.add_schema}")
    logger.info("")

    # Initialize extractor
    extractor = PromptNERExtractor(config)

    # Prepare data
    texts = [sample["text"] for sample in test_dataset]
    labels = [sample["entities"] for sample in test_dataset]

    # Run extraction
    logger.info("Starting extraction...")
    start_time = time.time()
    predictions = []

    for text, label in tqdm(zip(texts, labels), desc="Extracting entities", total=len(texts)):
        prediction = extractor.extract_entities(text)
        predictions.append(prediction)
        logger.debug(f"Ground truth: {label}")
        logger.debug(f"Prediction  : {prediction}")

    elapsed_time = time.time() - start_time

    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = calculate_metrics(predictions, labels)

    # Performance stats
    throughput = len(test_dataset) / elapsed_time if elapsed_time > 0 else 0
    avg_time = elapsed_time / len(test_dataset)

    # Results
    result = {
        "method": "Prompt Engineering - RAW Chat Mode",
        "config": {
            "model_name": config.model_name,
            "extraction_mode": config.extraction_mode.value,
            "add_schema": config.add_schema,
            "temperature": config.temperature,
            "max_new_tokens": config.max_new_tokens,
        },
        "performance": {
            "total_samples": len(test_dataset),
            "elapsed_time": round(elapsed_time, 2),
            "throughput": round(throughput, 2),
            "avg_time_per_sample": round(avg_time, 2),
        },
        "metrics": metrics,
    }

    # Print results
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total samples: {len(test_dataset)}")
    logger.info(f"Total time: {elapsed_time:.2f}s")
    logger.info(f"Throughput: {throughput:.2f} samples/s")
    logger.info(f"Avg time per sample: {avg_time:.2f}s")
    logger.info("")
    logger.info("OVERALL METRICS:")
    logger.info(f"  Precision: {metrics['overall']['precision']:.4f}")
    logger.info(f"  Recall   : {metrics['overall']['recall']:.4f}")
    logger.info(f"  F1 Score : {metrics['overall']['f1']:.4f}")
    logger.info("")
    logger.info("PER-ENTITY METRICS:")
    for entity_type in ["person", "organizations", "address"]:
        m = metrics[entity_type]
        logger.info(f"  {entity_type.upper()}:")
        logger.info(f"    Precision: {m['precision']:.4f}")
        logger.info(f"    Recall   : {m['recall']:.4f}")
        logger.info(f"    F1 Score : {m['f1']:.4f}")
    logger.info("=" * 80)

    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.success(f"Results saved to {results_file}")

    extractor.cleanup()
    logger.success("Evaluation complete!")