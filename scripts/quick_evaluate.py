import sys

from loguru import logger

sys.path.append(".")

from src.config import PROCESSED_DATA_DIR, ExtractionMode, NERPromptEngineeringConfig
from src.prompt_engineering import PromptNERExtractor
from src.services.data_processor import DataProcessorService


def quick_evaluate(
    extraction_mode: ExtractionMode = ExtractionMode.RAW,
    use_chat_format: bool = False,
    add_schema: bool = False,
    num_samples: int = 5,
):
    """
    Quick evaluation on a few samples.

    Args:
        extraction_mode: Extraction mode to use
        use_chat_format: Whether to use chat format
        add_schema: Whether to include schema in prompt
        num_samples: Number of samples to test
    """
    logger.info("=" * 60)
    logger.info("Quick NER Evaluation")
    logger.info("=" * 60)

    # Load test dataset
    test_dataset_path = PROCESSED_DATA_DIR / "test" / "test.json"
    if not test_dataset_path.exists():
        logger.error(f"Test dataset not found: {test_dataset_path}")
        return

    test_dataset = DataProcessorService.load_dataset(test_dataset_path)
    logger.info(f"Loaded {len(test_dataset)} test samples")

    # Create configuration
    config = NERPromptEngineeringConfig(
        model_name="Qwen/Qwen3-4B",
        extraction_mode=extraction_mode,
        use_chat_format=use_chat_format,
        add_schema=add_schema,
        temperature=0.1,
        max_new_tokens=2048,
    )

    logger.info(f"Configuration:")
    logger.info(f"  - Model: {config.model_name}")
    logger.info(f"  - Mode: {config.extraction_mode.value}")
    logger.info(f"  - Chat format: {config.use_chat_format}")
    logger.info(f"  - Add schema: {config.add_schema}")
    logger.info("")

    # Initialize extractor
    try:
        extractor = PromptNERExtractor(config)
    except ImportError as e:
        logger.error(f"Failed to initialize extractor: {e}")
        return

    # Test on a few samples
    test_samples = test_dataset[:num_samples]

    logger.info(f"Testing on {len(test_samples)} samples")
    logger.info("=" * 60)

    correct = 0
    total = 0

    for i, sample in enumerate(test_samples, 1):
        logger.info(f"\nSample {i}/{len(test_samples)}")
        logger.info("-" * 60)

        text = sample["text"]
        ground_truth = sample["entities"]

        # Show truncated text
        text_preview = text[:200] + "..." if len(text) > 200 else text
        logger.info(f"Text: {text_preview}")

        # Extract entities
        try:
            predicted = extractor.extract_entities(text)
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            continue

        # Display results
        logger.info(f"\nGround Truth:")
        for entity_type in ["person", "organizations", "address"]:
            entities = ground_truth.get(entity_type, [])
            logger.info(f"  {entity_type}: {entities}")

        logger.info(f"\nPredicted:")
        for entity_type in ["person", "organizations", "address"]:
            entities = predicted.get(entity_type, [])
            logger.info(f"  {entity_type}: {entities}")

        # Calculate match
        match_score = 0
        for entity_type in ["person", "organizations", "address"]:
            pred_set = set(predicted.get(entity_type, []))
            truth_set = set(ground_truth.get(entity_type, []))
            if pred_set == truth_set:
                match_score += 1
            total += 1

        correct += match_score

        logger.info(f"\nMatch score: {match_score}/3")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    accuracy = (correct / total * 100) if total > 0 else 0
    logger.info(f"Exact match accuracy: {correct}/{total} ({accuracy:.1f}%)")
    logger.info(f"(Percentage of entity types that exactly matched ground truth)")


if __name__ == "__main__":
    # Quick test with default settings
    quick_evaluate(
        extraction_mode=ExtractionMode.RAW,
        use_chat_format=False,
        add_schema=False,
        num_samples=5,
    )

    # Test with schema
    quick_evaluate(
        extraction_mode=ExtractionMode.RAW,
        use_chat_format=False,
        add_schema=True,
        num_samples=5,
    )

    # Test with chat format
    quick_evaluate(
        extraction_mode=ExtractionMode.RAW,
        use_chat_format=True,
        add_schema=False,
        num_samples=5,
    )

    # Test with vLLM (requires vLLM installed)
    quick_evaluate(
        extraction_mode=ExtractionMode.STRUCTURED_OUTPUT,
        use_chat_format=True,
        add_schema=True,
        num_samples=5,
    )

    # Test with Outlines (requires outlines installed)
    quick_evaluate(
        extraction_mode=ExtractionMode.OUTLINES,
        use_chat_format=False,
        add_schema=False,
        num_samples=5,
    )
