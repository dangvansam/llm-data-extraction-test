import sys
from loguru import logger

sys.path.append(".")

from src.config import ExtractionMode, NERPromptEngineeringConfig
from src.prompt_engineering import PromptNERExtractor


def example_raw_mode():
    """Example using RAW mode (standard generation)."""
    logger.info("=" * 60)
    logger.info("Example 1: RAW Mode (Standard Generation)")
    logger.info("=" * 60)

    config = NERPromptEngineeringConfig(extraction_mode=ExtractionMode.RAW)
    extractor = PromptNERExtractor(config)

    text = """
    Công ty Cổ phần FPT là một trong những doanh nghiệp hàng đầu Việt Nam.
    CEO Trương Gia Bình đã có buổi gặp gỡ tại Hà Nội.
    """

    entities = extractor.extract_entities(text)

    logger.info("Extracted entities (RAW mode):")
    logger.info(f"  - Person: {entities['person']}")
    logger.info(f"  - Organizations: {entities['organizations']}")
    logger.info(f"  - Address: {entities['address']}")
    logger.info("")


def example_structured_output_mode():
    """Example using STRUCTURED_OUTPUT mode (vLLM)."""
    logger.info("=" * 60)
    logger.info("Example 2: STRUCTURED_OUTPUT Mode (vLLM)")
    logger.info("=" * 60)

    config = NERPromptEngineeringConfig(extraction_mode=ExtractionMode.STRUCTURED_OUTPUT)
    extractor = PromptNERExtractor(config)

    text = """
    Vingroup công bố dự án đầu tư 5000 tỷ đồng tại Bình Dương.
    Chủ tịch Phạm Nhật Vượng tham dự buổi lễ khởi công.
    """

    entities = extractor.extract_entities(text)

    logger.info("Extracted entities (STRUCTURED_OUTPUT mode):")
    logger.info(f"  - Person: {entities['person']}")
    logger.info(f"  - Organizations: {entities['organizations']}")
    logger.info(f"  - Address: {entities['address']}")
    logger.info("")


def example_outlines_mode():
    """Example using OUTLINES mode (Outlines library)."""
    logger.info("=" * 60)
    logger.info("Example 3: OUTLINES Mode (Outlines Library)")
    logger.info("=" * 60)

    config = NERPromptEngineeringConfig(extraction_mode=ExtractionMode.OUTLINES)
    extractor = PromptNERExtractor(config)

    text = """
    Bệnh viện Bạch Mai ở Hà Nội là một trong những bệnh viện lớn nhất.
    Bác sĩ Nguyễn Văn A làm Trưởng khoa Tim mạch.
    """

    entities = extractor.extract_entities(text)

    logger.info("Extracted entities (OUTLINES mode):")
    logger.info(f"  - Person: {entities['person']}")
    logger.info(f"  - Organizations: {entities['organizations']}")
    logger.info(f"  - Address: {entities['address']}")
    logger.info("")


def example_mode_features():
    """Display features of each extraction mode."""
    logger.info("=" * 60)
    logger.info("Extraction Mode Comparison")
    logger.info("=" * 60)

    features = {
        "RAW": {
            "Speed": "Standard",
            "Structured Output": "No (requires parsing)",
            "Guaranteed Valid JSON": "No",
            "GPU Required": "No (can use CPU)",
            "Dependencies": "transformers only",
            "Best For": "Simple cases, CPU inference",
        },
        "STRUCTURED_OUTPUT": {
            "Speed": "Very Fast (10-50x)",
            "Structured Output": "Yes (JSON schema)",
            "Guaranteed Valid JSON": "Yes",
            "GPU Required": "Yes",
            "Dependencies": "vLLM",
            "Best For": "Production, batch processing",
        },
        "OUTLINES": {
            "Speed": "Fast (constraint-guided)",
            "Structured Output": "Yes (Pydantic models)",
            "Guaranteed Valid JSON": "Yes",
            "GPU Required": "No (can use CPU)",
            "Dependencies": "outlines",
            "Best For": "Type-safe extraction, complex schemas",
        },
    }

    for mode, attrs in features.items():
        logger.info(f"\n{mode} Mode:")
        for key, value in attrs.items():
            logger.info(f"  {key:25s}: {value}")


if __name__ == "__main__":
    logger.info("NER Extraction Modes Examples\n")

    example_mode_features()

    # Run examples
    example_raw_mode()
    example_structured_output_mode()
    example_outlines_mode()