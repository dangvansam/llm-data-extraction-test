import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from loguru import logger
from tqdm import tqdm

sys.path.append(".")

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, NERLangExtractConfig
from src.pipeline.langextract_pipeline import LangExtractNERExtractor
from src.data_processor import DataProcessor


def get_available_categories(split: str) -> List[str]:
    """
    Get list of available categories in a split.

    Args:
        split: Split name ('train' or 'test')

    Returns:
        List of category names
    """
    split_path = RAW_DATA_DIR / split
    if not split_path.exists():
        return []

    categories = [d.name for d in split_path.iterdir() if d.is_dir()]
    return sorted(categories)


def count_processed_samples(split: str, category: str, output_dir: Path) -> int:
    """
    Count how many samples have already been processed for a category.

    Args:
        split: Split name
        category: Category name
        output_dir: Base output directory

    Returns:
        Number of processed samples
    """
    category_output_dir = output_dir / split / category / "json"
    if not category_output_dir.exists():
        return 0

    return len(list(category_output_dir.glob("*.json")))


def process_single_article(
    article: dict,
    category_name: str,
    split: str,
    output_dir: Path,
    extractor: LangExtractNERExtractor,
    extraction_passes: int = 2,
    max_workers: int = 4
) -> dict:
    """
    Process a single article and save to individual JSON file.

    Args:
        article: Article dictionary with file_name and text
        category_name: Category name
        split: Split name ('train' or 'test')
        output_dir: Base output directory
        extractor: LangExtract extractor instance
        extraction_passes: Number of extraction passes
        max_workers: Number of parallel workers

    Returns:
        Processed sample dictionary
    """
    category_output_dir = output_dir / split / category_name / "json"
    category_output_dir.mkdir(parents=True, exist_ok=True)

    file_stem = Path(article["file_name"]).stem
    output_file = category_output_dir / f"{file_stem}.json"

    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                sample = json.load(f)
            logger.debug(f"Loaded cached result for {article['file_name']}")
            return sample
        except Exception as e:
            logger.warning(f"Failed to load cached result for {article['file_name']}: {e}")

    entities = extractor.extract_entities(text=article["text"])

    sample = {
        "file_name": article["file_name"],
        "category": category_name,
        "text": article["text"],
        "entities": entities
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

    return sample


def process_category(
    category_name: str,
    split: str,
    extractor: LangExtractNERExtractor,
    extraction_passes: int = 2,
    max_workers: int = 5,
    output_dir: Path = None
) -> List[dict]:
    """
    Process all articles in a category with parallel processing and incremental saving.

    Args:
        category_name: Category name
        split: Split name ('train' or 'test')
        extractor: LangExtract extractor instance
        extraction_passes: Number of extraction passes
        max_workers: Number of parallel workers for extraction
        output_dir: Base output directory

    Returns:
        List of processed samples
    """
    category_path = RAW_DATA_DIR / split / category_name

    if not category_path.exists():
        logger.warning(f"Category path not found: {category_path}")
        return []

    output_dir = output_dir or PROCESSED_DATA_DIR

    logger.info(f"Processing category: {category_name} ({split})")

    articles = DataProcessor.load_articles_from_folder(category_path)

    if not articles:
        logger.warning(f"No valid articles found in {category_name}")
        return []

    processed_count = count_processed_samples(split, category_name, output_dir)
    if processed_count > 0:
        logger.info(f"Found {processed_count} already processed samples - will skip those")

    samples = []
    failed_articles = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_article = {
            executor.submit(
                process_single_article,
                article,
                category_name,
                split,
                output_dir,
                extractor,
                extraction_passes,
                1
            ): article
            for article in articles
        }

        with tqdm(total=len(articles), desc=f"{category_name}") as pbar:
            for future in as_completed(future_to_article):
                article = future_to_article[future]
                try:
                    sample = future.result()
                    samples.append(sample)
                except Exception as e:
                    logger.error(f"Failed to extract from {article['file_name']}: {e}")
                    failed_articles.append(article["file_name"])
                finally:
                    pbar.update(1)

    if failed_articles:
        logger.warning(f"Failed articles ({len(failed_articles)}): {', '.join(failed_articles[:5])}{'...' if len(failed_articles) > 5 else ''}")

    logger.info(f"Processed {len(samples)}/{len(articles)} articles from {category_name}")
    return samples


def process_split(
    split: str,
    categories: List[str],
    extraction_passes: int = 2,
    max_workers: int = 5,
    output_dir: Path = None
) -> dict:
    """
    Process all specified categories in a split.

    Args:
        split: Split name ('train' or 'test')
        categories: List of category names
        extraction_passes: Number of extraction passes
        max_workers: Number of parallel workers
        output_dir: Output directory (default: PROCESSED_DATA_DIR)

    Returns:
        Dictionary with results and statistics
    """
    logger.info("=" * 70)
    logger.info(f"Processing {split.upper()} split with {len(categories)} categories")
    logger.info("=" * 70)

    output_dir = output_dir or PROCESSED_DATA_DIR
    config = NERLangExtractConfig()
    extractor = LangExtractNERExtractor(config=config)

    all_samples = []

    for category in categories:
        samples = process_category(
            category_name=category,
            split=split,
            extractor=extractor,
            extraction_passes=extraction_passes,
            max_workers=max_workers,
            output_dir=output_dir
        )
        all_samples.extend(samples)

    if not all_samples:
        logger.error(f"No samples extracted for {split} split")
        return {"samples": [], "statistics": {}}

    all_samples = DataProcessor.deduplicate_entities(all_samples)

    valid_samples, invalid_samples = DataProcessor.validate_samples(all_samples)

    if invalid_samples:
        logger.warning(f"Found {len(invalid_samples)} invalid samples")

    output_path = output_dir / f"{split}.json"
    DataProcessor.save_dataset(valid_samples, output_path)

    jsonl_path = output_dir / f"{split}.jsonl"
    DataProcessor.save_jsonl(valid_samples, jsonl_path)

    finetuning_chat_path = output_dir / f"{split}_finetuning_chat.jsonl"
    DataProcessor.export_for_finetuning(
        valid_samples,
        finetuning_chat_path,
        format="chat"
    )
    
    finetuning_instruction_path = output_dir / f"{split}_finetuning_instruction.jsonl"
    DataProcessor.export_for_finetuning(
        valid_samples,
        finetuning_instruction_path
    )

    statistics = DataProcessor.compute_statistics(valid_samples)

    logger.info(f"\n{split.upper()} Statistics:")
    logger.info(json.dumps(statistics, indent=2, ensure_ascii=False))

    return {
        "samples": valid_samples,
        "statistics": statistics,
        "output_files": {
            "json": str(output_path),
            "jsonl": str(jsonl_path),
            "finetuning_chat": str(finetuning_chat_path),
            "finetuning_instruction": str(finetuning_instruction_path)
        }
    }


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(
        description="Extract entities from raw articles using Langextract"
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "both"],
        default="both",
        help="Which split to process (default: both)"
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        help="Specific categories to process (e.g., 'Doi song' 'Van hoa')"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available categories"
    )

    parser.add_argument(
        "--passes",
        type=int,
        default=2,
        help="Number of extraction passes (default: 2)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help=f"Output directory (default: {PROCESSED_DATA_DIR})"
    )

    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available categories and exit"
    )

    args = parser.parse_args()

    if args.list_categories:
        train_cats = get_available_categories("train")
        test_cats = get_available_categories("test")

        logger.info("Available categories:")
        logger.info(f"  Train: {', '.join(train_cats)}")
        logger.info(f"  Test: {', '.join(test_cats)}")
        return

    if not args.all and not args.categories:
        logger.error("Please specify --all or --categories")
        parser.print_help()
        return

    output_dir = Path(args.output_dir) if args.output_dir else PROCESSED_DATA_DIR

    splits_to_process = []
    if args.split == "both":
        splits_to_process = ["train", "test"]
    else:
        splits_to_process = [args.split]

    all_results = {}

    for split in splits_to_process:
        if args.all:
            categories = get_available_categories(split)
        else:
            categories = args.categories

        if not categories:
            logger.warning(f"No categories to process for {split} split")
            continue

        logger.info(f"Categories to process in {split}: {', '.join(categories)}")

        result = process_split(
            split=split,
            categories=categories,
            extraction_passes=args.passes,
            max_workers=args.workers,
            output_dir=output_dir
        )

        all_results[split] = result

    combined_stats_path = output_dir / "statistics.json"
    combined_stats = {
        split: result["statistics"]
        for split, result in all_results.items()
    }

    with open(combined_stats_path, "w", encoding="utf-8") as f:
        json.dump(combined_stats, f, ensure_ascii=False, indent=2)

    logger.info(f"\nCombined statistics saved to: {combined_stats_path}")

    logger.info("\n" + "=" * 70)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 70)

    for split, result in all_results.items():
        logger.info(f"\n{split.upper()}:")
        logger.info(f"  Samples: {len(result['samples'])}")
        logger.info("  Output files:")
        for file_type, path in result["output_files"].items():
            logger.info(f"    {file_type}: {path}")


if __name__ == "__main__":
    """
    Extract entities from raw article data using Langextract pipeline.

    This script processes raw Vietnamese news articles stored in data/raw/train and data/raw/test
    folders, extracts named entities (person, organizations, address) using Langextract,
    and creates labeled datasets for LLM evaluation and finetuning.

    Usage:
        # Process specific categories
        python scripts/data_preparation.py --categories "Doi song"

        # Process only train split
        python scripts/data_preparation.py --split train --all

        # Process with custom settings
        python scripts/data_preparation.py --all --passes 3 --workers 6
    """

    main()
