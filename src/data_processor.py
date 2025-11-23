import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

from datasets import Dataset, load_dataset
from loguru import logger
from tokenizers import Tokenizer
from torch import Tensor

from src.config import NUM_TEST_SAMPLES, NUM_TRAIN_SAMPLES
from src.schema import NEREntities


class DataProcessor:
    """Service for processing and managing NER datasets."""

    @staticmethod
    def load_dataset_from_hub(
        dataset_name: str,
        split: str = "train",
        subset: Optional[str] = None,
        token: Optional[str] = None
    ) -> Dataset:
        """
        Load dataset from HuggingFace Hub.

        Args:
            dataset_name: HuggingFace dataset name (e.g., "conll2003")
            split: Dataset split (train/validation/test)
            subset: Dataset subset/configuration
            token: HuggingFace token for private datasets

        Returns:
            Dataset object
        """
        logger.info(f"Loading dataset {dataset_name} (split: {split})")
        dataset = load_dataset(
            dataset_name,
            subset,
            split=split,
            token=token
        )
        logger.info(f"Loaded {len(dataset)} samples")
        return dataset

    @staticmethod
    def load_dataset_from_file(
        file_path: Path,
        split: Optional[str] = None
    ) -> Dataset:
        """
        Load dataset from local JSON file.

        Args:
            file_path: Path to JSON file (or directory containing split files)
            split: Dataset split name (if file_path is a directory)

        Returns:
            Dataset object
        """
        # If file_path is a directory and split is provided
        if file_path.is_dir() and split:
            file_path = file_path / f"{split}.json"

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        logger.info(f"Loading dataset from {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = Dataset.from_list(data)
        logger.info(f"Loaded {len(dataset)} samples from {file_path}")
        return dataset

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special characters that might interfere with parsing
        text = text.strip()
        return text

    @staticmethod
    def load_articles_from_folder(folder_path: Path) -> List[Dict]:
        """
        Load all text articles from a folder.

        Args:
            folder_path: Path to folder containing txt files

        Returns:
            List of article dictionaries with file_name and text
        """
        articles = []
        txt_files = list(folder_path.glob("*.txt"))

        logger.info(f"Found {len(txt_files)} articles in {folder_path}")

        if "train" in folder_path.parts:
            txt_files = txt_files[:NUM_TRAIN_SAMPLES]
            logger.warning(f"Limiting to {NUM_TRAIN_SAMPLES} articles for training set")

        elif "test" in folder_path.parts:
            txt_files = txt_files[:NUM_TEST_SAMPLES]
            logger.warning(f"Limiting to {NUM_TEST_SAMPLES} articles for test set")

        for file_path in txt_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except UnicodeDecodeError as e:
                logger.error(f"Encoding error reading {file_path.name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error reading {file_path.name}: {e}")
                continue

            if text and len(text) >= 50:
                articles.append({"file_name": file_path.name, "text": text})
            else:
                logger.warning(f"Skipping empty/short file: {file_path.name}")

        return articles

    @staticmethod
    def load_dataset(file_path: Path) -> List[Dict]:
        """
        Load dataset from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            List of samples
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_dataset(samples: List[Dict], output_path: Path):
        """
        Save dataset to JSON file.

        Args:
            samples: List of samples
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(samples)} samples to {output_path}")

    @staticmethod
    def load_jsonl(file_path: Path) -> List[Dict]:
        """
        Load dataset from JSONL file (one JSON object per line).

        Args:
            file_path: Path to JSONL file

        Returns:
            List of samples
        """
        logger.info(f"Loading data from {file_path}")
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        logger.info(f"Loaded {len(samples)} samples from {file_path}")
        return samples

    @staticmethod
    def save_jsonl(samples: List[Dict], output_path: Path):
        """
        Save dataset in JSONL format (one JSON object per line).

        Args:
            samples: List of samples
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(samples)} samples to {output_path} (JSONL)")

    @staticmethod
    def compute_statistics(samples: List[Dict]) -> Dict:
        """
        Compute dataset statistics.

        Args:
            samples: List of samples with entities

        Returns:
            Statistics dictionary
        """
        if not samples:
            return {"error": "No samples provided"}

        total_samples = len(samples)
        entity_counts = {"person": 0, "organizations": 0, "address": 0}
        samples_with_entities = 0
        text_lengths = []
        entities_per_sample = []

        for sample in samples:
            text = sample.get("text", "")
            text_lengths.append(len(text))

            entities = sample.get("entities", {})
            sample_entity_count = 0

            for entity_type in ["person", "organizations", "address"]:
                entity_list = entities.get(entity_type, [])
                entity_counts[entity_type] += len(entity_list)
                sample_entity_count += len(entity_list)

            if sample_entity_count > 0:
                samples_with_entities += 1

            entities_per_sample.append(sample_entity_count)

        stats = {
            "total_samples": total_samples,
            "samples_with_entities": samples_with_entities,
            "samples_without_entities": total_samples - samples_with_entities,
            "entity_counts": entity_counts,
            "total_entities": sum(entity_counts.values()),
            "avg_entities_per_sample": round(
                sum(entities_per_sample) / total_samples, 2
            ),
            "avg_text_length": round(sum(text_lengths) / total_samples, 2),
            "min_text_length": min(text_lengths),
            "max_text_length": max(text_lengths),
        }

        return stats

    @staticmethod
    def merge_datasets(datasets: List[List[Dict]]) -> List[Dict]:
        """
        Merge multiple datasets into one.

        Args:
            datasets: List of dataset lists

        Returns:
            Merged dataset
        """
        merged = []
        for dataset in datasets:
            merged.extend(dataset)

        logger.info(f"Merged {len(datasets)} datasets into {len(merged)} samples")
        return merged

    @staticmethod
    def filter_by_category(samples: List[Dict], categories: List[str]) -> List[Dict]:
        """
        Filter samples by category.

        Args:
            samples: List of samples
            categories: List of category names to keep

        Returns:
            Filtered samples
        """
        filtered = [s for s in samples if s.get("category") in categories]
        logger.info(
            f"Filtered {len(filtered)} samples from {len(samples)} by categories: {categories}"
        )
        return filtered

    @staticmethod
    def split_dataset(
        samples: List[Dict], train_ratio: float = 0.8
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Split dataset into train and validation sets.

        Args:
            samples: List of samples
            train_ratio: Ratio for training set

        Returns:
            Tuple of (train_samples, val_samples)
        """
        import random

        random.shuffle(samples)

        split_idx = int(len(samples) * train_ratio)
        train = samples[:split_idx]
        val = samples[split_idx:]

        logger.info(
            f"Split {len(samples)} samples into {len(train)} train and {len(val)} val"
        )
        return train, val

    @staticmethod
    def deduplicate_entities(samples: List[Dict]) -> List[Dict]:
        """
        Remove duplicate entities within each sample.

        Args:
            samples: List of samples

        Returns:
            Samples with deduplicated entities
        """
        for sample in samples:
            entities = sample.get("entities", {})
            for entity_type in ["person", "organizations", "address"]:
                if entity_type in entities:
                    entities[entity_type] = list(set(entities[entity_type]))

        return samples

    @staticmethod
    def validate_samples(samples: List[Dict]) -> tuple[List[Dict], List[Dict]]:
        """
        Validate samples and separate valid from invalid ones.

        Args:
            samples: List of samples

        Returns:
            Tuple of (valid_samples, invalid_samples)
        """
        valid = []
        invalid = []

        for sample in samples:
            if not sample.get("text"):
                invalid.append(sample)
                continue

            if "entities" not in sample:
                invalid.append(sample)
                continue

            entities = sample["entities"]
            if not isinstance(entities, dict):
                invalid.append(sample)
                continue

            required_keys = ["person", "organizations", "address"]
            if not all(key in entities for key in required_keys):
                invalid.append(sample)
                continue

            valid.append(sample)

        logger.info(f"Validated: {len(valid)} valid, {len(invalid)} invalid samples")
        return valid, invalid

    @staticmethod
    def format_sample_for_finetuning(
        sample: Dict[str, List[str]],
        format: str = "chat",
        add_schema: bool = False
    ) -> Dict:
        """
        Format sample for LLM finetuning.

        Args:
            samples: Sample
            format: Export format ('chat' or 'instruction')
            add_schema: If True, adds JSON schema definition to the prompt for better structured output
        """

        instruction_with_schema = NEREntities.get_system_instruction(add_schema=add_schema)

        text = sample["text"]
        entities = sample["entities"]
        entities_str = json.dumps(entities, ensure_ascii=False)

        if format == "chat":
            # Chat format for conversational fine-tuning
            formatted = {
                "messages": [
                    {"role": "system", "content": instruction_with_schema},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": entities_str}
                ]
            }
        else:
            # Instruction format - split prompt into instruction and input
            formatted = {
                "instruction": instruction_with_schema,
                "input": f"Text:\n{text}\n\nJSON output:",
                "output": entities_str,
            }

        return formatted

    @staticmethod
    def export_for_finetuning(
        samples: List[Dict],
        output_path: Path,
        format: str = "chat",
        add_schema: bool = False,
    ):
        """
        Export dataset in format suitable for LLM finetuning.

        Uses the exact prompt format from prompt_engineering.py to ensure
        training data matches inference prompts.

        Args:
            samples: List of samples
            output_path: Output file path
            format: Export format ('chat' or 'instruction')
            add_schema: If True, adds JSON schema definition to the prompt for better structured output
        """

        formatted_samples = []
        for sample in samples:
            formatted = DataProcessor.format_sample_for_finetuning(
                sample=sample,
                format=format,
                add_schema=add_schema
            )
            formatted_samples.append(formatted)

        DataProcessor.save_jsonl(formatted_samples, output_path)
        logger.info(f"Exported {len(formatted_samples)} samples in {format} format (schema={'included' if add_schema else 'excluded'})")

    @staticmethod
    def format_chat_message(
        text: str,
        tokenizer: Tokenizer,
        tokenize: bool = False,
        add_schema: bool = False,
        enable_thinking: bool = False
    ) -> str | Tensor:
        """
        Format text as chat message using tokenizer's chat template.

        Args:
            text: Input text
            tokenizer: Tokenizer with apply_chat_template method
            add_schema: Include JSON schema in system instruction
            enable_thinking: Enable thinking mode in chat template

        Returns:
            Formatted prompt string
        """
        system_message = NEREntities.get_system_instruction(add_schema=add_schema)
        output = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": text}
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        if tokenize:
            output = tokenizer(
                output,
                return_tensors="pt"
            )
        return output
