"""Data loading and preprocessing utilities."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset


class NERDataLoader:
    """Load and preprocess NER datasets."""

    def __init__(
        self, dataset_name: Optional[str] = None, data_path: Optional[Path] = None
    ):
        """
        Initialize data loader.

        Args:
            dataset_name: HuggingFace dataset name (e.g., "conll2003")
            data_path: Path to local dataset
        """
        self.dataset_name = dataset_name
        self.data_path = data_path

    def load_dataset(self, split: str = "train") -> Dataset:
        """
        Load dataset from HuggingFace or local path.

        Args:
            split: Dataset split (train/validation/test)

        Returns:
            Dataset object
        """
        if self.dataset_name:
            return load_dataset(self.dataset_name, split=split)
        elif self.data_path:
            return self._load_local_dataset(split)
        else:
            raise ValueError("Either dataset_name or data_path must be provided")

    def _load_local_dataset(self, split: str) -> Dataset:
        """Load dataset from local JSON/CSV file."""
        file_path = self.data_path / f"{split}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Dataset.from_list(data)
        else:
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

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
    def create_train_val_split(
        data: List[Dict], val_ratio: float = 0.1
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Split data into train and validation sets.

        Args:
            data: List of samples
            val_ratio: Validation set ratio

        Returns:
            Tuple of (train_data, val_data)
        """
        import random

        random.shuffle(data)
        split_idx = int(len(data) * (1 - val_ratio))
        return data[:split_idx], data[split_idx:]

    @staticmethod
    def save_dataset(data: List[Dict], output_path: Path):
        """Save dataset to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_json_dataset(file_path: Path) -> List[Dict]:
        """Load dataset from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
