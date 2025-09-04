"""
Data Loader Module
=================

This module handles loading and batching of AMR data for training and evaluation.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

from tqdm.auto import tqdm


@dataclass
class DataSplit:
    """Data class representing train/validation/test split information."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    def __post_init__(self):
        """Validate that ratios sum to 1.0."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


class DataLoader:
    """
    Handles loading and processing of AMR training data.

    This class provides functionality for:
    - Loading JSONL data files
    - Splitting data into train/validation/test sets
    - Batching data for training
    - Data validation and statistics
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize data loader.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def load_jsonl(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load data from JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            List of dictionaries containing input/output pairs

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSONL format is invalid
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {file_path}")

        self.logger.info(f"Loading data from: {file_path}")

        data = []
        try:
            # First pass: count total lines for progress bar
            with open(file_path, "r", encoding="utf-8") as f:
                total_lines = sum(1 for line in f if line.strip())

            # Second pass: load data with progress bar
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [line for line in f if line.strip()]

                for line_num, line in enumerate(
                    tqdm(lines, desc="📚 Loading data", unit="lines"), 1
                ):
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            if self._validate_item(item):
                                data.append(item)
                            else:
                                self.logger.warning(
                                    f"Invalid item at line {line_num}: {item}"
                                )
                        except json.JSONDecodeError as e:
                            self.logger.error(
                                f"JSON decode error at line {line_num}: {e}"
                            )
                            continue

            self.logger.info(f"✅ Loaded {len(data)} samples from JSONL")
            return data

        except IOError as e:
            self.logger.error(f"Error reading JSONL file: {e}")
            raise

    def _validate_item(self, item: Dict) -> bool:
        """
        Validate that an item has required fields.

        Args:
            item: Dictionary to validate

        Returns:
            True if item is valid, False otherwise
        """
        required_fields = ["input", "output"]
        return all(
            field in item and isinstance(item[field], str) and item[field].strip()
            for field in required_fields
        )

    def split_data(
        self, data: List[Dict[str, str]], split: DataSplit, random_seed: int = 42
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Split data into train/validation/test sets.

        Args:
            data: List of data samples
            split: DataSplit configuration
            random_seed: Random seed for reproducible splits

        Returns:
            Dictionary with 'train', 'val', and 'test' keys
        """
        if not data:
            raise ValueError("Cannot split empty data")

        # Set random seed for reproducible splits
        random.seed(random_seed)
        data_copy = data.copy()
        random.shuffle(data_copy)

        total_samples = len(data_copy)
        train_size = int(total_samples * split.train_ratio)
        val_size = int(total_samples * split.val_ratio)

        train_data = data_copy[:train_size]
        val_data = data_copy[train_size : train_size + val_size]
        test_data = data_copy[train_size + val_size :]

        self.logger.info(
            f"Data split - Train: {len(train_data)}, "
            f"Val: {len(val_data)}, Test: {len(test_data)}"
        )

        return {"train": train_data, "val": val_data, "test": test_data}

    def save_split_data(
        self, split_data: Dict[str, List[Dict[str, str]]], output_dir: str
    ) -> Dict[str, str]:
        """
        Save split data to separate JSONL files.

        Args:
            split_data: Dictionary containing train/val/test data
            output_dir: Directory to save split files

        Returns:
            Dictionary mapping split names to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        file_paths = {}

        for split_name, data in split_data.items():
            file_path = output_path / f"{split_name}.jsonl"

            with open(file_path, "w", encoding="utf-8") as f:
                for item in data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")

            file_paths[split_name] = str(file_path)
            self.logger.info(
                f"✅ Saved {len(data)} {split_name} samples to: {file_path}"
            )

        return file_paths

    def get_data_statistics(
        self, data: List[Dict[str, str]]
    ) -> Dict[str, Union[int, float]]:
        """
        Calculate statistics for the dataset.

        Args:
            data: List of data samples

        Returns:
            Dictionary containing dataset statistics
        """
        if not data:
            return {"total_samples": 0}

        input_lengths = [len(item["input"].split()) for item in data]
        output_lengths = [len(item["output"].split()) for item in data]

        stats = {
            "total_samples": len(data),
            "avg_input_length": sum(input_lengths) / len(input_lengths),
            "avg_output_length": sum(output_lengths) / len(output_lengths),
            "max_input_length": max(input_lengths),
            "max_output_length": max(output_lengths),
            "min_input_length": min(input_lengths),
            "min_output_length": min(output_lengths),
        }

        return stats

    def create_batches(
        self, data: List[Dict[str, str]], batch_size: int
    ) -> Iterator[List[Dict[str, str]]]:
        """
        Create batches from data for training.

        Args:
            data: List of data samples
            batch_size: Size of each batch

        Yields:
            Batches of data samples
        """
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]

    def filter_by_length(
        self,
        data: List[Dict[str, str]],
        max_input_length: Optional[int] = None,
        max_output_length: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Filter data by sequence length to avoid memory issues.

        Args:
            data: List of data samples
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length

        Returns:
            Filtered list of data samples
        """
        filtered_data = []

        for item in data:
            input_len = len(item["input"].split())
            output_len = len(item["output"].split())

            if max_input_length and input_len > max_input_length:
                continue
            if max_output_length and output_len > max_output_length:
                continue

            filtered_data.append(item)

        removed_count = len(data) - len(filtered_data)
        if removed_count > 0:
            self.logger.info(
                f"Filtered out {removed_count} samples due to length constraints"
            )

        return filtered_data
