"""
ViT5 Tokenizer Module
====================

This module handles tokenization using VietAI/vit5-base model with optimized processing.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer


@dataclass
class TokenizedSample:
    """Data class representing a tokenized sample."""

    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]

    def to_dict(self) -> Dict[str, List[int]]:
        """Convert to dictionary format."""
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
        }


class ViT5Tokenizer:
    """
    Handles tokenization using VietAI/vit5-base model.

    This class provides:
    - Efficient tokenization for Vietnamese text
    - Batch processing capabilities
    - Memory-optimized operations
    - Support for both training and inference
    """

    def __init__(
        self,
        model_name: str = "VietAI/vit5-base",
        max_length: int = 512,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize ViT5 tokenizer.

        Args:
            model_name: Name of the model to load tokenizer from
            max_length: Maximum sequence length
            logger: Optional logger instance
        """
        self.model_name = model_name
        self.max_length = max_length
        self.logger = logger or logging.getLogger(__name__)

        self.logger.info(f"Loading tokenizer: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.logger.info("✅ Tokenizer loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            raise

    def tokenize_sample(
        self, input_text: str, output_text: str, return_tensors: bool = False
    ) -> Union[TokenizedSample, Dict[str, torch.Tensor]]:
        """
        Tokenize a single input-output pair.

        Args:
            input_text: Input sentence
            output_text: Target AMR representation
            return_tensors: Whether to return PyTorch tensors

        Returns:
            TokenizedSample or dictionary with tensors
        """
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt" if return_tensors else None,
        )

        # Tokenize output (labels)
        with self.tokenizer.as_target_tokenizer():
            label_encoding = self.tokenizer(
                output_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt" if return_tensors else None,
            )

        if return_tensors:
            return {
                "input_ids": input_encoding["input_ids"].squeeze(),
                "attention_mask": input_encoding["attention_mask"].squeeze(),
                "labels": label_encoding["input_ids"].squeeze(),
            }
        else:
            return TokenizedSample(
                input_ids=input_encoding["input_ids"],
                attention_mask=input_encoding["attention_mask"],
                labels=label_encoding["input_ids"],
            )

    def tokenize_batch(
        self,
        input_texts: List[str],
        output_texts: List[str],
        return_tensors: bool = False,
    ) -> Union[List[TokenizedSample], Dict[str, torch.Tensor]]:
        """
        Tokenize a batch of input-output pairs efficiently.

        Args:
            input_texts: List of input sentences
            output_texts: List of target AMR representations
            return_tensors: Whether to return PyTorch tensors

        Returns:
            List of TokenizedSample or dictionary with batched tensors
        """
        if len(input_texts) != len(output_texts):
            raise ValueError("Input and output lists must have the same length")

        # Batch tokenize inputs
        input_encodings = self.tokenizer(
            input_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt" if return_tensors else None,
        )

        # Batch tokenize outputs (labels)
        with self.tokenizer.as_target_tokenizer():
            label_encodings = self.tokenizer(
                output_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt" if return_tensors else None,
            )

        if return_tensors:
            return {
                "input_ids": input_encodings["input_ids"],
                "attention_mask": input_encodings["attention_mask"],
                "labels": label_encodings["input_ids"],
            }
        else:
            samples = []
            batch_size = len(input_texts)

            for i in range(batch_size):
                sample = TokenizedSample(
                    input_ids=input_encodings["input_ids"][i],
                    attention_mask=input_encodings["attention_mask"][i],
                    labels=label_encodings["input_ids"][i],
                )
                samples.append(sample)

            return samples

    def process_jsonl_file(
        self, input_file: str, output_file: str, batch_size: int = 32
    ) -> None:
        """
        Process a JSONL file and save tokenized data.

        Args:
            input_file: Path to input JSONL file
            output_file: Path to output tokenized JSONL file
            batch_size: Batch size for processing
        """
        input_path = Path(input_file)
        output_path = Path(output_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Processing JSONL file: {input_file}")

        # Read all data
        data = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        total_samples = len(data)
        self.logger.info(
            f"Processing {total_samples} samples in batches of {batch_size}"
        )

        # Process in batches and save
        with open(output_path, "w", encoding="utf-8") as f:
            batch_indices = range(0, total_samples, batch_size)

            for i in tqdm(batch_indices, desc="🔤 Tokenizing batches", unit="batch"):
                batch = data[i : i + batch_size]

                input_texts = [item["input"] for item in batch]
                output_texts = [item["output"] for item in batch]

                tokenized_samples = self.tokenize_batch(input_texts, output_texts)

                # Save tokenized samples
                for sample in tokenized_samples:
                    json.dump(sample.to_dict(), f, ensure_ascii=False)
                    f.write("\n")

                if (i + batch_size) % (batch_size * 10) == 0:
                    self.logger.info(
                        f"Processed {min(i + batch_size, total_samples)}/{total_samples} samples"
                    )

        self.logger.info(f"✅ Tokenized data saved to: {output_file}")

    def decode_tokens(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_vocab_size(self) -> int:
        """Get vocabulary size of the tokenizer."""
        return len(self.tokenizer)

    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs."""
        return {
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "unk_token_id": self.tokenizer.unk_token_id,
        }

    def preprocess_for_training(self, data_file: str, output_dir: str) -> str:
        """
        Preprocess data for training by tokenizing and saving in optimized format.

        Args:
            data_file: Path to input JSONL file
            output_dir: Directory to save preprocessed data

        Returns:
            Path to preprocessed data file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        tokenized_file = output_path / "tokenized_data.jsonl"

        self.process_jsonl_file(data_file, str(tokenized_file))

        return str(tokenized_file)

    def create_data_collator(self):
        """
        Create a data collator for training.

        Returns:
            Data collator function
        """

        def collate_fn(batch):
            """Collate function for DataLoader."""
            input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
            attention_mask = torch.stack(
                [torch.tensor(item["attention_mask"]) for item in batch]
            )
            labels = torch.stack([torch.tensor(item["labels"]) for item in batch])

            # Replace padding token id's of the labels by -100 so they are ignored by loss
            labels[labels == self.tokenizer.pad_token_id] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        return collate_fn
