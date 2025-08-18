import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase
from datasets import DatasetDict


class AMRTokenizer:
    """Wrapper class to handle input/output tokenization for AMR datasets."""

    def __init__(
        self,
        model_name: str = "VietAI/vit5-base",
        max_length_input: int = 128,
        max_length_output: int = 256,
        save_dir: Optional[Union[str, Path]] = None,
    ):
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name
        )
        self.max_length_input = max_length_input
        self.max_length_ouput = max_length_output
        self.save_dir = Path(save_dir) if save_dir else None

    def _tokenize_batch(self, batch: Dict[str, Any]) -> BatchEncoding:
        """Tokenize both input (Vietnamese sentence) and output (AMR graph) for a batch."""
        # Tokenize input sentences
        model_inputs = self.tokenizer(
            batch["input"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length_input,
            return_tensors="pt",
        )

        # Tokenize ouput graphs if preset (train/val splits)
        if "output" in batch:
            labels = self.tokenizer(
                batch["output"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length_ouput,
                return_tensors="pt",
            )
            model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def tokenize_dataset(
        self, dataset: DatasetDict, num_proc: Optional[int] = None, batched: bool = True
    ):
        """
        Apply tokenization to all splits in a DatasetDict.
        If save_dir is defined, the tokenized dataset will also be saved.
        """
        tokenized_dataset = DatasetDict()

        for split, ds in dataset.items():
            # Skip empty splits
            if len(ds) == 0:
                logging.warning(f"Skipping empty split: {split}")
                continue

            remove_columns = None if split == "test" else ds.column_names

            tokenized_dataset[split] = ds.map(
                self._tokenize_batch,
                batched=batched,
                num_proc=num_proc,
                remove_columns=remove_columns,
            )

        # Auto-save if save_dir is defined
        if self.save_dir:
            self._save(tokenized_dataset)

        return tokenized_dataset

    def _save(self, dataset: DatasetDict) -> None:
        """Private save method (internal use only)."""
        path = self.save_dir
        if not path:
            raise ValueError("Save directory is not defined.")
        path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(path))
        print(f"Tokenized dataset saved to {path}")
