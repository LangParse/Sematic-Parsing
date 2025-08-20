
from typing import Dict
from transformers import AutoTokenizer

class AMRPreprocessor:
    def __init__(self, model_name: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length

    def __call__(self, example: Dict[str, str]) -> Dict[str, list]:
        model_input = self.tokenizer(
            example["input"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        labels = self.tokenizer(
            example["output"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        model_input["labels"] = labels["input_ids"]
        return model_input
