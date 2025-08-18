import concurrent.futures
import json
import logging
import re
from pathlib import Path
from typing import Callable, Dict, List, Union

from pyvi import ViTokenizer


class AMRProcessor:
    """A class to process and tokenize AMR files into JSONL format."""

    def __init__(self, input_tokenizer: Callable[[str], List[str]] = None):  # type: ignore
        """Initialize with an optional input tokenizer."""
        self.input_tokenizer = input_tokenizer or self._default_vi_tokenize
        self.logger = self._setup_logger()

    @staticmethod
    def _default_vi_tokenize(text: str) -> List[str]:
        """Tokenize a Vietnamese string using pyvi."""
        try:
            return ViTokenizer.tokenize(text).split()
        except Exception as e:
            logging.error(f"Error tokenizing text: {e}")
            return []

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Set up logging for the processor."""
        logger = logging.getLogger("AMRProcessor")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        return logger

    def _validate_amr_format(self, lines: List[str]) -> bool:
        """Validate basic AMR format (presence of #::snt and non-comment content)."""
        return any(line.startswith("#::snt") for line in lines) and any(
            line.strip() and not line.startswith("#") for line in lines
        )

    def _parse_amr_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Parse an AMR text file into input-output pairs."""
        entries: List[Dict[str, str]] = []
        sentence = None
        amr_lines = []

        try:
            with file_path.open(encoding="utf-8") as f:
                lines = [line.strip() for line in f]
                if not self._validate_amr_format(lines):
                    self.logger.warning(f"Invalid AMR format in {file_path}")
                    return []

                for line in lines:
                    if line.startswith("#::snt"):
                        if sentence and amr_lines:
                            entries.append(
                                {"input": sentence, "output": "\n".join(amr_lines)}
                            )
                        sentence = line[7:].strip()
                        amr_lines = []
                    elif line and sentence is not None:
                        amr_lines.append(line)

                if sentence and amr_lines:
                    entries.append({"input": sentence, "output": "\n".join(amr_lines)})

            return entries
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {e}")
            return []

    @staticmethod
    def _tokenize_amr_output(text: str) -> List[str]:
        """Tokenize AMR string, preserving constructs like :ARG1(."""
        try:
            text = text.replace("\n", " ")
            pattern = r"\s*(:?\w+)\("
            text = re.sub(pattern, r" \1(", text, flags=re.VERBOSE)
            return [token for token in text.strip().split() if token]
        except Exception as e:
            logging.error(f"Error tokenizing AMR output: {e}")
            return []

    def _process_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Process a single file and tokenize its content."""
        entries = self._parse_amr_file(file_path)
        return [
            {
                "input": " ".join(self.input_tokenizer(entry["input"])),
                "output": " ".join(self._tokenize_amr_output(entry["output"])),
            }
            for entry in entries
            if entry["input"] and entry["output"]
        ]

    def tokenize_to_jsonl(self, files: List[Path]) -> List[Dict[str, str]]:
        """Tokenize AMR files into JSON-compatible dictionaries with space-separated tokens."""
        self.logger.info(f"Processing {len(files)} file(s) in parallel...")
        combined_data: List[Dict[str, str]] = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_file = {executor.submit(self._process_file, f): f for f in files}
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    combined_data.extend(future.result())
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")

        return combined_data

    def save_to_jsonl(self, data: List[Dict[str, str]], output_path: Path) -> None:
        """Save data to a JSONL file."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                for item in data:
                    if item["input"] and item["output"]:
                        json.dump(item, f, ensure_ascii=False)
                        f.write("\n")
            self.logger.info(f"JSONL data saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving to {output_path}: {e}")

    def process(
        self, input_path: Union[str, Path], output_path: Union[str, Path]
    ) -> List[Dict[str, str]]:
        """Process AMR input, tokenize, and save to JSONL."""
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")

        files = (
            [input_path] if input_path.is_file() else sorted(input_path.glob("*.txt"))
        )
        if not files:
            raise ValueError(f"No .txt files found in {input_path}")

        combined_data = self.tokenize_to_jsonl(files)
        self.save_to_jsonl(combined_data, output_path)
        return combined_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = AMRProcessor()
    processor.process(
        input_path="data/train",
        output_path="data/processed/train_tokenized.jsonl",
    )
