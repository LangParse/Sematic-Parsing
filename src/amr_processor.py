import concurrent.futures
import json
import logging
from pathlib import Path
from typing import Dict, Generator, List, Union, Optional, cast

from datasets import Dataset, DatasetDict


def _parse_amr_file(file_path: Path) -> List[Dict[str, str]]:
    """Parse an AMR text file into input-output pairs."""
    entries: List[Dict[str, str]] = []
    sentence = None
    amr_lines = []

    try:
        with file_path.open(encoding="utf-8") as f:
            lines = [line.strip() for line in f]
            if not any(line.startswith("#::snt") for line in lines):
                logging.warning(f"Invalid AMR format in {file_path}")
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
        logging.error(f"Error parsing {file_path}: {e}")
        return []


def _process_file(file_path: Path) -> List[Dict[str, str]]:
    """Process a single AMR file and segment its content."""
    return _parse_amr_file(file_path)


class AMRProcessor:
    """A class to process and tokenize AMR files into JSONL or Hugging Face Dataset format."""

    def __init__(self):
        self.logger = self._setup_logger()

    @staticmethod
    def _setup_logger() -> logging.Logger:
        logger = logging.getLogger("AMRProcessor")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        if not logger.hasHandlers():
            logger.addHandler(handler)

        return logger

    def file_to_jsonl(self, files: List[Path]) -> List[Dict[str, str]]:
        """Process AMR files into JSON-compatible dictionaries."""
        self.logger.info(f"Processing {len(files)} file(s) in parallel...")
        combined_data: List[Dict[str, str]] = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_file = {executor.submit(_process_file, f): f for f in files}
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

    def _process_test_file(
        self, file_path: Path
    ) -> Generator[Dict[str, str], None, None]:
        """Preprocess test file: each line is a sentence."""
        try:
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    sentence = line.strip()
                    if sentence:
                        yield {"input": sentence}
        except Exception as e:
            self.logger.error(f"Error processing test file {file_path}: {e}")
            return

    @staticmethod
    def _collect_files(
        path_or_list: Union[str, Path, List[Union[str, Path]]],
    ) -> List[Path]:
        """Convert str/Path/List to list[Path] without redundant recursion."""
        if isinstance(path_or_list, (str, Path)):
            path = Path(path_or_list)
            if path.is_file():
                return [path]
            elif path.is_dir():
                return sorted(path.glob("*.txt"))
            else:
                raise FileNotFoundError(f"Path not found: {path}")
        elif isinstance(path_or_list, list):
            result_files: List[Path] = []
            for p in path_or_list:
                result_files.extend(AMRProcessor._collect_files(p))
            return result_files

    def process_to_dataset(
        self,
        train_path: Union[str, Path, List[Union[str, Path]]],
        split_ratio: float = 0.8,
        val_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        test_path: Optional[Union[str, Path]] = None,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> DatasetDict:
        """Process AMR train/val/test into a Hugging Face DatasetDict."""
        train_files = self._collect_files(train_path)
        train_val_data = self.file_to_jsonl(train_files)

        # Split train/val
        if val_path:
            val_files = self._collect_files(val_path)
            val_data = self.file_to_jsonl(val_files)
            train_data = train_val_data
        else:
            if len(train_val_data) < 2:
                self.logger.warning(
                    "Not enough samples to split train/val. Using all for train."
                )
                train_data, val_data = train_val_data, []
            else:
                split_idx = int(len(train_val_data) * split_ratio)
                train_data = train_val_data[:split_idx]
                val_data = train_val_data[split_idx:]

        # --- Build dataset dict dynamically ---
        dataset_splits: Dict[str, Dataset] = {
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
        }

        # Process test
        if test_path:
            test_data = (
                list(self._process_test_file(Path(test_path))) if test_path else []
            )
            dataset_splits["test"] = Dataset.from_list(test_data)

        dataset = DatasetDict(cast(dict, dataset_splits))

        if save_dir:
            dataset.save_to_disk(str(save_dir))
            self.logger.info(f"Dataset saved to {save_dir}")

        return dataset
