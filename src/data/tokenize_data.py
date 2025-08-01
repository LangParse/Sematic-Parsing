import json
import re
from pathlib import Path
from typing import Dict, List, Union

from pyvi import ViTokenizer


def vi_tokenize(text: str) -> List[str]:
    """Tokenize a Vietnamese string using pyvi."""
    return ViTokenizer.tokenize(text).split()


def parse_amr_file(file_path: Path) -> List[Dict[str, str]]:
    """Parse an AMR text file into input-output pairs."""
    entries = []
    sentence = None
    amr_lines = []

    with file_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#::snt"):
                if sentence and amr_lines:
                    entries.append({"input": sentence, "output": "\n".join(amr_lines)})
                sentence = line[7:].strip()
                amr_lines = []
            elif line and sentence is not None:
                amr_lines.append(line)

    if sentence and amr_lines:
        entries.append({"input": sentence, "output": "\n".join(amr_lines)})

    return entries


def tokenize_amr_output(text: str) -> List[str]:
    """Tokenize AMR string, preserving constructs like :ARG1(."""
    text = text.replace("\n", " ")
    pattern = r"\s*(:?\w+)\("
    text = re.sub(pattern, r" \1(", text, flags=re.VERBOSE)
    return [token for token in text.strip().split() if token]


def tokenize_to_jsonl(
    files: List[Path],
    input_tokenizer=vi_tokenize,
    output_tokenizer=tokenize_amr_output,
) -> List[Dict[str, str]]:
    """Tokenize AMR files into JSON-compatible dictionaries with space-separated tokens."""
    return [
        {
            "input": " ".join(input_tokenizer(entry["input"])),
            "output": " ".join(output_tokenizer(entry["output"])),
        }
        for file in files
        for entry in parse_amr_file(file)
        if entry["input"] and entry["output"]  # Skip empty entries
    ]


def save_to_jsonl(data: List[Dict[str, str]], output_path: Path) -> None:
    """Save data to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in data:
            if item["input"] and item["output"]:  # Ensure non-empty entries
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")


def process_amr_to_jsonl(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    tokenizer=vi_tokenize,
) -> List[Dict[str, str]]:
    """Process AMR input, tokenize, and save to JSONL."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    files = [input_path] if input_path.is_file() else sorted(input_path.glob("*.txt"))
    if not files:
        raise ValueError(f"No .txt files found in {input_path}")

    print(f"Processing {len(files)} file(s)...")
    combined_data = tokenize_to_jsonl(files, input_tokenizer=tokenizer)
    save_to_jsonl(combined_data, output_path)
    print(f"JSONL data saved to: {output_path}")
    return combined_data


if __name__ == "__main__":
    try:
        process_amr_to_jsonl(
            input_path="data/train",
            output_path="data/processed/train_tokenized.jsonl",
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
