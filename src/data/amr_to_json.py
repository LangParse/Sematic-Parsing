import json
import os


def amr_to_json(input_file, output_file=None):
    """
    Convert one or more cleaned AMR files to a deduplicated JSON format for training.
    Args:
        input_file (str or list[str]): Path(s) to the input AMR file(s).
        output_file (str, optional): Path to save the output JSON. If None, returns the data as a list.
    Returns:
        list: The combined and deduplicated data if output_file is None, else None.
    """

    def parse_amr_lines(lines: list[str]) -> list[dict[str, str]]:
        data = []
        current_sentence = None
        amr_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith("#::snt"):
                if current_sentence and amr_lines:
                    data.append(
                        {"input": current_sentence, "output": "\n".join(amr_lines)}
                    )
                current_sentence = line[7:].strip()
                amr_lines = []
            elif current_sentence is not None:
                amr_lines.append(line)
        if current_sentence and amr_lines:
            data.append({"input": current_sentence, "output": "\n".join(amr_lines)})
        return data

    def remove_duplicates(data: list[dict[str, str]]) -> list[dict[str, str]]:
        seen = set()
        filtered = []
        for entry in data:
            key = (entry["input"], entry["output"])
            if key not in seen:
                filtered.append(entry)
                seen.add(key)
        return filtered

    # Accept a single file, a list of files, or a directory
    if isinstance(input_file, str):
        if os.path.isdir(input_file):
            # Get all .txt files in the directory
            files = [
                os.path.join(input_file, f)
                for f in os.listdir(input_file)
                if f.endswith(".txt")
            ]
        else:
            files = [input_file]
    else:
        files = list(input_file)

    all_data = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        all_data.extend(parse_amr_lines(lines))

    filtered_data = remove_duplicates(all_data)

    if output_file is None:
        return filtered_data

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(filtered_data)} input/output pairs to: {output_file}")


# Example usage:
# Combine all .txt files in data/train and save to a single JSON file
# amr_to_json("data/train", "data/processed/train/combined_train.json")
# Or get the combined data as a list (no file saved):
amr_to_json(
    "data/train",
    "data/processed/train/combined_train.json",
)
