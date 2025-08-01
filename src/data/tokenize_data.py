import json
import os
import re


def simple_tokenize(text: str) -> list[str]:
    # Tokenizes words and punctuation
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def tokenize_amr_data_basic(
    input_json_path: str,
    output_path: str | None = None,
):
    def write_tokenized_data(data, output_path):
        with open(output_path, "w", encoding="utf-8") as f_out:
            for item in data:
                input_text = item["input"]
                output_text = item["output"]
                input_tokens = simple_tokenize(input_text)
                output_tokens = simple_tokenize(output_text)

                f_out.write("#::input_tokens\n")
                f_out.write(" ".join(input_tokens) + "\n\n")
                f_out.write("#::output_tokens\n")
                f_out.write(" ".join(output_tokens) + "\n")
                f_out.write("-" * 40 + "\n")

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(input_json_path), "tokenize_data.txt"
        )

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    write_tokenized_data(data, output_path)
    print(f"✅ Tokenized data saved at: {output_path}")


# Run this to tokenize your data
tokenize_amr_data_basic(
    "data/processed/train/train_amr.json",
    output_path="data/processed/train/tokenized_train.txt",
)
