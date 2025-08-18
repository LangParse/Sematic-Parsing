import os
from datetime import datetime

from src.data.amr_to_json import amr_to_json
from src.models.train import train_model


def main():
    # Versioned model output directory
    version = datetime.now().strftime("train_%Y%m%d_%H%M%S")
    model_output_dir = os.path.join("models", version)
    log_dir = os.path.join("reports", "logs", version)
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Convert AMR files to JSON (combine 1 and 2)
    amr1 = os.path.join("data", "train", "train_amr_1.txt")
    amr2 = os.path.join("data", "train", "train_amr_2.txt")
    combined_amr = os.path.join("data", "processed", "amr_combined.txt")
    json_path = os.path.join("data", "processed", "amr_data.json")
    # Combine the two AMR files
    with (
        open(amr1, "r", encoding="utf-8") as f1,
        open(amr2, "r", encoding="utf-8") as f2,
        open(combined_amr, "w", encoding="utf-8") as fout,
    ):
        fout.write(f1.read())
        fout.write("\n")
        fout.write(f2.read())
    # Convert to JSON
    amr_to_json(combined_amr, json_path)

    # Train model
    model, tokenizer = train_model(json_path, model_output_dir, log_dir)
    # Save model and tokenizer
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"Model and tokenizer saved to {model_output_dir}")


if __name__ == "__main__":
    main()
