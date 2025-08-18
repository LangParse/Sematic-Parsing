from src.amr_processor import AMRProcessor
from src.amr_tokenizer import AMRTokenizer
from src.amr_trainer import AMRTrainer


def main():
    processor = AMRProcessor()
    dataset = processor.process_to_dataset(
        train_path="./data/train/train_amr_2.txt", save_dir="./data/datasets"
    )

    amr_tokenizer = AMRTokenizer(save_dir="./data/datasets")
    tokenized_dataset = amr_tokenizer.tokenize_dataset(dataset, num_proc=4)

    trainer = AMRTrainer(
        dataset=tokenized_dataset, save_dir="./models/v1", use_wandb=True
    )
    trainer.train()


if __name__ == "__main__":
    main()
