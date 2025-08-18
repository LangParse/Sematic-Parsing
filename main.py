from src.amr_processor import AMRProcessor
from src.amr_trainer import AMRTrainer


def main():
    processor = AMRProcessor()
    dataset = processor.process_to_dataset(
        train_path=["../data/train/train_amr_1.txt", "../data/train/train_amr_2.txt"]
    )

    trainer = AMRTrainer(dataset, use_wandb=True)
    trainer.train()


if __name__ == "__main__":
    main()
