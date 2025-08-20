from enum import Enum
from src.amr_demo import AMRDemo
from src.amr_evaluator import AMREvaluator
from src.amr_prediction import AMRPredictor
from src.amr_processor import AMRProcessor
from src.amr_trainer import AMRTrainer


class RunMode(Enum):
    TRAIN = "train"
    EVALUATE = "evaluate"
    PREDICT = "predict"
    DEMO = "demo"


def main():
    RUN_MODE = RunMode.TRAIN  # Chọn chế độ: TRAIN, EVALUATE, PREDICT

    processor = AMRProcessor()
    dataset = processor.process_to_dataset(
        train_path=["./data/train/train_amr_1.txt", "./data/train/train_amr_2.txt"],
        test_path="./data/test/public_test.txt",
    )

    match RUN_MODE.value:
        case "train":
            trainer = AMRTrainer(dataset, use_wandb=True)
            trainer.train()
        case "evaluate":
            evaluator = AMREvaluator(
                model_dir="./models/v1/amr_parser",
                dataset=dataset,  # pyright: ignore
            )

            # Evaluate trên validation
            val_results = evaluator.evaluate("validation")
            print("Validation Results:", val_results)

            test_preds = evaluator.generate_predictions("test", num_samples=3)
        case "predict":
            predictor = AMRPredictor(model_dir="./models/v1/amr_parser")
            test_preds = predictor.predict_file(
                test_path="./data/test/public_test.txt",
                output_path="./data/test/predicted_amr.txt",
            )

            print("Predictions saved to ./data/test/predicted_amr.txt")
            for sent, amr in test_preds[:3]:
                print("Sentence:", sent)
                print("Predicted AMR:\n", amr)
                print("------")
        case "demo":
            demo = AMRDemo(model_dir="./models/v1/amr_parser")
            demo.launch()


if __name__ == "__main__":
    main()
