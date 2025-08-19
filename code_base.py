import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party imports
import wandb
from datasets import Dataset
from nltk.translate.bleu_score import sentence_bleu
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
)


# ======================
# CONFIGURATION CLASS
# ======================
class Config:
    """Centralized configuration for the AMR pipeline"""

    # Paths (update these according to your Google Drive structure)
    BASE_DIR = "/content/drive/MyDrive/CITD/semantic-parsing"
    DATA_DIR = f"{BASE_DIR}/data/train"
    MODEL_DIR = f"{BASE_DIR}/models"
    LOG_DIR = f"{BASE_DIR}/logs"

    # File names
    RAW_AMR_FILE = "train_amr_1.txt"
    CLEANED_AMR_FILE = "input_amr.txt"
    JSON_DATA_FILE = "train_amr_1.json"
    TOKENIZED_DATA_FILE = "tokenized_data.txt"

    # Model settings
    MODEL_NAME = "VietAI/vit5-base"
    OUTPUT_MODEL_DIR = f"{MODEL_DIR}/amr_model"

    # Training parameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 10
    MAX_LENGTH = 512
    LOGGING_STEPS = 500
    SAVE_STEPS = 200
    SAVE_TOTAL_LIMIT = 2

    # WandB configuration
    WANDB_API_KEY = "449ae55008e2e327116d3d500dfda77cdf77ce70"
    WANDB_PROJECT = "AMR_ViT5_T4GPU"

    # Evaluation settings
    EVAL_SAMPLES = 100


# ======================
# LOGGING SETUP
# ======================
def setup_logging() -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{Config.LOG_DIR}/amr_pipeline.log"),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ======================
# UTILITY FUNCTIONS
# ======================
def ensure_dir_exists(file_path: str) -> None:
    """Ensure directory exists for given file path"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def get_file_paths() -> Dict[str, str]:
    """Get all file paths used in the pipeline"""
    return {
        "raw_amr": f"{Config.DATA_DIR}/{Config.RAW_AMR_FILE}",
        "cleaned_amr": f"{Config.DATA_DIR}/{Config.CLEANED_AMR_FILE}",
        "json_data": f"{Config.DATA_DIR}/{Config.JSON_DATA_FILE}",
        "tokenized_data": f"{Config.DATA_DIR}/{Config.TOKENIZED_DATA_FILE}",
        "model_output": Config.OUTPUT_MODEL_DIR,
    }


# ======================
# MODULE 1: DATA CLEANING
# ======================
class AMRDataCleaner:
    """Handles AMR data cleaning and preprocessing"""

    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        ensure_dir_exists(output_path)

    def clean_amr_file(self) -> bool:
        """
        Clean and normalize AMR file for further processing

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"🧼 Starting AMR file cleaning: {self.input_path}")

            # Read all lines
            with open(self.input_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Group lines into blocks
            blocks = self._group_into_blocks(lines)

            # Write cleaned blocks
            self._write_blocks(blocks)

            logger.info(f"✅ AMR file cleaned successfully: {self.output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Error cleaning AMR file: {e}")
            return False

    def _group_into_blocks(self, lines: List[str]) -> List[List[str]]:
        """Group lines into AMR blocks"""
        blocks = []
        current_block = []

        for line in lines:
            line = line.rstrip()

            if line.startswith("#::snt"):
                if current_block:
                    blocks.append(current_block)
                    current_block = []

            current_block.append(line)

        # Add final block
        if current_block:
            blocks.append(current_block)

        return blocks

    def _write_blocks(self, blocks: List[List[str]]) -> None:
        """Write blocks to output file"""
        with open(self.output_path, "w", encoding="utf-8") as f:
            for block in blocks:
                for line in block:
                    f.write(line + "\n")
                f.write("-----------------------------------------------\n")


# ======================
# MODULE 2: JSON CONVERSION
# ======================
class AMRToJSONConverter:
    """Converts AMR data to JSON format for training"""

    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        ensure_dir_exists(output_path)

    def convert_to_json(self) -> bool:
        """
        Convert AMR file to JSON format

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"🔄 Converting AMR to JSON: {self.input_path}")

            # Read and parse data
            data = self._parse_amr_data()

            # Remove duplicates
            filtered_data = self._remove_duplicates(data)

            # Save to JSON
            self._save_json(filtered_data)

            logger.info(
                f"✅ Saved {len(filtered_data)} input/output pairs to: {self.output_path}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Error converting to JSON: {e}")
            return False

    def _parse_amr_data(self) -> List[Dict[str, str]]:
        """Parse AMR data from file"""
        with open(self.input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

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

            elif line == "-----------------------------------------------":
                if current_sentence and amr_lines:
                    data.append(
                        {"input": current_sentence, "output": "\n".join(amr_lines)}
                    )
                    current_sentence = None
                    amr_lines = []

            elif current_sentence is not None and line:
                amr_lines.append(line)

        # Handle final block
        if current_sentence and amr_lines:
            data.append({"input": current_sentence, "output": "\n".join(amr_lines)})

        return data

    def _remove_duplicates(self, data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate entries"""
        seen = set()
        filtered_data = []

        for entry in data:
            key = (entry["input"], entry["output"])
            if key not in seen:
                filtered_data.append(entry)
                seen.add(key)

        return filtered_data

    def _save_json(self, data: List[Dict[str, str]]) -> None:
        """Save data to JSON file"""
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ======================
# MODULE 3: TOKENIZATION
# ======================
class AMRTokenizer:
    """Handles tokenization of AMR data"""

    def __init__(
        self, json_path: str, output_path: str, model_name: str = Config.MODEL_NAME
    ):
        self.json_path = json_path
        self.output_path = output_path
        self.model_name = model_name
        ensure_dir_exists(output_path)

    def tokenize_data(self) -> bool:
        """
        Tokenize AMR data for model training

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"✂️ Starting tokenization: {self.json_path}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load data
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Tokenize and save
            self._tokenize_and_save(data, tokenizer)

            logger.info(f"✅ Tokenization completed: {self.output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Error during tokenization: {e}")
            return False

    def _tokenize_and_save(self, data: List[Dict[str, str]], tokenizer) -> None:
        """Tokenize data and save to file"""
        with open(self.output_path, "w", encoding="utf-8") as f:
            for item in data:
                input_tokens = tokenizer.tokenize(item["input"])
                output_tokens = tokenizer.tokenize(item["output"])

                f.write("#::input_tokens\n")
                f.write(" ".join(input_tokens) + "\n\n")
                f.write("#::output_tokens\n")
                f.write(" ".join(output_tokens) + "\n")
                f.write("-" * 40 + "\n")


# ======================
# MODULE 4: MODEL TRAINING
# ======================
class AMRTrainer:
    """Handles model training with WandB integration"""

    def __init__(self, json_path: str, model_name: str = Config.MODEL_NAME):
        self.json_path = json_path
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

        # Setup WandB
        self._setup_wandb()

    def _setup_wandb(self) -> None:
        """Setup WandB for experiment tracking"""
        try:
            wandb.login(key=Config.WANDB_API_KEY)
            logger.info("📊 WandB connected successfully")
        except Exception as e:
            logger.warning(f"⚠️ WandB connection failed: {e}")

    def train_model(self) -> Optional[T5ForConditionalGeneration]:
        """
        Train the AMR model

        Returns:
            Trained model or None if failed
        """
        try:
            logger.info(f"🚀 Starting model training: {self.model_name}")

            # Load and prepare data
            dataset = self._prepare_dataset()

            # Load model and tokenizer
            self._load_model_and_tokenizer()

            # Setup training arguments
            training_args = self._get_training_args()

            # Create trainer
            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
            )

            # Start training
            trainer.train()

            logger.info("✅ Model training completed successfully")
            return self.model

        except Exception as e:
            logger.error(f"❌ Error during training: {e}")
            return None

    def _prepare_dataset(self) -> Dataset:
        """Prepare dataset for training"""
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = Dataset.from_list(data)
        return dataset.map(self._preprocess_function)

    def _load_model_and_tokenizer(self) -> None:
        """Load model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    def _preprocess_function(self, example: Dict[str, str]) -> Dict[str, any]:
        """Preprocess function for tokenization"""
        model_input = self.tokenizer(
            example["input"],
            padding="max_length",
            truncation=True,
            max_length=Config.MAX_LENGTH,
        )
        labels = self.tokenizer(
            example["output"],
            padding="max_length",
            truncation=True,
            max_length=Config.MAX_LENGTH,
        )
        model_input["labels"] = labels["input_ids"]
        return model_input

    def _get_training_args(self) -> Seq2SeqTrainingArguments:
        """Get training arguments configuration"""
        ensure_dir_exists(Config.OUTPUT_MODEL_DIR)
        ensure_dir_exists(Config.LOG_DIR)

        return Seq2SeqTrainingArguments(
            output_dir=Config.OUTPUT_MODEL_DIR,
            per_device_train_batch_size=Config.BATCH_SIZE,
            num_train_epochs=Config.NUM_EPOCHS,
            logging_dir=Config.LOG_DIR,
            logging_steps=Config.LOGGING_STEPS,
            save_steps=Config.SAVE_STEPS,
            save_total_limit=Config.SAVE_TOTAL_LIMIT,
            report_to=["wandb"],
            run_name=Config.WANDB_PROJECT,
            fp16=True,
            push_to_hub=False,
        )


# ======================
# MODULE 5: MODEL EVALUATION
# ======================
class AMREvaluator:
    """Handles model evaluation using BLEU score"""

    def __init__(self, model, tokenizer, json_path: str):
        self.model = model
        self.tokenizer = tokenizer
        self.json_path = json_path

    def evaluate_model(self) -> float:
        """
        Evaluate model using BLEU score

        Returns:
            Average BLEU score
        """
        try:
            logger.info("🧪 Starting model evaluation")

            with open(self.json_path, "r", encoding="utf-8") as f:
                samples = json.load(f)

            total_score = 0
            device = next(self.model.parameters()).device

            # Evaluate on first N samples
            eval_samples = min(len(samples), Config.EVAL_SAMPLES)

            for i, sample in enumerate(samples[:eval_samples]):
                if i % 20 == 0:
                    logger.info(f"Evaluating sample {i + 1}/{eval_samples}")

                score = self._calculate_bleu_score(sample, device)
                total_score += score

            avg_bleu = round(total_score / eval_samples, 4)
            logger.info(f"📊 Average BLEU score: {avg_bleu}")
            return avg_bleu

        except Exception as e:
            logger.error(f"❌ Error during evaluation: {e}")
            return 0.0

    def _calculate_bleu_score(self, sample: Dict[str, str], device) -> float:
        """Calculate BLEU score for a single sample"""
        try:
            input_ids = self.tokenizer(
                sample["input"], return_tensors="pt"
            ).input_ids.to(device)
            output_ids = self.model.generate(input_ids, max_length=Config.MAX_LENGTH)[0]
            prediction = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            reference = sample["output"].split()
            hypothesis = prediction.split()

            return sentence_bleu([reference], hypothesis)
        except Exception:
            return 0.0


# ======================
# MODULE 6: MODEL PERSISTENCE
# ======================
class ModelManager:
    """Handles model saving and loading"""

    @staticmethod
    def save_model(model, tokenizer, save_path: str = Config.OUTPUT_MODEL_DIR) -> bool:
        """
        Save model and tokenizer

        Args:
            model: Trained model
            tokenizer: Model tokenizer
            save_path: Path to save the model

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"💾 Saving model to: {save_path}")
            ensure_dir_exists(save_path)

            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

            logger.info("✅ Model saved successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return False

    @staticmethod
    def load_model(
        model_path: str = Config.OUTPUT_MODEL_DIR,
    ) -> Tuple[Optional[any], Optional[any]]:
        """
        Load saved model and tokenizer

        Args:
            model_path: Path to load the model from

        Returns:
            Tuple of (model, tokenizer) or (None, None) if failed
        """
        try:
            logger.info(f"📥 Loading model from: {model_path}")

            model = T5ForConditionalGeneration.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            logger.info("✅ Model loaded successfully")
            return model, tokenizer

        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return None, None


# ======================
# MODULE 7: PREDICTION
# ======================
class AMRPredictor:
    """Handles AMR prediction from Vietnamese sentences"""

    def __init__(self, model_path: str = Config.OUTPUT_MODEL_DIR):
        self.model, self.tokenizer = ModelManager.load_model(model_path)

        if self.model is None or self.tokenizer is None:
            raise ValueError("Failed to load model and tokenizer")

    def predict_interactive(self) -> None:
        """Interactive prediction mode"""
        logger.info("🎯 Starting interactive prediction mode")
        print("\n" + "=" * 50)
        print("🎯 AMR PREDICTION MODE")
        print("=" * 50)
        print("Nhập 'quit' để thoát")

        while True:
            try:
                input_sentence = input("\nNhập câu tiếng Việt: ").strip()

                if input_sentence.lower() in ["quit", "exit", "q"]:
                    print("👋 Tạm biệt!")
                    break

                if not input_sentence:
                    print("⚠️ Vui lòng nhập một câu hợp lệ")
                    continue

                amr_output = self.predict(input_sentence)
                print(f"\n📝 Sơ đồ AMR dự đoán:\n{amr_output}")
                print("-" * 50)

            except KeyboardInterrupt:
                print("\n👋 Tạm biệt!")
                break
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                print("❌ Có lỗi xảy ra trong quá trình dự đoán")

    def predict(self, input_sentence: str) -> str:
        """
        Predict AMR for a given Vietnamese sentence

        Args:
            input_sentence: Vietnamese sentence

        Returns:
            Predicted AMR string
        """
        try:
            input_ids = self.tokenizer(input_sentence, return_tensors="pt").input_ids

            with torch.no_grad():  # Save memory during inference
                output_ids = self.model.generate(
                    input_ids,
                    max_length=Config.MAX_LENGTH,
                    num_beams=4,  # Use beam search for better quality
                    early_stopping=True,
                )[0]

            prediction = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            return prediction

        except Exception as e:
            logger.error(f"Error predicting AMR: {e}")
            return "❌ Không thể dự đoán AMR cho câu này"


# ======================
# MAIN PIPELINE CLASS
# ======================
class AMRPipeline:
    """Main pipeline orchestrator"""

    def __init__(self):
        self.paths = get_file_paths()
        logger.info("🔧 AMR Pipeline initialized")

    def run_full_pipeline(self) -> bool:
        """
        Run the complete AMR processing pipeline

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("🚀 Starting full AMR pipeline")

        # Module 1: Clean data
        if not self._run_data_cleaning():
            return False

        # Module 2: Convert to JSON
        if not self._run_json_conversion():
            return False

        # Module 3: Tokenize data
        if not self._run_tokenization():
            return False

        # Module 4: Train model
        model = self._run_training()
        if model is None:
            return False

        # Module 5: Evaluate model
        self._run_evaluation(model)

        # Module 6: Save model
        if not self._run_model_saving(model):
            return False

        logger.info("🎉 Full pipeline completed successfully!")
        return True

    def _run_data_cleaning(self) -> bool:
        """Run data cleaning module"""
        cleaner = AMRDataCleaner(self.paths["raw_amr"], self.paths["cleaned_amr"])
        return cleaner.clean_amr_file()

    def _run_json_conversion(self) -> bool:
        """Run JSON conversion module"""
        converter = AMRToJSONConverter(
            self.paths["cleaned_amr"], self.paths["json_data"]
        )
        return converter.convert_to_json()

    def _run_tokenization(self) -> bool:
        """Run tokenization module"""
        tokenizer = AMRTokenizer(self.paths["json_data"], self.paths["tokenized_data"])
        return tokenizer.tokenize_data()

    def _run_training(self) -> Optional[any]:
        """Run model training"""
        trainer = AMRTrainer(self.paths["json_data"])
        return trainer.train_model()

    def _run_evaluation(self, model) -> None:
        """Run model evaluation"""
        trainer = AMRTrainer(self.paths["json_data"])
        evaluator = AMREvaluator(model, trainer.tokenizer, self.paths["json_data"])
        evaluator.evaluate_model()

    def _run_model_saving(self, model) -> bool:
        """Save the trained model"""
        trainer = AMRTrainer(self.paths["json_data"])
        return ModelManager.save_model(model, trainer.tokenizer)

    def run_prediction_mode(self) -> None:
        """Run interactive prediction mode"""
        try:
            predictor = AMRPredictor()
            predictor.predict_interactive()
        except Exception as e:
            logger.error(f"❌ Error in prediction mode: {e}")
            print(
                "❌ Không thể khởi chạy chế độ dự đoán. Vui lòng kiểm tra model đã được train chưa."
            )


# ======================
# MAIN EXECUTION
# ======================
def main():
    """Main execution function"""
    print("🎯 AMR SEMANTIC PARSING PIPELINE")
    print("=" * 50)
    print("1. Run full pipeline (clean → train → evaluate)")
    print("2. Run prediction mode only")
    print("3. Run individual modules")

    choice = input("\nChọn chế độ (1/2/3): ").strip()

    pipeline = AMRPipeline()

    if choice == "1":
        success = pipeline.run_full_pipeline()
        if success:
            print("\n🎉 Pipeline hoàn thành! Bạn có thể chạy chế độ dự đoán.")

    elif choice == "2":
        pipeline.run_prediction_mode()

    elif choice == "3":
        print("\n📋 Individual modules:")
        print("1. Data cleaning")
        print("2. JSON conversion")
        print("3. Tokenization")
        print("4. Model training")
        print("5. Model evaluation")

        module_choice = input("Chọn module (1-5): ").strip()

        if module_choice == "1":
            pipeline._run_data_cleaning()
        elif module_choice == "2":
            pipeline._run_json_conversion()
        elif module_choice == "3":
            pipeline._run_tokenization()
        elif module_choice == "4":
            model = pipeline._run_training()
            if model:
                pipeline._run_model_saving(model)
        elif module_choice == "5":
            # Load existing model for evaluation
            model, tokenizer = ModelManager.load_model()
            if model and tokenizer:
                evaluator = AMREvaluator(model, tokenizer, pipeline.paths["json_data"])
                evaluator.evaluate_model()
    else:
        print("❌ Lựa chọn không hợp lệ")


# ======================
# ENTRY POINT
# ======================
if __name__ == "__main__":
    # Add missing import for torch
    try:
        import torch
    except ImportError:
        logger.error("❌ PyTorch not installed. Please install: pip install torch")
        exit(1)

    main()
