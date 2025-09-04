#!/usr/bin/env python3
"""
AMR Semantic Parsing - Main Script
==================================

This is the main entry point for the AMR semantic parsing project.
It provides a command-line interface for all major operations.

Usage:
    python main.py --help
    python main.py process-data --input-dir data/train --output-dir data/processed
    python main.py train --config config/training_config.yaml
    python main.py evaluate --model-path models/amr_model --test-data data/processed/test.jsonl
    python main.py predict --model-path models/amr_model --text "Tôi yêu Việt Nam"
    python main.py interactive --model-path models/amr_model
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_processing import AMRProcessor, DataLoader
from src.evaluation import AMREvaluator
from src.inference import AMRPredictor
from src.training import AMRTrainer
from src.training.model_config import TrainingConfig
from src.utils.config import Config
from src.utils.logger import setup_project_logging


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="AMR Semantic Parsing - Vietnamese to AMR conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process raw AMR data
  python main.py process-data --input-dir data/train --output-dir data/processed
  
  # Train model
  python main.py train --config config/training_config.yaml
  
  # Evaluate model
  python main.py evaluate --model-path models/amr_model --test-data data/processed/test.jsonl
  
  # Single prediction
  python main.py predict --model-path models/amr_model --text "Tôi yêu Việt Nam"
  
  # Interactive mode
  python main.py interactive --model-path models/amr_model
  
  # Process file
  python main.py predict-file --model-path models/amr_model --input-file input.txt --output-file output.txt
        """,
    )

    # Global arguments
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress console output"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Data processing command
    process_parser = subparsers.add_parser("process-data", help="Process raw AMR data")
    process_parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing AMR files",
    )
    process_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for processed data",
    )
    process_parser.add_argument(
        "--split-data", action="store_true", help="Split data into train/val/test sets"
    )

    # Training command
    train_parser = subparsers.add_parser("train", help="Train AMR model")
    train_parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Training configuration file",
    )
    train_parser.add_argument(
        "--resume", type=str, help="Resume training from checkpoint"
    )

    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model"
    )
    eval_parser.add_argument(
        "--test-data", type=str, required=True, help="Path to test data (JSONL format)"
    )
    eval_parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for evaluation results",
    )
    eval_parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for evaluation"
    )

    # Prediction command
    predict_parser = subparsers.add_parser(
        "predict", help="Predict AMR for single text"
    )
    predict_parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model"
    )
    predict_parser.add_argument(
        "--text", type=str, required=True, help="Vietnamese text to convert to AMR"
    )
    predict_parser.add_argument(
        "--format",
        action="store_true",
        help="Format AMR output with proper indentation",
    )

    # File prediction command
    predict_file_parser = subparsers.add_parser(
        "predict-file", help="Predict AMR for file"
    )
    predict_file_parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model"
    )
    predict_file_parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Input file with sentences (one per line)",
    )
    predict_file_parser.add_argument(
        "--output-file", type=str, required=True, help="Output file for AMR predictions"
    )
    predict_file_parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for processing"
    )

    # Interactive command
    interactive_parser = subparsers.add_parser(
        "interactive", help="Interactive prediction mode"
    )
    interactive_parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model"
    )

    # Predict test directory
    predict_test_parser = subparsers.add_parser(
        "predict-test", help="Predict AMR for all files in test directory"
    )
    predict_test_parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model"
    )
    predict_test_parser.add_argument(
        "--test-dir",
        type=str,
        default="data/test",
        help="Directory containing test files",
    )
    predict_test_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/predictions",
        help="Directory to save predictions",
    )
    predict_test_parser.add_argument(
        "--config", type=str, help="Configuration file path"
    )
    predict_test_parser.add_argument(
        "--format", action="store_true", help="Format AMR output"
    )

    # Push model to Hugging Face
    push_model_parser = subparsers.add_parser(
        "push-model", help="Push trained model to Hugging Face Hub"
    )
    push_model_parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model"
    )
    push_model_parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Hugging Face repository name (e.g., username/model-name)",
    )
    push_model_parser.add_argument(
        "--token", type=str, help="Hugging Face token (or set HF_TOKEN env var)"
    )
    push_model_parser.add_argument(
        "--private", action="store_true", help="Make repository private"
    )
    push_model_parser.add_argument(
        "--commit-message", type=str, default="Upload AMR model", help="Commit message"
    )

    # Gradio web interface
    gradio_parser = subparsers.add_parser("gradio", help="Launch Gradio web interface")
    gradio_parser.add_argument(
        "--model-path", type=str, help="Path to local trained model"
    )
    gradio_parser.add_argument("--hf-model", type=str, help="Hugging Face model name")
    gradio_parser.add_argument(
        "--port", type=int, default=7860, help="Port to run the app"
    )
    gradio_parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to run the app"
    )
    gradio_parser.add_argument(
        "--share", action="store_true", help="Create public link"
    )

    return parser


def process_data_command(args, config: Config, logger) -> None:
    """Handle data processing command."""
    logger.info("🔄 Starting data processing...")

    # Initialize processor
    processor = AMRProcessor(logger=logger)
    data_loader = DataLoader(logger=logger)

    # Process AMR files
    output_jsonl = processor.process_amr_files(args.input_dir, args.output_dir)

    # Split data if requested
    if args.split_data:
        logger.info("Splitting data into train/val/test sets...")

        # Load processed data
        data = data_loader.load_jsonl(output_jsonl)

        # Split data
        from src.data_processing.data_loader import DataSplit

        split_config = DataSplit(
            train_ratio=config.get("data", "train_ratio", 0.8),
            val_ratio=config.get("data", "val_ratio", 0.1),
            test_ratio=config.get("data", "test_ratio", 0.1),
        )

        split_data = data_loader.split_data(data, split_config)

        # Save split data
        data_loader.save_split_data(split_data, args.output_dir)

        # Print statistics
        for split_name, split_data_list in split_data.items():
            stats = data_loader.get_data_statistics(split_data_list)
            logger.info(f"{split_name.upper()} set statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")

    logger.info("✅ Data processing completed!")


def train_command(args, config: Config, logger) -> None:
    """Handle training command."""
    logger.info("🚀 Starting model training...")

    # Load training configuration
    if hasattr(args, "config") and args.config:
        training_config = TrainingConfig.from_yaml(args.config)
    else:
        from src.training.model_config import DataConfig, ModelConfig

        training_config = TrainingConfig(
            model=ModelConfig(**config.get_model_config()),
            data=DataConfig(**config.get_data_config()),
        )

    # Handle resume from checkpoint
    if args.resume:
        training_config.model.resume_from_checkpoint = args.resume

    # Initialize trainer
    trainer = AMRTrainer(training_config, logger=logger)

    # Start training
    trainer.train()

    # Save configuration
    trainer.save_config()

    logger.info("🎉 Training completed successfully!")


def evaluate_command(args, config: Config, logger) -> None:
    """Handle evaluation command."""
    logger.info("📊 Starting model evaluation...")

    # Initialize evaluator
    evaluator = AMREvaluator(args.model_path, logger=logger)

    # Load test data
    data_loader = DataLoader(logger=logger)
    test_data = data_loader.load_jsonl(args.test_data)

    logger.info(f"Evaluating on {len(test_data)} samples...")

    # Evaluate model
    metrics = evaluator.evaluate_dataset(test_data, output_dir=args.output_dir)

    # Create evaluation report
    evaluator.create_evaluation_report(metrics, args.output_dir)

    logger.info("✅ Evaluation completed!")
    logger.info(f"Results saved to: {args.output_dir}")


def predict_command(args, config: Config, logger) -> None:
    """Handle single prediction command."""
    logger.info("🔮 Making prediction...")

    # Initialize predictor
    predictor = AMRPredictor(args.model_path, logger=logger)

    # Make prediction
    if args.format:
        result = predictor.predict_and_format(args.text)
        print(f"\n📥 Input: {result['input']}")
        print("📤 AMR Output:")
        print(result["formatted"])
    else:
        prediction = predictor.predict(args.text)
        print(f"\n📥 Input: {args.text}")
        print(f"📤 AMR: {prediction}")


def predict_file_command(args, config: Config, logger) -> None:
    """Handle file prediction command."""
    logger.info("📁 Processing file predictions...")

    # Initialize predictor
    predictor = AMRPredictor(args.model_path, logger=logger)

    # Process file
    predictor.predict_from_file(
        args.input_file, args.output_file, batch_size=args.batch_size
    )

    logger.info("✅ File processing completed!")


def interactive_command(args, config: Config, logger) -> None:
    """Handle interactive prediction command."""
    # Initialize predictor
    predictor = AMRPredictor(args.model_path, logger=logger)

    # Start interactive mode
    predictor.interactive_mode()


def predict_test_command(args, config: Config, logger) -> None:
    """Handle predict-test command - predict AMR for all files in test directory."""
    logger.info("🔮 Starting test directory prediction...")

    from pathlib import Path

    try:
        from src.inference import AMRPredictor

        # Initialize predictor
        predictor = AMRPredictor(args.model_path, logger=logger)

        test_dir = Path(args.test_dir)
        output_dir = Path(args.output_dir)

        # Check test directory exists
        if not test_dir.exists():
            logger.error(f"Test directory not found: {test_dir}")
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Find all text files in test directory
        text_files = list(test_dir.glob("*.txt"))
        if not text_files:
            logger.warning(f"No .txt files found in {test_dir}")
            return

        logger.info(f"Found {len(text_files)} test files")

        total_predictions = 0

        for text_file in text_files:
            logger.info(f"Processing: {text_file.name}")

            try:
                # Read input sentences
                with open(text_file, "r", encoding="utf-8") as f:
                    sentences = [line.strip() for line in f if line.strip()]

                if not sentences:
                    logger.warning(f"No sentences found in {text_file.name}")
                    continue

                logger.info(f"  Found {len(sentences)} sentences")

                # Predict AMR for each sentence
                predictions = []
                for i, sentence in enumerate(sentences, 1):
                    try:
                        amr = predictor.predict(sentence)
                        if args.format:
                            # Simple formatting
                            amr = amr.replace(" :", "\n   :")

                        predictions.append(f"# Sentence {i}: {sentence}")
                        predictions.append(amr)
                        predictions.append("")  # Empty line separator

                        if i % 10 == 0:
                            logger.info(f"  Processed {i}/{len(sentences)} sentences")

                    except Exception as e:
                        logger.warning(f"  Failed to predict sentence {i}: {e}")
                        predictions.append(f"# Sentence {i}: {sentence}")
                        predictions.append(f"# ERROR: {e}")
                        predictions.append("")

                # Save predictions
                output_file = output_dir / f"{text_file.stem}_predictions.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(predictions))

                logger.info(f"  Saved predictions to: {output_file}")
                total_predictions += len(sentences)

            except Exception as e:
                logger.error(f"Failed to process {text_file.name}: {e}")
                continue

        logger.info(
            f"🎉 Completed! Processed {total_predictions} sentences from {len(text_files)} files"
        )
        logger.info(f"Predictions saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Test prediction failed: {e}")
        raise


def push_model_command(args, config: Config, logger) -> None:
    """Handle push-model command - push trained model to Hugging Face Hub."""
    logger.info("🚀 Pushing model to Hugging Face Hub...")

    import os
    from pathlib import Path

    try:
        # Check if huggingface_hub is available
        try:
            from huggingface_hub import HfApi, create_repo
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError:
            logger.error(
                "huggingface_hub not installed. Install with: pip install huggingface_hub"
            )
            raise ImportError("huggingface_hub package required")

        model_path = Path(args.model_path)

        # Check model exists
        if not model_path.exists():
            logger.error(f"Model path not found: {model_path}")
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # Get token
        token = args.token or os.getenv("HF_TOKEN")
        if not token:
            logger.error(
                "Hugging Face token required. Use --token or set HF_TOKEN environment variable"
            )
            raise ValueError("Hugging Face token required")

        logger.info(f"Model path: {model_path}")
        logger.info(f"Repository: {args.repo_name}")
        logger.info(f"Private: {args.private}")

        # Initialize HF API
        api = HfApi()

        # Create repository if it doesn't exist
        try:
            logger.info("Creating repository...")
            create_repo(
                repo_id=args.repo_name, token=token, private=args.private, exist_ok=True
            )
            logger.info("✅ Repository created/verified")
        except Exception as e:
            logger.warning(f"Repository creation warning: {e}")

        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info("✅ Model and tokenizer loaded")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Push model to hub
        logger.info("Pushing model to Hub...")
        model.push_to_hub(
            repo_id=args.repo_name, token=token, commit_message=args.commit_message
        )
        logger.info("✅ Model pushed to Hub")

        # Push tokenizer to hub
        logger.info("Pushing tokenizer to Hub...")
        tokenizer.push_to_hub(
            repo_id=args.repo_name, token=token, commit_message=args.commit_message
        )
        logger.info("✅ Tokenizer pushed to Hub")

        # Create model card
        model_card_content = f"""---
language: vi
tags:
- amr
- semantic-parsing
- vietnamese
- seq2seq
license: apache-2.0
---

# Vietnamese AMR Semantic Parsing Model

This model performs Abstract Meaning Representation (AMR) semantic parsing for Vietnamese text.

## Model Description

- **Language**: Vietnamese
- **Task**: AMR Semantic Parsing
- **Base Model**: VietAI/vit5-base
- **Framework**: Transformers

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("{args.repo_name}")
model = AutoModelForSeq2SeqLM.from_pretrained("{args.repo_name}")

# Example usage
text = "Tôi yêu Việt Nam"
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=512, num_beams=4)
amr = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(amr)
```

## Training Data

This model was trained on Vietnamese AMR data for semantic parsing tasks.

## Citation

If you use this model, please cite:

```
@misc{{vietnamese-amr-model,
  title={{Vietnamese AMR Semantic Parsing Model}},
  author={{AMR Team}},
  year={{2024}},
  url={{https://huggingface.co/{args.repo_name}}}
}}
```
"""

        # Upload model card
        try:
            api.upload_file(
                path_or_fileobj=model_card_content.encode(),
                path_in_repo="README.md",
                repo_id=args.repo_name,
                token=token,
                commit_message="Add model card",
            )
            logger.info("✅ Model card uploaded")
        except Exception as e:
            logger.warning(f"Model card upload warning: {e}")

        logger.info(
            f"🎉 Model successfully pushed to: https://huggingface.co/{args.repo_name}"
        )

    except Exception as e:
        logger.error(f"Failed to push model: {e}")
        raise


def gradio_command(args, config: Config, logger) -> None:
    """Handle gradio command - launch Gradio web interface."""
    logger.info("🌐 Launching Gradio web interface...")

    try:
        # Check if gradio is available
        try:
            import gradio as gr
        except ImportError:
            logger.error("Gradio not installed. Install with: pip install gradio")
            raise ImportError("Gradio package required")

        # Import and run gradio app
        import subprocess
        import sys

        # Build command
        cmd = [sys.executable, "gradio_app.py"]

        if args.model_path:
            cmd.extend(["--model-path", args.model_path])

        if args.hf_model:
            cmd.extend(["--hf-model", args.hf_model])

        cmd.extend(["--port", str(args.port)])
        cmd.extend(["--host", args.host])

        if args.share:
            cmd.append("--share")

        logger.info(f"Starting Gradio app: {' '.join(cmd)}")

        # Run gradio app
        subprocess.run(cmd, check=True)

    except Exception as e:
        logger.error(f"Failed to launch Gradio app: {e}")
        raise


def main():
    """Main entry point."""
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        # Load configuration
        config = Config(config_file=args.config if hasattr(args, "config") else None)
        config.validate()

        # Setup logging
        log_level = "DEBUG" if args.verbose else "ERROR" if args.quiet else "INFO"
        config.set("logging", "level", log_level)

        logging_config = config.get("logging", default={})
        logger = setup_project_logging(logging_config, project_name="amr_project")

        # Setup directories
        config.setup_directories()

        logger.info(f"🚀 Starting AMR Semantic Parsing - Command: {args.command}")

        # Route to appropriate command handler
        if args.command == "process-data":
            process_data_command(args, config, logger)
        elif args.command == "train":
            train_command(args, config, logger)
        elif args.command == "evaluate":
            evaluate_command(args, config, logger)
        elif args.command == "predict":
            predict_command(args, config, logger)
        elif args.command == "predict-file":
            predict_file_command(args, config, logger)
        elif args.command == "interactive":
            interactive_command(args, config, logger)
        elif args.command == "predict-test":
            predict_test_command(args, config, logger)
        elif args.command == "push-model":
            push_model_command(args, config, logger)
        elif args.command == "gradio":
            gradio_command(args, config, logger)
        else:
            logger.error(f"Unknown command: {args.command}")
            parser.print_help()
            return

        logger.info("🎉 Operation completed successfully!")

    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
