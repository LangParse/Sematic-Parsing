"""
AMR Trainer Module
=================

This module handles model training with advanced configuration management and logging.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    TrainerCallback,
)

from ..data_processing import DataLoader as AMRDataLoader
from ..tokenization import ViT5Tokenizer
from ..utils import setup_logger
from .model_config import TrainingConfig


class AMRDataset(Dataset):
    """Custom dataset class for AMR data."""

    def __init__(self, data: List[Dict[str, Any]], tokenizer: ViT5Tokenizer):
        """
        Initialize dataset.

        Args:
            data: List of tokenized data samples
            tokenizer: ViT5Tokenizer instance
        """
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index."""
        item = self.data[idx]

        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


class ProgressBarCallback(TrainerCallback):
    """Custom callback with progress bar and enhanced logging."""

    def __init__(self, logger: logging.Logger):
        """Initialize callback with logger."""
        self.logger = logger
        self.progress_bar = None
        self.epoch_bar = None
        self.current_epoch = 0
        self.total_epochs = 0

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Initialize progress bars at training start."""
        self.total_epochs = int(args.num_train_epochs)
        self.current_epoch = 0

        # Create epoch progress bar
        self.epoch_bar = tqdm(
            total=self.total_epochs,
            desc="🚀 Training Progress",
            unit="epoch",
            position=0,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        self.logger.info(f"🎯 Starting training for {self.total_epochs} epochs")

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        """Initialize step progress bar for each epoch."""
        if hasattr(state, "max_steps") and state.max_steps > 0:
            steps_per_epoch = state.max_steps // self.total_epochs
        else:
            # Estimate steps per epoch
            steps_per_epoch = (
                len(state.train_dataloader)
                if hasattr(state, "train_dataloader")
                else 100
            )

        # Create step progress bar
        self.progress_bar = tqdm(
            total=steps_per_epoch,
            desc=f"📚 Epoch {self.current_epoch + 1}/{self.total_epochs}",
            unit="step",
            position=1,
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]",
        )

    def on_step_end(self, args, state, control, model=None, logs=None, **kwargs):
        """Update progress bar on each step."""
        if self.progress_bar:
            # Update postfix with current metrics
            postfix = {}
            if logs:
                for key, value in logs.items():
                    if isinstance(value, (int, float)) and key in [
                        "train_loss",
                        "learning_rate",
                    ]:
                        if key == "train_loss":
                            postfix["loss"] = f"{value:.4f}"
                        elif key == "learning_rate":
                            postfix["lr"] = f"{value:.2e}"

            self.progress_bar.set_postfix(postfix)
            self.progress_bar.update(1)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log training metrics."""
        if logs:
            step = state.global_step
            # Only log important metrics to avoid spam
            important_metrics = ["train_loss", "eval_loss", "learning_rate"]
            for key, value in logs.items():
                if isinstance(value, (int, float)) and key in important_metrics:
                    self.logger.debug(f"Step {step} - {key}: {value:.4f}")

    def on_epoch_end(self, args, state, control, model=None, logs=None, **kwargs):
        """Update progress bars at epoch end."""
        self.current_epoch += 1

        # Close step progress bar
        if self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = None

        # Update epoch progress bar
        if self.epoch_bar:
            epoch_info = f"Epoch {self.current_epoch}"
            if logs:
                if "eval_loss" in logs:
                    epoch_info += f" | Val Loss: {logs['eval_loss']:.4f}"
                if "train_loss" in logs:
                    epoch_info += f" | Train Loss: {logs['train_loss']:.4f}"

            self.epoch_bar.set_description(f"✅ {epoch_info}")
            self.epoch_bar.update(1)

        self.logger.info(f"✅ Completed epoch {self.current_epoch}/{self.total_epochs}")

    def on_train_end(self, args, state, control, model=None, logs=None, **kwargs):
        """Clean up progress bars at training end."""
        if self.progress_bar:
            self.progress_bar.close()
        if self.epoch_bar:
            self.epoch_bar.close()

        self.logger.info("🎉 Training completed successfully!")


class AMRTrainer:
    """
    Advanced trainer for AMR models with configuration management.

    This class provides:
    - Flexible configuration management
    - Advanced logging and monitoring
    - Model checkpointing and evaluation
    - Integration with Weights & Biases
    - Memory optimization
    """

    def __init__(self, config: TrainingConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize AMR trainer.

        Args:
            config: Training configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or setup_logger(__name__)

        # Validate configuration
        self.config.validate()

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.data_loader = AMRDataLoader(logger=self.logger)

        # Setup directories
        self._setup_directories()

        self.logger.info("AMR Trainer initialized successfully")

    def _setup_directories(self) -> None:
        """Create necessary directories."""
        Path(self.config.model.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.model.logging_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.data.processed_data_dir).mkdir(parents=True, exist_ok=True)

    def setup_model_and_tokenizer(self) -> None:
        """Initialize model and tokenizer."""
        self.logger.info(f"Loading model and tokenizer: {self.config.model.model_name}")

        # Initialize tokenizer
        self.tokenizer = ViT5Tokenizer(
            model_name=self.config.model.model_name,
            max_length=self.config.model.max_length,
            logger=self.logger,
        )

        # Initialize model
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.config.model.model_name, cache_dir=self.config.model.cache_dir
        )

        self.logger.info("✅ Model and tokenizer loaded successfully")

    def prepare_data(self) -> Tuple[HFDataset, HFDataset]:
        """
        Prepare training and validation datasets.

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        self.logger.info("Preparing training data...")

        # Load and process data
        train_data = self.data_loader.load_jsonl(
            os.path.join(self.config.data.processed_data_dir, "train.jsonl")
        )
        val_data = self.data_loader.load_jsonl(
            os.path.join(self.config.data.processed_data_dir, "val.jsonl")
        )

        # Filter by length if specified
        if self.config.data.max_input_length or self.config.data.max_output_length:
            train_data = self.data_loader.filter_by_length(
                train_data,
                max_input_length=self.config.data.max_input_length,
                max_output_length=self.config.data.max_output_length,
            )
            val_data = self.data_loader.filter_by_length(
                val_data,
                max_input_length=self.config.data.max_input_length,
                max_output_length=self.config.data.max_output_length,
            )

        # Limit samples if specified
        if self.config.data.max_samples:
            train_data = train_data[: self.config.data.max_samples]
            val_data = val_data[
                : min(len(val_data), self.config.data.max_samples // 10)
            ]

        self.logger.info(f"Training samples: {len(train_data)}")
        self.logger.info(f"Validation samples: {len(val_data)}")

        # Convert to HuggingFace datasets
        train_dataset = HFDataset.from_list(train_data)
        eval_dataset = HFDataset.from_list(val_data)

        # Tokenize datasets
        train_dataset = train_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )

        eval_dataset = eval_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
        )

        return train_dataset, eval_dataset

    def _tokenize_function(self, examples):
        """Tokenize function for dataset mapping."""
        inputs = examples["input"]
        targets = examples["output"]

        model_inputs = self.tokenizer.tokenizer(
            inputs,
            max_length=self.config.model.max_input_length,
            truncation=True,
            padding=True,
        )

        with self.tokenizer.tokenizer.as_target_tokenizer():
            labels = self.tokenizer.tokenizer(
                targets,
                max_length=self.config.model.max_output_length,
                truncation=True,
                padding=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def setup_training_arguments(self) -> Seq2SeqTrainingArguments:
        """Setup training arguments from configuration."""
        return Seq2SeqTrainingArguments(
            output_dir=self.config.model.output_dir,
            num_train_epochs=self.config.model.num_train_epochs,
            per_device_train_batch_size=self.config.model.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.model.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.model.gradient_accumulation_steps,
            warmup_steps=self.config.model.warmup_steps,
            weight_decay=self.config.model.weight_decay,
            learning_rate=self.config.model.learning_rate,
            adam_epsilon=self.config.model.adam_epsilon,
            max_grad_norm=self.config.model.max_grad_norm,
            lr_scheduler_type=self.config.model.lr_scheduler_type,
            eval_strategy=self.config.model.evaluation_strategy,
            eval_steps=self.config.model.eval_steps,
            save_steps=self.config.model.save_steps,
            save_total_limit=self.config.model.save_total_limit,
            load_best_model_at_end=self.config.model.load_best_model_at_end,
            metric_for_best_model=self.config.model.metric_for_best_model,
            greater_is_better=self.config.model.greater_is_better,
            logging_steps=self.config.model.logging_steps,
            logging_dir=self.config.model.logging_dir,
            report_to=getattr(self.config.training, "report_to", "none")
            if hasattr(self.config, "training")
            else "none",
            run_name=self.config.model.run_name,
            seed=self.config.model.seed,
            fp16=self.config.model.fp16,
            dataloader_num_workers=self.config.model.dataloader_num_workers,
            remove_unused_columns=self.config.model.remove_unused_columns,
            predict_with_generate=True,
            generation_max_length=self.config.model.max_output_length,
        )

    def setup_trainer(self, train_dataset: HFDataset, eval_dataset: HFDataset) -> None:
        """Setup the Seq2Seq trainer."""
        training_args = self.setup_training_arguments()

        # Setup callbacks
        callbacks = [ProgressBarCallback(self.logger)]

        early_stopping_patience = (
            int(self.config.model.early_stopping_patience)
            if isinstance(self.config.model.early_stopping_patience, str)
            else self.config.model.early_stopping_patience
        )
        if early_stopping_patience > 0:
            early_stopping_threshold = (
                float(self.config.model.early_stopping_threshold)
                if isinstance(self.config.model.early_stopping_threshold, str)
                else self.config.model.early_stopping_threshold
            )
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_threshold=early_stopping_threshold,
                )
            )

        # Initialize trainer
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer.tokenizer,
            callbacks=callbacks,
        )

        self.logger.info("✅ Trainer setup completed")

    def train(self) -> None:
        """Execute the complete training pipeline."""
        self.logger.info("🚀 Starting AMR model training...")

        # Setup model and tokenizer
        self.setup_model_and_tokenizer()

        # Prepare data
        train_dataset, eval_dataset = self.prepare_data()

        # Setup trainer
        self.setup_trainer(train_dataset, eval_dataset)

        # Initialize wandb if configured
        if self.config.model.report_to == "wandb":
            self._setup_wandb()

        # Start training
        self.logger.info("Starting training process...")
        train_result = self.trainer.train()

        # Save final model
        self.trainer.save_model()
        self.tokenizer.tokenizer.save_pretrained(self.config.model.output_dir)

        # Save training metrics
        self._save_training_metrics(train_result)

        self.logger.info("🎉 Training completed successfully!")

    def _setup_wandb(self) -> None:
        """Setup Weights & Biases logging."""
        if not WANDB_AVAILABLE:
            self.logger.warning("Weights & Biases not available - skipping")
            return

        try:
            wandb.init(
                project="amr-semantic-parsing",
                name=self.config.model.run_name or "amr-training",
                config=self.config.model.to_dict(),
            )
            self.logger.info("✅ Weights & Biases initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")

    def _save_training_metrics(self, train_result) -> None:
        """Save training metrics to file."""
        metrics_file = Path(self.config.model.output_dir) / "training_metrics.json"

        metrics = {
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get(
                "train_samples_per_second", 0
            ),
            "train_steps_per_second": train_result.metrics.get(
                "train_steps_per_second", 0
            ),
            "total_flos": train_result.metrics.get("total_flos", 0),
            "train_loss": train_result.metrics.get("train_loss", 0),
        }

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Training metrics saved to: {metrics_file}")

    def evaluate(self, eval_dataset: Optional[HFDataset] = None) -> Dict[str, float]:
        """
        Evaluate the trained model.

        Args:
            eval_dataset: Optional evaluation dataset

        Returns:
            Dictionary containing evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call train() first.")

        self.logger.info("Evaluating model...")

        if eval_dataset is None:
            eval_result = self.trainer.evaluate()
        else:
            eval_result = self.trainer.evaluate(eval_dataset=eval_dataset)

        self.logger.info("Evaluation completed")
        for key, value in eval_result.items():
            self.logger.info(f"{key}: {value:.4f}")

        return eval_result

    def save_config(self) -> None:
        """Save training configuration to output directory."""
        config_file = Path(self.config.model.output_dir) / "training_config.yaml"
        self.config.save_yaml(str(config_file))
        self.logger.info(f"Configuration saved to: {config_file}")
