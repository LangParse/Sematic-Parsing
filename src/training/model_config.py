"""
Model Configuration Module
==========================

This module handles model configuration and hyperparameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ModelConfig:
    """Configuration class for model parameters."""

    # Model settings
    model_name: str = "VietAI/vit5-base"
    max_length: int = 512

    # Training hyperparameters
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 500
    weight_decay: float = 0.01

    # Optimization settings
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Scheduler settings
    lr_scheduler_type: str = "linear"

    # Evaluation settings
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Logging settings
    logging_steps: int = 100
    report_to: str = "wandb"
    run_name: Optional[str] = None

    # Data settings
    max_input_length: int = 512
    max_output_length: int = 512

    # Paths
    output_dir: str = "./models/amr_model"
    logging_dir: str = "./logs"
    cache_dir: Optional[str] = None

    # Other settings
    seed: int = 42
    fp16: bool = True
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        """Load config from YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def save_yaml(self, yaml_path: str) -> None:
        """Save config to YAML file."""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def update(self, **kwargs) -> "ModelConfig":
        """Update config with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        return self

    def validate(self) -> None:
        """Validate configuration parameters."""
        # Convert string values to appropriate types
        learning_rate = (
            float(self.learning_rate)
            if isinstance(self.learning_rate, str)
            else self.learning_rate
        )
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        num_epochs = (
            int(self.num_train_epochs)
            if isinstance(self.num_train_epochs, str)
            else self.num_train_epochs
        )
        if num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")

        batch_size = (
            int(self.per_device_train_batch_size)
            if isinstance(self.per_device_train_batch_size, str)
            else self.per_device_train_batch_size
        )
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")

        max_length = (
            int(self.max_length)
            if isinstance(self.max_length, str)
            else self.max_length
        )
        if max_length <= 0:
            raise ValueError("Max length must be positive")

        warmup_steps = (
            int(self.warmup_steps)
            if isinstance(self.warmup_steps, str)
            else self.warmup_steps
        )
        if warmup_steps < 0:
            raise ValueError("Warmup steps must be non-negative")

        weight_decay = (
            float(self.weight_decay)
            if isinstance(self.weight_decay, str)
            else self.weight_decay
        )
        if not 0 <= weight_decay <= 1:
            raise ValueError("Weight decay must be between 0 and 1")

        if self.evaluation_strategy not in ["no", "steps", "epoch"]:
            raise ValueError("Invalid evaluation strategy")

        if self.lr_scheduler_type not in [
            "linear",
            "cosine",
            "constant",
            "constant_with_warmup",
        ]:
            raise ValueError("Invalid learning rate scheduler type")


@dataclass
class DataConfig:
    """Configuration class for data processing parameters."""

    # Data paths
    train_data_path: str = "data/train"
    test_data_path: str = "data/test"
    processed_data_dir: str = "data/processed"

    # Data split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Data processing settings
    max_samples: Optional[int] = None
    shuffle_data: bool = True
    random_seed: int = 42

    # Filtering settings
    min_input_length: int = 1
    max_input_length: int = 512
    min_output_length: int = 1
    max_output_length: int = 512

    def validate(self) -> None:
        """Validate data configuration."""
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError("Data split ratios must sum to 1.0")

        if self.min_input_length < 1:
            raise ValueError("Minimum input length must be at least 1")

        if self.max_input_length < self.min_input_length:
            raise ValueError("Maximum input length must be >= minimum input length")

        if self.min_output_length < 1:
            raise ValueError("Minimum output length must be at least 1")

        if self.max_output_length < self.min_output_length:
            raise ValueError("Maximum output length must be >= minimum output length")


@dataclass
class TrainingConfig:
    """Combined configuration for training."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load complete training config from YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        model_config = ModelConfig.from_dict(config_dict.get("model", {}))
        data_config = DataConfig(**config_dict.get("data", {}))

        return cls(model=model_config, data=data_config)

    def save_yaml(self, yaml_path: str) -> None:
        """Save complete training config to YAML file."""
        config_dict = {"model": self.model.to_dict(), "data": self.data.__dict__}

        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    def validate(self) -> None:
        """Validate all configurations."""
        self.model.validate()
        self.data.validate()
