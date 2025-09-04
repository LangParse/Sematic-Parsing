"""
Configuration Management Module
==============================

This module handles configuration loading and management with YAML files.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class PathConfig:
    """Configuration for file paths."""

    # Data paths
    data_dir: str = "data"
    train_dir: str = "data/train"
    test_dir: str = "data/test"
    processed_dir: str = "data/processed"

    # Model paths
    model_dir: str = "models"
    output_dir: str = "models/amr_model"
    checkpoint_dir: str = "models/checkpoints"

    # Log paths
    log_dir: str = "logs"

    # Cache paths
    cache_dir: str = ".cache"

    def resolve_paths(self, base_dir: Optional[str] = None) -> "PathConfig":
        """Resolve all paths relative to base directory."""
        if base_dir is None:
            base_dir = os.getcwd()

        base_path = Path(base_dir)

        # Resolve all paths
        for field_name in self.__dataclass_fields__:
            current_path = getattr(self, field_name)
            if isinstance(current_path, str):
                resolved_path = base_path / current_path
                setattr(self, field_name, str(resolved_path))

        return self

    def create_directories(self) -> None:
        """Create all directories if they don't exist."""
        for field_name in self.__dataclass_fields__:
            path = getattr(self, field_name)
            if isinstance(path, str):
                Path(path).mkdir(parents=True, exist_ok=True)


class Config:
    """
    Main configuration manager for the AMR project.

    This class handles:
    - Loading configuration from YAML files
    - Environment variable overrides
    - Path resolution and management
    - Configuration validation
    """

    def __init__(
        self, config_file: Optional[str] = None, base_dir: Optional[str] = None
    ):
        """
        Initialize configuration manager.

        Args:
            config_file: Path to YAML configuration file
            base_dir: Base directory for resolving relative paths
        """
        self.base_dir = base_dir or os.getcwd()
        self.config_data = {}
        self.paths = PathConfig()

        # Load default configuration
        self._load_defaults()

        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)

        # Apply environment overrides
        self._apply_env_overrides()

        # Resolve paths
        self.paths = self.paths.resolve_paths(self.base_dir)

    def _load_defaults(self) -> None:
        """Load default configuration values."""
        self.config_data = {
            "model": {
                "name": "VietAI/vit5-base",
                "max_length": 512,
                "batch_size": 8,
                "learning_rate": 5e-5,
                "num_epochs": 3,
                "warmup_steps": 500,
                "weight_decay": 0.01,
                "fp16": True,
                "gradient_accumulation_steps": 1,
                "eval_steps": 500,
                "save_steps": 500,
                "logging_steps": 100,
                "early_stopping_patience": 3,
                "num_beams": 4,
                "do_sample": False,
                "temperature": 1.0,
            },
            "data": {
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
                "max_samples": None,
                "shuffle": True,
                "random_seed": 42,
                "max_input_length": 512,
                "max_output_length": 512,
                "min_input_length": 1,
                "min_output_length": 1,
            },
            "training": {
                "use_wandb": False,
                "wandb_project": "amr-semantic-parsing",
                "run_name": None,
                "resume_from_checkpoint": None,
                "save_total_limit": 3,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "evaluation_strategy": "steps",
                "lr_scheduler_type": "linear",
                "dataloader_num_workers": 4,
            },
            "evaluation": {
                "batch_size": 16,
                "metrics": ["bleu", "rouge", "meteor", "exact_match"],
                "generate_report": True,
                "save_predictions": True,
            },
            "inference": {
                "batch_size": 8,
                "max_length": 512,
                "num_beams": 4,
                "do_sample": False,
                "temperature": 1.0,
                "format_output": True,
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_logging": True,
                "console_logging": True,
            },
        }

    def load_from_file(self, config_file: str) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_file: Path to YAML configuration file
        """
        config_path = Path(config_file)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f)

            # Deep merge with existing configuration
            self._deep_merge(self.config_data, file_config)

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")

    def _deep_merge(
        self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]
    ) -> None:
        """Deep merge two dictionaries."""
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_mappings = {
            "AMR_MODEL_NAME": ("model", "name"),
            "AMR_BATCH_SIZE": ("model", "batch_size"),
            "AMR_LEARNING_RATE": ("model", "learning_rate"),
            "AMR_NUM_EPOCHS": ("model", "num_epochs"),
            "AMR_MAX_LENGTH": ("model", "max_length"),
            "AMR_USE_WANDB": ("training", "use_wandb"),
            "AMR_WANDB_PROJECT": ("training", "wandb_project"),
            "AMR_LOG_LEVEL": ("logging", "level"),
            "AMR_DATA_DIR": ("paths", "data_dir"),
            "AMR_MODEL_DIR": ("paths", "model_dir"),
            "AMR_LOG_DIR": ("paths", "log_dir"),
        }

        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if section not in self.config_data:
                    self.config_data[section] = {}

                # Type conversion
                if key in ["batch_size", "num_epochs", "max_length", "warmup_steps"]:
                    value = int(value)
                elif key in ["learning_rate", "weight_decay", "temperature"]:
                    value = float(value)
                elif key in ["fp16", "use_wandb", "shuffle", "do_sample"]:
                    value = value.lower() in ("true", "1", "yes", "on")

                self.config_data[section][key] = value

    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            section: Configuration section
            key: Configuration key (optional)
            default: Default value if not found

        Returns:
            Configuration value
        """
        if key is None:
            return self.config_data.get(section, default)

        section_data = self.config_data.get(section, {})
        return section_data.get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
        """
        if section not in self.config_data:
            self.config_data[section] = {}

        self.config_data[section][key] = value

    def save_to_file(self, config_file: str) -> None:
        """
        Save current configuration to YAML file.

        Args:
            config_file: Path to save configuration
        """
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    self.config_data,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    indent=2,
                )
        except Exception as e:
            raise RuntimeError(f"Failed to save configuration: {e}")

    def validate(self) -> None:
        """Validate configuration values."""
        # Validate model configuration
        model_config = self.get("model", default={})

        # Convert and validate batch_size
        batch_size = model_config.get("batch_size", 1)
        if isinstance(batch_size, str):
            batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")

        # Convert and validate learning_rate
        learning_rate = model_config.get("learning_rate", 5e-5)
        if isinstance(learning_rate, str):
            learning_rate = float(learning_rate)
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        # Convert and validate num_epochs
        num_epochs = model_config.get("num_epochs", 1)
        if isinstance(num_epochs, str):
            num_epochs = int(num_epochs)
        if num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")

        # Validate data configuration
        data_config = self.get("data", default={})

        # Convert and validate ratios
        train_ratio = data_config.get("train_ratio", 0.8)
        if isinstance(train_ratio, str):
            train_ratio = float(train_ratio)

        val_ratio = data_config.get("val_ratio", 0.1)
        if isinstance(val_ratio, str):
            val_ratio = float(val_ratio)

        test_ratio = data_config.get("test_ratio", 0.1)
        if isinstance(test_ratio, str):
            test_ratio = float(test_ratio)

        ratios = [train_ratio, val_ratio, test_ratio]

        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("Data split ratios must sum to 1.0")

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get("model", default={})

    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.get("data", default={})

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get("training", default={})

    def get_paths(self) -> PathConfig:
        """Get path configuration."""
        return self.paths

    def setup_directories(self) -> None:
        """Create all necessary directories."""
        self.paths.create_directories()

    def __str__(self) -> str:
        """String representation of configuration."""
        return yaml.dump(self.config_data, default_flow_style=False, allow_unicode=True)
