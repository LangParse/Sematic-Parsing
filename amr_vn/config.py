
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class Paths:
    project_root: Path
    train_files: List[Path] = field(default_factory=list)
    test_file: Optional[Path] = None
    work_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    prediction_dir: Path = field(init=False)
    json_path: Path = field(init=False)

    def __post_init__(self):
        self.work_dir = self.project_root / "outputs"
        self.model_dir = self.work_dir / "models"
        self.prediction_dir = self.work_dir / "predictions"
        self.json_path = self.work_dir / "amr_data.json"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class TrainConfig:
    model_name: str = "VietAI/vit5-base"
    max_length: int = 512
    batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 5e-5
    save_total_limit: int = 2
    fp16: bool = True  # set False if training on CPU
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    seed: int = 42
