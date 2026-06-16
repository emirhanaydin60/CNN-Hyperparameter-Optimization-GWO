from dataclasses import asdict, dataclass, field
from typing import Optional
import json


@dataclass
class ExperimentConfig:
    dataset: str = "fashionmnist"
    population_size: int = 6
    iteration_count: int = 10
    search_epochs: int = 6
    final_epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    subset_size: Optional[int] = None
    random_seed: int = 42
    patience: int = 5
    runs: int = 1
    monitor_ratio: float = 0.1
    train_size: int = 55000
    val_size: int = 5000
    num_workers: int = 0
    results_dir: str = "results"
    data_dir: str = "data"
    search_space: dict = field(
        default_factory=lambda: {
            "filter_sizes": [3, 5, 7],
            "filter_counts": [8, 16, 32, 64],
            "dilations": [1, 2, 3],
            "final_neurons": [64, 128, 256, 512],
            "dropout_min": 0.0,
            "dropout_max": 0.5,
        }
    )


def load_config(config_path: Optional[str] = None) -> ExperimentConfig:
    config = ExperimentConfig()
    if config_path:
        with open(config_path, "r", encoding="utf-8") as file_handle:
            overrides = json.load(file_handle)
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    return config


def save_config(config: ExperimentConfig, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(asdict(config), file_handle, indent=2, ensure_ascii=False)


def config_to_dict(config: ExperimentConfig) -> dict:
    return asdict(config)
