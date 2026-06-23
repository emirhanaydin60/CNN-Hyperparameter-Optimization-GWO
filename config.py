from dataclasses import asdict, dataclass, field
from typing import Optional
import json

OPTIMIZERS = ["gwo", "pso", "woa", "rao"]


@dataclass
class ExperimentConfig:
    dataset: str = "cifar10"
    model_name: str = "hybrid_cnn"
    optimizers: list[str] = field(default_factory=lambda: OPTIMIZERS.copy())
    population_size: int = 8
    iteration_count: int = 15
    search_epochs: int = 6
    final_epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    subset_size: Optional[int] = None
    random_seed: int = 42
    patience: int = 10
    lr_reduce_factor: float = 0.5
    lr_reduce_patience: int = 4
    runs: int = 3
    monitor_ratio: float = 0.1
    train_size: int = 45000
    val_size: int = 5000
    num_workers: int = 0
    results_dir: str = "results"
    data_dir: str = "data"
    search_space: dict = field(
        default_factory=lambda: {
            # Shared convolution path only: stem + residual blocks.
            "shared_conv_kernel_sizes": [3, 5, 7],
            "base_filters": [16, 32, 64, 128, 256],
            "dilations": [1, 2, 3],
            "final_neurons": [128, 256, 512],
            "se_ratios": [4, 8, 16],
            "batch_sizes": [32, 64, 128],
            "learning_rate_min": 0.0001,
            "learning_rate_max": 0.01,
            "dropout_min": 0.10,
            "dropout_max": 0.7,
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
