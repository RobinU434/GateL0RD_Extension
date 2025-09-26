from pathlib import Path
from omegaconf import DictConfig
from project.datasets.timeseries import TimeSeriesDataModule


def create_data_module(config: DictConfig) -> TimeSeriesDataModule:
    """Create and setup data module."""
    # Determine data path based on dataset name
    if config.data.dataset_name == "BilliardBall":
        data_path = Path(config.data.data_dir) / "BilliardBall"
    elif config.data.dataset_name == "FetchPickAndPlace":
        data_path = Path(config.data.data_dir) / "FetchPickAndPlace"
    else:
        data_path = Path(config.data.data_dir) / config.data.dataset_name

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    return TimeSeriesDataModule(
        data_dir=str(data_path),
        batch_size=config.data.batch_size,
        sequence_length=config.data.sequence_length,
        prediction_steps=config.data.prediction_steps,
        overlap_ratio=config.data.overlap_ratio,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        normalize=config.data.normalize,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
