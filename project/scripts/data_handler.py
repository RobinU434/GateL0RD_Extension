import logging
from pathlib import Path

import numpy as np

from project.datasets.data_handler import load_data, split_data, save_data
from project.utils.logger import setup_logging


def split_and_save_dataset(
    data_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    output_dir: Path = None,
    random_seed: int = None,
    log_level: str = "INFO",
):
    """Split dataset into train, val, test sets based on given ratios.

    Args:
        data_dir (Path): Directory containing the dataset files.
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
        test_ratio (float): Proportion of data to use for testing.
        output_dir (Path): Directory to save the split datasets.
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.

    Raises:
        ValueError: If any of the ratios are invalid or do not sum to 1.
        ValueError: If the data directory is empty.
    """
    logger = setup_logging(log_level)
    data_dir = Path(data_dir) if not isinstance(data_dir, Path) else data_dir
    output_dir = (
        Path(output_dir)
        if output_dir and not isinstance(output_dir, Path)
        else output_dir
    )
    
    if random_seed is not None:
        np.random.seed(random_seed)

    if (
        not (0 < train_ratio < 1)
        or not (0 <= val_ratio < 1)
        or not (0 <= test_ratio < 1)
    ):
        raise ValueError("Ratios must be between 0 and 1")
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6:
        raise ValueError("Ratios must sum to 1")

    logger.info(f"Loading data from {data_dir}")
    data = load_data(data_dir)
    logger.info(f"Loaded data from {len(data)} files")

    # for each file in the dictionary split the data into train, val, test sets
    logger.info("Splitting data...")
    data_splits = {}
    for file_name, file_data in data.items():
        if random_seed is not None:
            np.random.shuffle(file_data)
        train_data, val_data, test_data = split_data(
            file_data, train_ratio, val_ratio, test_ratio
        )
        data_splits[file_name] = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }

    logger.info(
        f"Data split into train/val/test with ratios {train_ratio}/{val_ratio}/{test_ratio}"
    )
    if output_dir is None:
        output_dir = data_dir
    for file_name, splits in data_splits.items():
        base_name = Path(file_name).stem
        save_data(splits["train"], output_dir / "train" / f"{base_name}.npy")
        save_data(splits["val"], output_dir / "val" / f"{base_name}.npy")
        save_data(splits["test"], output_dir / "test" / f"{base_name}.npy")
