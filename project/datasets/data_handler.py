import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np


def split_data(
    data: np.ndarray, train_ratio: float, val_ratio: float, test_ratio: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into training, validation, and test sets.

    Args:
        data: The input data array.
        train_ratio: Proportion of data to use for training.
        val_ratio: Proportion of data to use for validation.
        test_ratio: Proportion of data to use for testing.

    Returns:
        A tuple containing the training, validation, and test sets.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1."

    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]
    test_data = data[train_size + val_size :]

    return train_data, val_data, test_data


def load_data(data_path: Path, data_key: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load data from file(s).

    Args:
        data_path (str): Path to the data file or directory.
        data_key (Optional[str], optional): Key to load from .npz file. Defaults to None.

    Raises:
        ValueError: If the data path is invalid or unsupported.

    Returns:
        Dict[str, np.ndarray]: Loaded data.
    """
    if data_path.is_file():
        # Single file
        if data_path.suffix == ".npy":
            data = {data_path: np.load(data_path)}
        elif data_path.suffix == ".npz":
            loaded = np.load(data_path)
            if data_key:
                data = loaded[data_key]
            else:
                # Use first available key
                data = loaded[list(loaded.keys())[0]]
            data = {data_path: data}
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

    elif data_path.is_dir():
        # Directory with multiple files
        data_files = [f for f in data_path.iterdir() if f.suffix == ".npy"]
        if not data_files:
            raise ValueError(f"No .npy files found in {data_path}")

        data_list = []
        for file in sorted(data_files):
            file_data = np.load(file)
            data_list.append(file_data)

        data = dict(zip(data_files, data_list))
    else:
        raise ValueError(f"Data path not found: {data_path}")

    return data


def save_data(data: np.ndarray, save_path: Path, data_key: Optional[str] = None):
    """Save data to file.

    Args:
        data (np.ndarray): The data to save.
        save_path (Path): The path to save the data.
        data_key (Optional[str], optional): The key to use for saving. Defaults to None.

    Raises:
        ValueError: If the save path is invalid or unsupported.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_path.suffix == ".npy":
        np.save(save_path, data)
    elif save_path.suffix == ".npz":
        if data_key is None:
            data_key = "data"
        np.savez_compressed(save_path, **{data_key: data})
    else:
        raise ValueError(f"Unsupported file format: {save_path}")
