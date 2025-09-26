"""
Time Series Dataset for GateL0RD training.

Implements dataset loading for continuous time series data with discrete events,
following the setup from the GateL0RD paper (https://arxiv.org/pdf/2110.15949).
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class TimeSeriesDataset(Dataset):
    """
    Dataset for time series prediction with discrete events embedded in continuous streams.

    This dataset handles the specific format used in GateL0RD experiments:
    - Continuous data streams (e.g., robot positions, ball trajectories)
    - Discrete events within the stream (e.g., grasping, bouncing)
    - Variable sequence lengths
    - Multi-step prediction tasks
    """

    def __init__(
        self,
        data_path: str,
        sequence_length: int = 50,
        prediction_steps: int = 1,
        overlap_ratio: float = 0.5,
        normalize: bool = True,
        data_key: Optional[str] = None,
    ):
        """
        Initialize the TimeSeriesDataset.

        Args:
            data_path: Path to data file (.npy) or directory containing .npy files
            sequence_length: Length of input sequences
            prediction_steps: Number of steps to predict ahead
            overlap_ratio: Ratio of overlap between consecutive sequences (0.0 to 1.0)
            normalize: Whether to normalize the data
            data_key: If loading from .npz files, specify which key to use
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.overlap_ratio = overlap_ratio
        self.normalize = normalize
        self.data_key = data_key

        self.data = self._load_data()
        self.sequences = self._create_sequences()

        if self.normalize:
            self.data_mean = np.mean(self.data, axis=0, keepdims=True)
            self.data_std = np.std(self.data, axis=0, keepdims=True) + 1e-8
            self.data = (self.data - self.data_mean) / self.data_std

    def _load_data(self) -> np.ndarray:
        """Load data from file(s)."""
        if os.path.isfile(self.data_path):
            # Single file
            if self.data_path.endswith(".npy"):
                data = np.load(self.data_path)
            elif self.data_path.endswith(".npz"):
                loaded = np.load(self.data_path)
                if self.data_key:
                    data = loaded[self.data_key]
                else:
                    # Use first available key
                    data = loaded[list(loaded.keys())[0]]
            else:
                raise ValueError(f"Unsupported file format: {self.data_path}")

        elif os.path.isdir(self.data_path):
            # Directory with multiple files
            data_files = [f for f in os.listdir(self.data_path) if f.endswith(".npy")]
            if not data_files:
                raise ValueError(f"No .npy files found in {self.data_path}")

            data_list = []
            for file in sorted(data_files):
                file_data = np.load(os.path.join(self.data_path, file))
                print(file_data.shape)
                data_list.append(file_data)

            # Concatenate along batch axis
            data = np.concatenate(data_list, axis=0)
        else:
            raise ValueError(f"Data path not found: {self.data_path}")

        return data.astype(np.float32)

    def _create_sequences(self) -> List[Tuple[int, int]]:
        """Create sequence start and end indices."""
        sequences = []
        total_length = len(self.data)
        step_size = max(1, int(self.sequence_length * (1 - self.overlap_ratio)))

        for start_idx in range(
            0,
            total_length - self.sequence_length - self.prediction_steps + 1,
            step_size,
        ):
            end_idx = start_idx + self.sequence_length
            sequences.append((start_idx, end_idx))

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sequence.

        Returns:
            Dictionary containing:
                - 'input': Input sequence of shape (sequence_length, feature_dim)
                - 'target': Target sequence of shape (prediction_steps, feature_dim)
                - 'sequence_mask': Mask for valid timesteps (all 1s for this implementation)
        """
        start_idx, end_idx = self.sequences[idx]

        # Input sequence
        input_seq = self.data[start_idx:end_idx]

        # Target sequence (next prediction_steps timesteps)
        target_seq = self.data[end_idx : end_idx + self.prediction_steps]

        # Create mask (all valid for this simple implementation)
        sequence_mask = np.ones((self.sequence_length,), dtype=np.float32)

        return {
            "input": torch.from_numpy(input_seq),
            "target": torch.from_numpy(target_seq),
            "sequence_mask": torch.from_numpy(sequence_mask),
        }

    def get_feature_dim(self) -> int:
        """Get the feature dimension of the data."""
        return self.data.shape[-1]

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize data back to original scale."""
        if not self.normalize:
            return data

        data_mean = torch.from_numpy(self.data_mean).to(data.device)
        data_std = torch.from_numpy(self.data_std).to(data.device)

        return data * data_std + data_mean


class TimeSeriesDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for time series data.

    Handles data loading, splitting, and DataLoader creation for train/val/test sets.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        sequence_length: int = 50,
        prediction_steps: int = 1,
        overlap_ratio: float = 0.5,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        normalize: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """
        Initialize the TimeSeriesDataModule.

        Args:
            data_dir: Directory containing data files or specific data file path
            batch_size: Batch size for DataLoaders
            sequence_length: Length of input sequences
            prediction_steps: Number of steps to predict ahead
            overlap_ratio: Ratio of overlap between consecutive sequences
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            normalize: Whether to normalize the data
            num_workers: Number of workers for DataLoaders
            pin_memory: Whether to pin memory for DataLoaders
        """
        super().__init__()

        assert (
            abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        ), "Train, val, and test ratios must sum to 1.0"

        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.overlap_ratio = overlap_ratio
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.normalize = normalize
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for different stages."""
        if stage == "fit" or stage is None:
            # Load full dataset first to get split indices
            full_dataset = TimeSeriesDataset(
                data_path=self.data_dir,
                sequence_length=self.sequence_length,
                prediction_steps=self.prediction_steps,
                overlap_ratio=self.overlap_ratio,
                normalize=self.normalize,
            )

            # Split data
            total_len = len(full_dataset.data)
            train_len = int(total_len * self.train_ratio)
            val_len = int(total_len * self.val_ratio)

            # Create datasets for each split
            self.train_dataset = self._create_split_dataset(
                full_dataset.data[:train_len]
            )
            self.val_dataset = self._create_split_dataset(
                full_dataset.data[train_len : train_len + val_len]
            )

        if stage == "test" or stage is None:
            # Create test dataset from the remaining data
            full_dataset = TimeSeriesDataset(
                data_path=self.data_dir,
                sequence_length=self.sequence_length,
                prediction_steps=self.prediction_steps,
                overlap_ratio=self.overlap_ratio,
                normalize=self.normalize,
            )

            total_len = len(full_dataset.data)
            train_len = int(total_len * self.train_ratio)
            val_len = int(total_len * self.val_ratio)

            self.test_dataset = self._create_split_dataset(
                full_dataset.data[train_len + val_len :]
            )

    def _create_split_dataset(self, data: np.ndarray) -> TimeSeriesDataset:
        """Create a dataset from a data split."""
        # Save data temporarily and create dataset
        temp_path = f"/tmp/temp_data_{id(data)}.npy"
        np.save(temp_path, data)

        dataset = TimeSeriesDataset(
            data_path=temp_path,
            sequence_length=self.sequence_length,
            prediction_steps=self.prediction_steps,
            overlap_ratio=self.overlap_ratio,
            normalize=False,  # Already normalized in the full dataset
        )

        # Clean up temp file
        os.remove(temp_path)

        # Override the data directly
        dataset.data = data
        dataset.sequences = dataset._create_sequences()

        return dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_feature_dim(self) -> int:
        """Get the feature dimension of the data."""
        if self.train_dataset is None:
            # Create a temporary dataset to get feature dim
            temp_dataset = TimeSeriesDataset(
                data_path=self.data_dir,
                sequence_length=self.sequence_length,
                prediction_steps=self.prediction_steps,
                overlap_ratio=self.overlap_ratio,
                normalize=self.normalize,
            )
            return temp_dataset.get_feature_dim()
        return self.train_dataset.get_feature_dim()


# Specific dataset classes for different simulation types
class RobotManipulationDataset(TimeSeriesDataset):
    """Dataset for robot manipulation tasks (e.g., FetchPickAndPlace)."""

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path, **kwargs)

    def _load_data(self) -> np.ndarray:
        """Load robot manipulation data with specific preprocessing."""
        data = super()._load_data()

        # Robot manipulation specific preprocessing can go here
        # e.g., separate observations, actions, next_observations

        return data


class BouncingBallDataset(TimeSeriesDataset):
    """Dataset for bouncing ball simulations."""

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path, **kwargs)

    def _load_data(self) -> np.ndarray:
        """Load bouncing ball data with specific preprocessing."""
        data = super()._load_data()

        # Bouncing ball specific preprocessing can go here
        # e.g., position, velocity, collision detection

        return data
