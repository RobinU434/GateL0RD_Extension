import os
from typing import Optional
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from project.datasets.timeseries import TimeSeriesDataset


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
        # get mean and var from full dataset to normalize splits consistently
        if self.normalize:
            full_dataset = TimeSeriesDataset(
                data_path=self.data_dir,
                sequence_length=self.sequence_length,
                prediction_steps=self.prediction_steps,
                overlap_ratio=self.overlap_ratio,
                normalize=self.normalize,
            )
            self.data_mean = full_dataset.data_mean
            self.data_std = full_dataset.data_std
            del full_dataset  # free memory

        if stage == "fit" or stage is None:
            # Create datasets for each split
            self.train_dataset = TimeSeriesDataset(
                data_path=Path(self.data_dir) / "train",
                sequence_length=self.sequence_length,
                prediction_steps=self.prediction_steps,
                overlap_ratio=self.overlap_ratio,
                normalize=False,  # Already normalized in the full dataset
            )
            self.val_dataset = TimeSeriesDataset(
                data_path=Path(self.data_dir) / "val",
                sequence_length=self.sequence_length,
                prediction_steps=self.prediction_steps,
                overlap_ratio=self.overlap_ratio,
                normalize=False,  # Already normalized in the full dataset
            )

            if self.normalize:
                self.train_dataset.data_mean = self.data_mean
                self.train_dataset.data_std = self.data_std
                self.train_dataset.normalize_data()

                self.val_dataset.data_mean = self.data_mean
                self.val_dataset.data_std = self.data_std
                self.val_dataset.normalize_data()

        if stage == "test" or stage is None:
            # Create test dataset from the remaining data
            self.test_dataset = TimeSeriesDataset(
                data_path=Path(self.data_dir) / "val",
                sequence_length=self.sequence_length,
                prediction_steps=self.prediction_steps,
                overlap_ratio=self.overlap_ratio,
                normalize=False,  # Already normalized in the full dataset
            )

            if self.normalize:
                self.test_dataset.data_mean = self.data_mean
                self.test_dataset.data_std = self.data_std
                self.test_dataset.normalize_data()

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
