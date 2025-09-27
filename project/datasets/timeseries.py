"""
Time Series Dataset for GateL0RD training.

Implements dataset loading for continuous time series data with discrete events,
following the setup from the GateL0RD paper (https://arxiv.org/pdf/2110.15949).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from project.datasets.data_handler import load_data


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
        prefix_length: int = 0,
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
        self.prefix_length = prefix_length
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.overlap_ratio = overlap_ratio
        self.normalize = normalize
        self.data_key = data_key

        self.data = self._load_data()
        self.sequences = self._create_sequences()

        self.data_mean = np.mean(self.data, axis=0, keepdims=True)
        self.data_std = np.std(self.data, axis=0, keepdims=True) + 1e-8

        if self.normalize:
            self.normalize_data()

    def normalize_data(self):
        """Normalize data to zero mean and unit variance."""
        self.data = (self.data - self.data_mean) / self.data_std

    def _load_data(self) -> np.ndarray:
        """Load data from file(s)."""

        data = load_data(Path(self.data_path), self.data_key)
        data = np.concatenate(list(data.values()), axis=0)
        return data.astype(np.float32)

    def _create_sequences(self) -> List[Tuple[int, int]]:
        """Create sequence start and end indices.

        Assume all sequences have the same length for simplicity.
        """
        sequences = []
        total_length = self.data.shape[1]
        step_size = max(1, int(self.sequence_length * (1 - self.overlap_ratio)))

        for start_idx in range(
            0,
            total_length
            - self.sequence_length
            - self.prediction_steps
            - self.prefix_length
            + 1,
            step_size,
        ):
            end_idx = start_idx + self.sequence_length + self.prefix_length
            sequences.append((start_idx, end_idx))
        return sequences

    def __len__(self) -> int:
        return len(self.sequences) * len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sequence.

        ps = prediction_steps
        sl = sequence_length
        x_0, ... x_[t - 1],   -> prefix sequence
        x_t, ... x_[t + sl - 1] -> input sequence
        x_[t + ps], ... x_[t + sl + ps - 1] -> target sequence

        Returns:
            Dictionary containing:
                - 'input': Input sequence of shape (sequence_length, feature_dim)
                - 'target': Target sequence of shape (prediction_steps, feature_dim)
                - 'sequence_mask': Mask for valid timesteps (all 1s for this implementation)
        """
        sequence_idx = idx // len(self.data)
        data_idx = idx % len(self.data)

        start_idx, end_idx = self.sequences[sequence_idx]

        # prefix sequence
        prefix_seq = self.data[data_idx, start_idx : start_idx + self.prefix_length]

        # Input sequence
        input_seq = self.data[data_idx, start_idx + self.prefix_length : end_idx]

        # Target sequence (next prediction_steps timesteps)
        target_seq = self.data[
            data_idx,
            start_idx + self.prediction_steps : end_idx + self.prediction_steps,
        ]

        # Create mask (all valid for this simple implementation)
        sequence_mask = np.ones((self.sequence_length,), dtype=np.float32)

        return {
            "prefix": torch.from_numpy(prefix_seq),
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
